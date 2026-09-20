"""JEV (Judge & Evaluation via Vectorized-logits) VRAM Manager モジュール

詳細設計書 (JEV-DD-001) 2.4節のリソース保護契約に基づく。
1. 直列キューイング契約: threading.Semaphore(1) による単一バッチ直列実行とOOM防止
2. トークン長リミッター契約: Soft 2,000T (警告) / Hard 4,000T (超過時 PayloadTooLargeError)
"""

from contextlib import contextmanager
import logging
import sys
import threading
from typing import Generator, Optional

from jev.exceptions import PayloadTooLargeError, QueueTimeoutError

logger = logging.getLogger("jev.vram_manager")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class VRAMManager:
    """単一GPU環境におけるVRAMリソース保護およびリクエスト調停を行うマネージャー (JEV-DD-001 2.4)"""

    def __init__(
        self,
        soft_char_limit: int = 6000,
        hard_char_limit: int = 12000,
        soft_token_limit: int = 2000,
        hard_token_limit: int = 4000,
        default_timeout_sec: float = 60.0,
    ):
        """
        Args:
            soft_char_limit: 推奨ソフト上限文字数 (約2,000トークン相当)
            hard_char_limit: 強制遮断ハード上限文字数 (約4,000トークン相当)
            soft_token_limit: 推奨ソフト上限トークン数
            hard_token_limit: 強制遮断ハード上限トークン数
            default_timeout_sec: 直列キュー獲得のタイムアウト秒数
        """
        self.soft_char_limit = soft_char_limit
        self.hard_char_limit = hard_char_limit
        self.soft_token_limit = soft_token_limit
        self.hard_token_limit = hard_token_limit
        self.default_timeout_sec = default_timeout_sec

        # 直列実行を保証するセマフォ (容量1)
        self._semaphore = threading.Semaphore(1)
        self._lock = threading.Lock()
        self._waiting_count = 0
        self._is_running = False
        self._total_processed = 0

    def estimate_tokens(self, text: str) -> int:
        """テキストのトークン数を概算する (日本語・記号・英数混在の安全マージン換算)

        ※厳密なモデル別BPEトークナイザがない環境でも安全に判定できるよう、
          保守的な係数（日本語約2.0〜2.5文字/トークン）で概算。
        """
        if not text:
            return 0
        # 文字数ベースで保守的に見積もり（日本語中心テキストで約2.5文字/トークン）
        return int(len(text) / 2.5) + 1

    def validate_payload_limits(self, text: str) -> None:
        """トークン長および文字数リミッター契約の検証 (Issue #6 実証仕様)

        Raises:
            PayloadTooLargeError: ハードリミット (12,000文字 / 4,000トークン) 超過時
        """
        char_len = len(text)
        est_tokens = self.estimate_tokens(text)

        if char_len > self.hard_char_limit or est_tokens > self.hard_token_limit:
            msg = (
                f"Context exceeds hard limit: {char_len} chars (limit: {self.hard_char_limit}), "
                f"approx {est_tokens} tokens (limit: {self.hard_token_limit})"
            )
            logger.error(msg)
            raise PayloadTooLargeError(msg)

        if char_len > self.soft_char_limit or est_tokens > self.soft_token_limit:
            logger.warning(
                "Context exceeds soft limit: %d chars (soft limit: %d), approx %d tokens. "
                "Latency may increase.",
                char_len,
                self.soft_char_limit,
                est_tokens,
            )

    @contextmanager
    def acquire(self, timeout: Optional[float] = None) -> Generator[None, None, None]:
        """直列実行セマフォを獲得するコンテキストマネージャー (Issue #8 実証仕様)

        Args:
            timeout: ロック獲得待機タイムアウト秒数。Noneの場合は default_timeout_sec

        Raises:
            QueueTimeoutError: 指定秒数内にロックを獲得できなかった場合
        """
        timeout_val = self.default_timeout_sec if timeout is None else timeout

        with self._lock:
            self._waiting_count += 1

        acquired = False
        try:
            acquired = self._semaphore.acquire(timeout=timeout_val)
            if not acquired:
                msg = f"VRAM Queue acquisition timed out after {timeout_val} seconds. Queue size: {self._waiting_count}"
                logger.error(msg)
                raise QueueTimeoutError(msg)

            with self._lock:
                self._waiting_count -= 1
                self._is_running = True

            yield
        finally:
            if acquired:
                with self._lock:
                    self._is_running = False
                    self._total_processed += 1
                self._semaphore.release()
            else:
                with self._lock:
                    self._waiting_count -= 1

    def get_metrics(self) -> dict:
        """現在のキュー稼働状態メトリクスを取得する"""
        with self._lock:
            return {
                "waiting_queue_count": self._waiting_count,
                "is_running": self._is_running,
                "total_processed": self._total_processed,
            }


# シングルトンインスタンス
default_vram_manager = VRAMManager()
