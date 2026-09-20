"""JEV (Judge & Evaluation via Vectorized-logits) 例外定義モジュール

詳細設計書 (JEV-DD-001) 第5章のエラー処理・安全回路仕様に基づく型安全な例外群。
"""


class JevError(Exception):
    """JEV システムの基底例外"""

    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class PayloadTooLargeError(JevError):
    """コンテキスト長がハードリミット（4,000トークン / 12,000文字）を超過した場合の例外 (JEV-DD-001 2.4)"""

    def __init__(self, message: str = "Context exceeds hard limit (4,000 tokens / 12,000 characters)"):
        super().__init__(message, status_code=413)


class QueueTimeoutError(JevError):
    """VRAM Manager の直列キュー待機時間がタイムアウト（デフォルト60秒）した場合の例外 (JEV-DD-001 5章)"""

    def __init__(self, message: str = "VRAM Queue lock acquisition timed out"):
        super().__init__(message, status_code=504)


class InconclusiveVerdictError(JevError):
    """スワップ検証（A/B反転推論）で判定結果が不一致となり偽の判定を防止するための例外 (JEV-DD-001 5章)"""

    def __init__(self, message: str = "Swap verification inconclusive: conflicting verdicts detected"):
        super().__init__(message, status_code=422)


class TokenNotFoundError(JevError):
    """対象記号が上位Top-Logprobsに存在しない場合の例外（デフォルト極小値補完前の検知用）"""

    def __init__(self, message: str = "Target token not found in top logprobs candidates"):
        super().__init__(message, status_code=404)


class BackendConnectionError(JevError):
    """推論バックエンド（Ollama / OpenAI互換API）への接続失敗・通信障害"""

    def __init__(self, message: str = "Failed to connect to inference backend"):
        super().__init__(message, status_code=502)
