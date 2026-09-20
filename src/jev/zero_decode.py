"""JEV (Judge & Evaluation via Vectorized-logits) Zero-Decode Inference モジュール

詳細設計書 (JEV-DD-001) 2.3節のバックエンド接続仕様に基づく。
1サイクル推論（max_tokens=1, logprobs=True, top_logprobs=10）を実行し、
生成ループを行わずに最初の1トークンのTop-Logprobsを高速抽出する。
"""

import logging
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from jev.exceptions import BackendConnectionError

logger = logging.getLogger("jev.zero_decode")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class ZeroDecodeClient:
    """Ollama / OpenAI互換エンドポイントに対して1サイクル推論を実行するクライアント (JEV-DD-001 2.3)"""

    DEFAULT_ENDPOINT = "http://localhost:11434/v1/chat/completions"

    def __init__(self, endpoint_url: Optional[str] = None, timeout_sec: float = 30.0):
        """
        Args:
            endpoint_url: OpenAI互換エンドポイントURL。None時は DEFAULT_ENDPOINT
            timeout_sec: HTTPリクエストタイムアウト秒数
        """
        self.endpoint_url = endpoint_url or self.DEFAULT_ENDPOINT
        self.timeout_sec = timeout_sec

    def forward(
        self,
        model: str,
        messages: List[Dict[str, str]],
        top_logprobs: int = 10,
    ) -> Tuple[Dict[str, float], float]:
        """1サイクル推論 (Forwardパス) を実行し、Top-Logprobs辞書と所要時間を返却する

        Args:
            model: モデル識別名 (例: 'qwen3:8b', 'phi4-mini:latest')
            messages: プロンプトメッセージ列
            top_logprobs: 取得する上位候補数 (デフォルト10)

        Returns:
            Tuple[token_to_logprob_dict, latency_ms]

        Raises:
            BackendConnectionError: 通信エラーまたはバックエンドが非200を返却した場合
        """
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": top_logprobs,
        }

        t_start = time.perf_counter()
        try:
            resp = requests.post(
                self.endpoint_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout_sec,
            )
        except Exception as e:
            msg = f"Failed to connect to inference backend at {self.endpoint_url}: {e}"
            logger.error(msg)
            raise BackendConnectionError(msg) from e

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        if resp.status_code != 200:
            msg = f"Inference backend returned HTTP {resp.status_code}: {resp.text}"
            logger.error(msg)
            raise BackendConnectionError(msg)

        data = resp.json()
        logprobs_dict = self._parse_logprobs(data)
        return logprobs_dict, latency_ms

    def _parse_logprobs(self, response_data: Dict[str, Any]) -> Dict[str, float]:
        """レスポンスJSONから top_logprobs をパースして {token: logprob} に変換する"""
        choices = response_data.get("choices", [])
        if not choices:
            return {}

        choice = choices[0]
        logprobs_obj = choice.get("logprobs")
        if not logprobs_obj:
            return {}

        content_list = logprobs_obj.get("content", [])
        if not content_list:
            return {}

        top_candidates = content_list[0].get("top_logprobs", [])
        result = {}
        for item in top_candidates:
            token = item.get("token")
            logprob = item.get("logprob")
            if token is not None and logprob is not None:
                result[token] = float(logprob)

        return result
