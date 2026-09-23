"""JEV (Judge & Evaluation via Vectorized-logits) Zero-Decode Inference モジュール

詳細設計書 (JEV-DD-001) 2.3節のバックエンド接続仕様に基づく。
1サイクル推論（max_tokens=1, logprobs=True, top_logprobs=10）を実行し、
生成ループを行わずに最初の1トークンのTop-Logprobsを高速抽出する。
ローカル (Ollama) および クラウド (OpenAI互換) のマルチバックエンドをサポート (Issue #15)。
"""

from abc import ABC, abstractmethod
import logging
import os
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


def _parse_logprobs(response_data: Dict[str, Any]) -> Dict[str, float]:
    """レスポンスJSONから top_logprobs をパースして {token: logprob} に変換する (Ollama/OpenAI共通)"""
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


class InferenceBackend(ABC):
    """Zero-Decode 推論バックエンド抽象基底クラス (Issue #15)"""

    @abstractmethod
    def forward(
        self,
        model: str,
        messages: List[Dict[str, str]],
        top_logprobs: int = 10,
    ) -> Tuple[Dict[str, float], float]:
        """1サイクル推論 (Forwardパス) を実行し、Top-Logprobs辞書と所要時間(ms)を返却する"""
        pass

    @property
    @abstractmethod
    def is_cloud(self) -> bool:
        """ローカルGPU管理 (VRAMManager) をバイパスすべきクラウドバックエンドか"""
        pass


class OllamaBackend(InferenceBackend):
    """ローカル Ollama 用 Zero-Decode バックエンド"""

    DEFAULT_ENDPOINT = "http://localhost:11434/v1/chat/completions"

    def __init__(self, endpoint_url: Optional[str] = None, timeout_sec: float = 30.0):
        self.endpoint_url = (
            endpoint_url
            or os.environ.get("OLLAMA_ENDPOINT")
            or os.environ.get("JEV_OLLAMA_ENDPOINT")
            or self.DEFAULT_ENDPOINT
        )
        self.timeout_sec = timeout_sec

    @property
    def is_cloud(self) -> bool:
        return False

    def forward(
        self,
        model: str,
        messages: List[Dict[str, str]],
        top_logprobs: int = 10,
    ) -> Tuple[Dict[str, float], float]:
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
            msg = f"Failed to connect to local Ollama at {self.endpoint_url}: {e}"
            logger.error(msg)
            raise BackendConnectionError(msg) from e

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        if resp.status_code != 200:
            msg = f"Ollama backend returned HTTP {resp.status_code}: {resp.text}"
            logger.error(msg)
            raise BackendConnectionError(msg)

        data = resp.json()
        logprobs_dict = _parse_logprobs(data)
        return logprobs_dict, latency_ms


class OpenAIBackend(InferenceBackend):
    """クラウド OpenAI 互換用 Zero-Decode バックエンド (Issue #15)"""

    DEFAULT_BASE_URL = "https://api.openai.com/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_sec: float = 30.0,
    ):
        self.api_key = (
            api_key
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("OPENJEV_API_KEY")
            or os.environ.get("CEREBRAS_API_KEY")
            or os.environ.get("GROQ_API_KEY")
            or ""
        )
        base = (
            base_url
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENJEV_BASE_URL")
            or self.DEFAULT_BASE_URL
        )
        self.endpoint_url = f"{base.rstrip('/')}/chat/completions"
        self.timeout_sec = timeout_sec

    @property
    def is_cloud(self) -> bool:
        return True

    def forward(
        self,
        model: str,
        messages: List[Dict[str, str]],
        top_logprobs: int = 10,
    ) -> Tuple[Dict[str, float], float]:
        if not self.api_key:
            raise BackendConnectionError(
                "OpenAI API key is missing. Set OPENAI_API_KEY in environment or pass api_key."
            )

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": top_logprobs,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        t_start = time.perf_counter()
        try:
            resp = requests.post(
                self.endpoint_url,
                json=payload,
                headers=headers,
                timeout=self.timeout_sec,
            )
        except Exception as e:
            msg = f"Failed to connect to cloud backend at {self.endpoint_url}: {e}"
            logger.error(msg)
            raise BackendConnectionError(msg) from e

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        if resp.status_code != 200:
            msg = f"Cloud backend returned HTTP {resp.status_code}: {resp.text}"
            logger.error(msg)
            raise BackendConnectionError(msg)

        data = resp.json()
        logprobs_dict = _parse_logprobs(data)
        return logprobs_dict, latency_ms


class ZeroDecodeClient:
    """Ollama / OpenAI互換エンドポイントに対して1サイクル推論を実行するファサード (JEV-DD-001 2.3)"""

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        timeout_sec: float = 30.0,
        backend: Optional[InferenceBackend] = None,
    ):
        """
        Args:
            endpoint_url: 互換用エンドポイントURL (None時はデフォルト)
            timeout_sec: HTTPリクエストタイムアウト秒数
            backend: 明示的な推論バックエンド。省略時はOllamaBackend
        """
        if backend is not None:
            self.backend = backend
        else:
            self.backend = OllamaBackend(endpoint_url=endpoint_url, timeout_sec=timeout_sec)
        self.timeout_sec = timeout_sec

    @property
    def is_cloud(self) -> bool:
        """現在のバックエンドがクラウドであるか"""
        return self.backend.is_cloud

    def forward(
        self,
        model: str,
        messages: List[Dict[str, str]],
        top_logprobs: int = 10,
    ) -> Tuple[Dict[str, float], float]:
        """1サイクル推論 (Forwardパス) を実行し、Top-Logprobs辞書と所要時間を返却する"""
        return self.backend.forward(model, messages, top_logprobs=top_logprobs)

