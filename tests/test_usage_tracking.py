"""バックエンド usage（トークン数）取り込み・合算テストスイート (Issue #70)."""

from typing import Dict, List, Optional, Sequence, Tuple
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
import pytest

from jev import JudgePipeline, ZeroDecodeClient
from jev.server.api.v1.judge import get_pipeline
from jev.server.app import app
from jev.zero_decode import InferenceBackend, OllamaBackend, OpenAIBackend, UsageDict, merge_usage


class _UsageBackend(InferenceBackend):
    """テスト用の固定 logprobs と usage を返すバックエンド。"""

    def __init__(self, usage_per_call: Optional[UsageDict]) -> None:
        self._usage = usage_per_call
        self.calls = 0

    @property
    def is_cloud(self) -> bool:
        return False

    def forward(
        self,
        model: str,
        messages: List[Dict[str, str]],
        top_logprobs: int = 10,
    ) -> Tuple[Dict[str, float], float]:
        self.calls += 1
        return {
            "Yes": -0.05,
            "No": -3.0,
            "A": -0.05,
            "B": -3.0,
            "1": -3.0,
            "2": -3.0,
            "3": -2.0,
            "4": -0.2,
            "5": -3.0,
        }, 5.0

    def forward_with_usage(
        self,
        model: str,
        messages: List[Dict[str, str]],
        target_tokens: Optional[Sequence[str]] = None,
        top_logprobs: int = 10,
    ) -> Tuple[Dict[str, float], float, Optional[UsageDict]]:
        logprobs, latency_ms = self.forward(model, messages, top_logprobs=top_logprobs)
        return logprobs, latency_ms, self._usage


class _LegacyNoUsageBackend(InferenceBackend):
    """forward のみ実装した従来互換バックエンド（usage なし）。"""

    @property
    def is_cloud(self) -> bool:
        return False

    def forward(
        self,
        model: str,
        messages: List[Dict[str, str]],
        top_logprobs: int = 10,
    ) -> Tuple[Dict[str, float], float]:
        return {"Yes": -0.1, "No": -2.5}, 2.0


def test_merge_usage_handles_none_and_partial_values() -> None:
    assert merge_usage([None, None]) is None
    assert merge_usage([{"input_tokens": 10, "output_tokens": 1}, None]) == {
        "input_tokens": 10,
        "output_tokens": 1,
    }
    assert merge_usage(
        [
            {"input_tokens": 10, "output_tokens": None},
            {"input_tokens": 15, "output_tokens": 2},
        ]
    ) == {"input_tokens": 25, "output_tokens": 2}


@patch("requests.post")
def test_ollama_backend_extracts_usage_and_preserves_forward_tuple(mock_post: MagicMock) -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"logprobs": {"content": [{"top_logprobs": [{"token": "Yes", "logprob": -0.05}]}]}}],
        "usage": {
            "prompt_tokens": 42,
            "prompt_tokens_details": {"cached_tokens": 12},
            "completion_tokens": 1,
            "total_tokens": 43,
        },
    }
    mock_post.return_value = mock_resp

    backend = OllamaBackend()
    lp, _, usage = backend.forward_with_usage("qwen3:8b", [{"role": "user", "content": "hi"}])
    assert lp == {"Yes": -0.05}
    assert usage == {"input_tokens": 42, "output_tokens": 1}

    # 既存の 2要素タプル forward() 契約を維持
    lp2, latency2 = backend.forward("qwen3:8b", [{"role": "user", "content": "hi"}])
    assert lp2 == {"Yes": -0.05}
    assert latency2 >= 0.0


@patch("requests.post")
def test_openai_backend_usage_and_missing_usage_handling(mock_post: MagicMock) -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"logprobs": {"content": [{"top_logprobs": [{"token": "Yes", "logprob": -0.1}]}]}}],
        "usage": {"input_tokens": 18, "output_tokens": 1},
    }
    mock_post.return_value = mock_resp

    backend = OpenAIBackend(api_key="sk-test")
    _, _, usage = backend.forward_with_usage("gpt-4o-mini", [{"role": "user", "content": "hi"}])
    assert usage == {"input_tokens": 18, "output_tokens": 1}

    # usage が未含または不正型でも失敗しない
    mock_resp.json.return_value = {
        "choices": [{"logprobs": {"content": [{"top_logprobs": [{"token": "Yes", "logprob": -0.1}]}]}}],
        "usage": {"prompt_tokens": True},
    }
    _, _, missing_usage = backend.forward_with_usage("gpt-4o-mini", [{"role": "user", "content": "hi"}])
    assert missing_usage is None


def test_pipeline_accumulates_usage_for_swap_and_multilabel() -> None:
    backend = _UsageBackend({"input_tokens": 100, "output_tokens": 1})
    pipeline = JudgePipeline(client=ZeroDecodeClient(backend=backend))

    noul_res = pipeline.judge_noul("テスト", "基準")
    assert noul_res.usage == {"input_tokens": 100, "output_tokens": 1}

    swap_res = pipeline.judge_choice("テスト", labels=["L1", "L2"], swap_verify=True)
    assert swap_res.usage == {"input_tokens": 200, "output_tokens": 2}

    ml_res = pipeline.judge_multilabel("テスト", labels=["A", "B", "C"])
    assert ml_res.usage == {"input_tokens": 300, "output_tokens": 3}


def test_legacy_backend_keeps_usage_none_in_pipeline_and_empty_in_evaluate() -> None:
    pipeline = JudgePipeline(client=ZeroDecodeClient(backend=_LegacyNoUsageBackend()))
    res = pipeline.judge_noul("テスト", "基準")
    assert res.status == "SUCCESS"
    assert res.usage is None

    app.dependency_overrides[get_pipeline] = lambda: pipeline
    try:
        client = TestClient(app)
        resp = client.post(
            "/api/v1/evaluate",
            json={
                "state": "テスト",
                "questions": {
                    "q1": {"type": "noul", "instructions": "基準"},
                },
            },
        )
        assert resp.status_code == 200
        assert resp.json()["usage"] == {"input_tokens": None, "output_tokens": None}
    finally:
        app.dependency_overrides.clear()


def test_api_evaluate_and_judge_return_aggregated_usage() -> None:
    backend = _UsageBackend({"input_tokens": 50, "output_tokens": 1})
    pipeline = JudgePipeline(client=ZeroDecodeClient(backend=backend))

    app.dependency_overrides[get_pipeline] = lambda: pipeline
    try:
        client = TestClient(app)

        judge_resp = client.post(
            "/api/v1/judge",
            json={"task_type": "noul", "context_text": "テスト", "rule_definition": "基準"},
        )
        assert judge_resp.status_code == 200
        assert judge_resp.json()["usage"] == {"input_tokens": 50, "output_tokens": 1}

        eval_resp = client.post(
            "/api/v1/evaluate",
            json={
                "state": "テスト文章",
                "swap_verify": True,
                "questions": {
                    "q_noul": {"type": "noul", "instructions": "基準"},  # 1 call
                    "q_choice": {"type": "choice", "criteria": ["A案", "B案"]},  # 2 calls (swap)
                    "q_ml": {"type": "multilabel", "criteria": ["L1", "L2", "L3"]},  # 3 calls
                },
            },
        )
        assert eval_resp.status_code == 200
        # 計 6 calls -> input=300, output=6
        assert eval_resp.json()["usage"] == {"input_tokens": 300, "output_tokens": 6}
    finally:
        app.dependency_overrides.clear()


@pytest.mark.ollama
def test_ollama_evaluate_reports_real_token_usage(target_model: str) -> None:
    """実機 Ollama 経由の /api/v1/evaluate が正の入力・出力トークン数を返すことを確認する。"""
    client = TestClient(app)
    resp = client.post(
        "/api/v1/evaluate",
        json={
            "state": "本番環境へのログインには多要素認証(MFA)が必須です。",
            "model": target_model,
            "questions": {
                "mfa": {"type": "noul", "instructions": "MFAは必須ですか？"},
                "topics": {"type": "multilabel", "criteria": ["Authentication", "Billing"]},
            },
        },
    )
    assert resp.status_code == 200
    usage = resp.json()["usage"]
    assert isinstance(usage["input_tokens"], int) and usage["input_tokens"] > 0
    assert isinstance(usage["output_tokens"], int) and usage["output_tokens"] >= 3
