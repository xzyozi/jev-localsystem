"""REST API におけるリクエスト指定値の検証・反映テスト (Issue #19)

get_pipeline を偽バックエンドに差し替えるため、Ollama は不要。
"""

import math
from typing import Dict, Iterator, List, Tuple

from fastapi.testclient import TestClient
import pytest

from jev import JudgePipeline, ZeroDecodeClient
from jev.server.api.v1.judge import get_pipeline
from jev.server.app import app
from jev.zero_decode import InferenceBackend


class _FixedBackend(InferenceBackend):
    def __init__(self, logprobs: Dict[str, float]):
        self._logprobs = logprobs

    @property
    def is_cloud(self) -> bool:
        return False

    def forward(
        self, model: str, messages: List[Dict[str, str]], top_logprobs: int = 10
    ) -> Tuple[Dict[str, float], float]:
        return self._logprobs, 1.0


def _override(logprobs: Dict[str, float]) -> TestClient:
    pipeline = JudgePipeline(client=ZeroDecodeClient(backend=_FixedBackend(logprobs)))
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clear_overrides() -> Iterator[None]:
    yield
    app.dependency_overrides.clear()


_P70 = {"Yes": math.log(0.7), "No": math.log(0.3)}  # Sigmoid 確率 0.7


def test_evaluate_multilabel_threshold_is_applied():
    client = _override(_P70)

    def run(threshold: float):
        payload = {
            "state": "x",
            "questions": {"c": {"type": "multilabel", "criteria": ["Security"], "threshold": threshold}},
        }
        resp = client.post("/api/v1/evaluate", json=payload)
        assert resp.status_code == 200, resp.text
        return resp.json()["answers"]["c"]

    assert run(0.5)["matched_labels"] == ["Security"]
    assert run(0.9)["matched_labels"] == []


def test_multilabel_endpoint_threshold_is_applied():
    client = _override(_P70)
    resp = client.post(
        "/api/v1/multilabel", json={"context_text": "x", "labels": ["Security"], "threshold": 0.9}
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["verdict"] == []


def test_evaluate_score_levels_follow_criteria_length():
    client = _override({"1": math.log(0.1), "2": math.log(0.2), "3": math.log(0.7)})
    payload = {
        "state": "x",
        "questions": {"s": {"type": "score", "criteria": ["Low", "Medium", "High"]}},
    }
    resp = client.post("/api/v1/evaluate", json=payload)
    assert resp.status_code == 200, resp.text
    answer = resp.json()["answers"]["s"]
    assert answer["status"] == "SUCCESS"
    assert list(answer["probabilities"]) == ["1", "2", "3"]
    assert answer["score"] == pytest.approx(2.6, abs=1e-3)


def test_evaluate_choice_over_limit_is_422():
    client = _override(_P70)
    payload = {
        "state": "x",
        "questions": {"q": {"type": "choice", "criteria": [f"L{i}" for i in range(9)]}},
    }
    assert client.post("/api/v1/evaluate", json=payload).status_code == 422


def test_choice_endpoint_over_limit_is_422():
    client = _override(_P70)
    resp = client.post("/api/v1/choice", json={"context_text": "x", "labels": [f"L{i}" for i in range(9)]})
    assert resp.status_code == 422


def test_judge_endpoint_choice_over_limit_is_422():
    client = _override(_P70)
    resp = client.post(
        "/api/v1/judge",
        json={"task_type": "choice", "context_text": "x", "labels": [f"L{i}" for i in range(9)]},
    )
    assert resp.status_code == 422
