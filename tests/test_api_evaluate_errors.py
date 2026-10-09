"""/evaluate の非 SUCCESS 応答テスト (Issue #21)

バックエンド障害や判定不能のとき、中立値(noul=0.5 など)を作らず、
status と error_message を返すことを検証する。Ollama は不要（偽バックエンド）。
"""

from typing import Dict, Iterator, List, Tuple

from fastapi.testclient import TestClient
import pytest

from jev import BackendConnectionError, JudgePipeline, ZeroDecodeClient
from jev.server.api.v1.judge import get_pipeline
from jev.server.app import app
from jev.zero_decode import InferenceBackend


class _Backend(InferenceBackend):
    def __init__(self, logprobs: Dict[str, float], fail: bool = False):
        self._logprobs = logprobs
        self._fail = fail

    @property
    def is_cloud(self) -> bool:
        return False

    def forward(
        self, model: str, messages: List[Dict[str, str]], top_logprobs: int = 10
    ) -> Tuple[Dict[str, float], float]:
        if self._fail:
            raise BackendConnectionError("backend down")
        return self._logprobs, 1.0


def _client(backend: InferenceBackend) -> TestClient:
    pipeline = JudgePipeline(client=ZeroDecodeClient(backend=backend))
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clear_overrides() -> Iterator[None]:
    yield
    app.dependency_overrides.clear()


_PAYLOAD = {
    "state": "x",
    "questions": {
        "n": {"type": "noul"},
        "c": {"type": "choice", "criteria": ["a", "b"]},
        "s": {"type": "score"},
        "m": {"type": "multilabel", "criteria": ["x"]},
    },
}


def test_backend_failure_does_not_fabricate_neutral_values():
    resp = _client(_Backend({}, fail=True)).post("/api/v1/evaluate", json=_PAYLOAD)
    assert resp.status_code == 200, resp.text
    answers = resp.json()["answers"]
    for name in ("n", "c", "s", "m"):
        assert answers[name]["status"] == "ERROR"
        assert "backend down" in answers[name]["error_message"]
    assert answers["n"]["noul"] is None
    assert answers["n"]["margin"] is None
    assert answers["c"]["choice"] is None
    assert answers["s"]["score"] is None


def test_inconclusive_noul_has_no_probability():
    resp = _client(_Backend({"The": -0.1})).post("/api/v1/evaluate", json=_PAYLOAD)
    assert resp.status_code == 200, resp.text
    noul = resp.json()["answers"]["n"]
    assert noul["status"] == "INCONCLUSIVE"
    assert noul["noul"] is None
    assert noul["error_message"]


def test_success_response_keeps_values_and_has_null_error_message():
    import math

    backend = _Backend({"Yes": math.log(0.9), "No": math.log(0.1), "A": -0.1, "B": -3.0})
    payload = {"state": "x", "questions": {"n": {"type": "noul"}}}
    noul = _client(backend).post("/api/v1/evaluate", json=payload).json()["answers"]["n"]
    assert noul["status"] == "SUCCESS"
    assert noul["noul"] == pytest.approx(0.9, abs=1e-3)
    assert noul["error_message"] is None
