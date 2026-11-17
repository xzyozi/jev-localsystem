"""タスク種別レジストリ (Issue #72 / 23-C) の契約・互換性・拡張性テスト

- 組み込み 4 タスク (noul, choice, score, multilabel) が JudgePipeline.judge() と
  /api/v1/evaluate で従来通りに動作する
- 未知の task_type は JudgeRequestDTO で ValueError となり、/api/v1/judge で HTTP 422 を返す
- 隔離された TaskRegistry(include_builtins=True) へカスタムタスクを登録し、コアファイルを変更せず E2E 実行できる

Ollama 実機には依存しない（モックバックエンドのみ使用）。
"""

import math
import time
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from fastapi.testclient import TestClient
from pydantic import ValidationError
import pytest

from jev import (
    BackendConnectionError,
    JudgePipeline,
    JudgeRequestDTO,
    JudgeResponseDTO,
    TaskDefinition,
    TaskHandler,
    TaskRegistry,
    ZeroDecodeClient,
    default_task_registry,
)
from jev.dto import EvaluateRequestDTO
from jev.server.api.v1.judge import get_pipeline
from jev.server.app import app
from jev.zero_decode import InferenceBackend, UsageDict


class _MockBackend(InferenceBackend):
    """タスクごとの応答パターンを切り替えられるテスト用バックエンド"""

    def __init__(
        self,
        logprobs: Optional[Dict[str, float]] = None,
        usage: Optional[UsageDict] = None,
        fail: bool = False,
    ) -> None:
        self._logprobs = logprobs or {}
        self._usage = usage
        self._fail = fail
        self.calls: List[Tuple[str, List[Dict[str, str]], Optional[List[str]]]] = []

    @property
    def is_cloud(self) -> bool:
        return False

    def forward(
        self,
        model: str,
        messages: List[Dict[str, str]],
        top_logprobs: int = 10,
    ) -> Tuple[Dict[str, float], float]:
        return self._logprobs, 1.0

    def forward_with_usage(
        self,
        model: str,
        messages: List[Dict[str, str]],
        top_logprobs: int = 10,
        target_tokens: Optional[Sequence[str]] = None,
    ) -> Tuple[Dict[str, float], float, Optional[UsageDict]]:
        if self._fail:
            raise BackendConnectionError("mock backend unreachable")
        self.calls.append((model, messages, list(target_tokens) if target_tokens is not None else None))
        return self._logprobs, 1.0, self._usage


@pytest.fixture(autouse=True)
def _clear_overrides() -> Iterator[None]:
    yield
    app.dependency_overrides.clear()


# ==========================================
# 1. レジストリの基本契約・分離性テスト
# ==========================================


def test_default_registry_contains_builtin_tasks() -> None:
    assert set(default_task_registry.list_tasks()) == {"noul", "choice", "score", "multilabel"}
    for name in ("noul", "choice", "score", "multilabel"):
        assert default_task_registry.has(name)
        assert isinstance(default_task_registry.get(name), TaskHandler)


def test_isolated_registry_options() -> None:
    empty = TaskRegistry(include_builtins=False)
    assert empty.list_tasks() == ()
    assert not empty.has("noul")

    isolated = TaskRegistry(include_builtins=True)
    assert set(isolated.list_tasks()) == {"noul", "choice", "score", "multilabel"}
    isolated.unregister("score")
    assert not isolated.has("score")
    # default_task_registry には影響しない
    assert default_task_registry.has("score")


def test_unknown_task_type_rejected_by_dto_and_api() -> None:
    with pytest.raises(ValueError):
        JudgeRequestDTO(task_type="unknown_task", context_text="テスト")
    with pytest.raises(ValidationError):
        JudgeRequestDTO(task_type="unknown_task", context_text="テスト")

    client = TestClient(app)
    resp = client.post(
        "/api/v1/judge",
        json={"task_type": "unknown_task", "context_text": "テスト"},
    )
    assert resp.status_code == 422

    # DTO 検証をバイパスして pipeline.judge() に渡した場合は ERROR DTO を返す
    pipe = JudgePipeline(client=ZeroDecodeClient(backend=_MockBackend({"Yes": -0.1})))
    req = JudgeRequestDTO.model_construct(task_type="unknown_task", context_text="テスト", labels=[])
    res = pipe.judge(req)
    assert res.status == "ERROR"
    assert res.task_type == "unknown_task"
    assert "Unknown task_type: unknown_task" in (res.error_message or "")


# ==========================================
# 2. 組み込み 4 タスクの互換性テスト (Pipeline & /api/v1/evaluate)
# ==========================================


def test_builtin_tasks_via_pipeline_judge() -> None:
    usage = {"input_tokens": 10, "output_tokens": 1}

    # Noul
    noul_pipe = JudgePipeline(
        client=ZeroDecodeClient(backend=_MockBackend({"Yes": math.log(0.9), "No": math.log(0.1)}, usage=usage))
    )
    res_noul = noul_pipe.judge(JudgeRequestDTO(task_type="noul", context_text="正常終了"))
    assert res_noul.task_type == "noul"
    assert res_noul.status == "SUCCESS"
    assert res_noul.verdict == "Yes"
    assert res_noul.usage == usage

    # Choice (swap_verify=True で 2 回呼び出し & usage 合算)
    choice_pipe = JudgePipeline(
        client=ZeroDecodeClient(backend=_MockBackend({"A": math.log(0.8), "B": math.log(0.2)}, usage=usage))
    )
    res_choice = choice_pipe.judge(
        JudgeRequestDTO(task_type="choice", context_text="分類", labels=["X", "Y"], swap_verify=True)
    )
    assert res_choice.task_type == "choice"
    assert res_choice.status == "INCONCLUSIVE"
    assert res_choice.usage == {"input_tokens": 20, "output_tokens": 2}

    res_choice_ok = choice_pipe.judge(
        JudgeRequestDTO(task_type="choice", context_text="分類", labels=["X", "Y"], swap_verify=False)
    )
    assert res_choice_ok.status == "SUCCESS"
    assert res_choice_ok.verdict == "X"

    # Score (score_levels=3)
    score_pipe = JudgePipeline(
        client=ZeroDecodeClient(
            backend=_MockBackend({"1": math.log(0.1), "2": math.log(0.2), "3": math.log(0.7)}, usage=usage)
        )
    )
    res_score = score_pipe.judge(JudgeRequestDTO(task_type="score", context_text="評価", score_levels=3))
    assert res_score.task_type == "score"
    assert res_score.status == "SUCCESS"
    assert res_score.verdict == pytest.approx(2.6, abs=1e-3)
    assert res_score.usage == usage

    # Multi-Label (2ラベル分で usage 合算)
    ml_pipe = JudgePipeline(
        client=ZeroDecodeClient(backend=_MockBackend({"Yes": math.log(0.8), "No": math.log(0.2)}, usage=usage))
    )
    res_ml = ml_pipe.judge(
        JudgeRequestDTO(task_type="multilabel", context_text="タグ", labels=["Sec", "DB"], threshold=0.5)
    )
    assert res_ml.task_type == "multilabel"
    assert res_ml.status == "SUCCESS"
    assert res_ml.verdict == ["Sec", "DB"]
    assert res_ml.usage == {"input_tokens": 20, "output_tokens": 2}


def test_builtin_tasks_via_api_evaluate() -> None:
    logprobs = {
        "Yes": math.log(0.85),
        "No": math.log(0.15),
        "A": math.log(0.9),
        "B": math.log(0.1),
        "1": math.log(0.1),
        "2": math.log(0.2),
        "3": math.log(0.7),
    }
    usage = {"input_tokens": 5, "output_tokens": 1}
    pipe = JudgePipeline(client=ZeroDecodeClient(backend=_MockBackend(logprobs, usage=usage)))
    app.dependency_overrides[get_pipeline] = lambda: pipe
    client = TestClient(app)

    payload = {
        "state": {"msg": "Invoice charged twice"},
        "questions": {
            "dept": {"type": "choice", "criteria": {"billing": "Billing", "support": "Support"}},
            "urgency": {"type": "score", "criteria": ["Low", "Mid", "High"]},
            "angry": {"type": "noul", "instructions": "Angry?", "criteria": {"true": "yes", "false": "no"}},
            "flag": {"type": "boolean", "instructions": "Is boolean alias working?"},
            "tags": {"type": "multilabel", "criteria": ["Billing", "Bug"], "threshold": 0.5},
        },
    }
    resp = client.post("/api/v1/evaluate", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data["meta"]["total_questions"] == 5
    answers = data["answers"]
    assert answers["dept"]["type"] == "choice"
    assert answers["dept"]["choice"] == "billing"
    assert answers["dept"]["status"] == "SUCCESS"

    assert answers["urgency"]["type"] == "score"
    assert answers["urgency"]["score"] == pytest.approx(2.6, abs=1e-3)
    assert answers["urgency"]["status"] == "SUCCESS"

    assert answers["angry"]["type"] == "noul"
    assert answers["angry"]["verdict"] == "Yes"
    assert answers["angry"]["noul"] == pytest.approx(0.85, abs=1e-3)

    # NoulQuestionDTO.type == "boolean" の互換エイリアス
    assert answers["flag"]["type"] == "noul"
    assert answers["flag"]["verdict"] == "Yes"

    assert answers["tags"]["type"] == "multilabel"
    assert answers["tags"]["matched_labels"] == ["Billing", "Bug"]

    # choice(1) + score(1) + noul(1) + boolean(1) + multilabel(2) = 6 calls
    assert data["usage"] == {"input_tokens": 30, "output_tokens": 6}


# ==========================================
# 3. カスタムタスク登録による E2E 拡張性テスト
# ==========================================


def test_custom_task_on_isolated_registry_executes_end_to_end() -> None:
    registry = TaskRegistry(include_builtins=True)

    def validate_sentiment(request: JudgeRequestDTO) -> None:
        if not request.context_text.strip():
            raise ValueError("sentiment task requires non-empty context_text")

    def execute_sentiment(
        request: JudgeRequestDTO,
        model_name: str,
        t_start: float,
        client: ZeroDecodeClient,
    ) -> JudgeResponseDTO:
        messages = [
            {"role": "system", "content": "Classify sentiment as POS or NEG."},
            {"role": "user", "content": request.context_text},
        ]
        logprobs, _, usage = client.forward_with_usage(model_name, messages, target_tokens=["POS", "NEG"])
        p_pos = math.exp(logprobs.get("POS", -20.0))
        p_neg = math.exp(logprobs.get("NEG", -20.0))
        total = p_pos + p_neg
        verdict = "POS" if p_pos >= p_neg else "NEG"
        latency_ms = (time.perf_counter() - t_start) * 1000.0
        return JudgeResponseDTO(
            task_type="sentiment",
            status="SUCCESS",
            verdict=verdict,
            latency_ms=round(latency_ms, 2),
            confidence=round(max(p_pos, p_neg) / total, 4) if total > 0 else None,
            details={"pos": round(p_pos, 4), "neg": round(p_neg, 4)},
            usage=usage,
        )

    registry.register(
        TaskDefinition(
            name="sentiment",
            validate_request=validate_sentiment,
            execute=execute_sentiment,
        )
    )

    # default_task_registry は汚染されていないこと
    assert not default_task_registry.has("sentiment")
    with pytest.raises(ValueError):
        JudgeRequestDTO(task_type="sentiment", context_text="Great!")

    backend = _MockBackend(
        logprobs={"POS": math.log(0.9), "NEG": math.log(0.1)},
        usage={"input_tokens": 12, "output_tokens": 1},
    )
    pipeline = JudgePipeline(
        default_model="phi4-mini:latest",
        client=ZeroDecodeClient(backend=backend),
        task_registry=registry,
    )

    # 隔離レジストリを ValidationInfo.context に渡して DTO 検証を通過
    req = JudgeRequestDTO.model_validate(
        {"task_type": "sentiment", "context_text": "Awesome product!"},
        context={"task_registry": registry},
    )
    res = pipeline.judge(req)

    assert res.task_type == "sentiment"
    assert res.status == "SUCCESS"
    assert res.verdict == "POS"
    assert res.confidence == pytest.approx(0.9, abs=1e-4)
    assert res.usage == {"input_tokens": 12, "output_tokens": 1}
    assert backend.calls[0][2] == ["POS", "NEG"]

    # 組み込みタスクも隔離レジストリ経由で引き続き動作すること
    res_noul = pipeline.judge_noul("テスト")
    assert res_noul.task_type == "noul"

    # カスタムタスク固有バリデーションエラーは実行時に ERROR DTO へ変換される
    invalid_req = JudgeRequestDTO.model_construct(task_type="sentiment", context_text="   ", labels=[])
    err_res = pipeline.judge(invalid_req)
    assert err_res.status == "ERROR"
    assert "non-empty context_text" in (err_res.error_message or "")

    # BackendConnectionError はカスタムタスクでもそのまま送出される
    failing_pipe = JudgePipeline(
        client=ZeroDecodeClient(backend=_MockBackend(fail=True)),
        task_registry=registry,
    )
    with pytest.raises(BackendConnectionError):
        failing_pipe.judge(req)


def test_custom_task_handler_protocol_and_batch_handler() -> None:
    class _LengthTaskHandler:
        @property
        def name(self) -> str:
            return "char_count"

        def validate_request(self, request: JudgeRequestDTO) -> None:
            return None

        def execute(
            self,
            request: JudgeRequestDTO,
            model_name: str,
            t_start: float,
            client: ZeroDecodeClient,
        ) -> JudgeResponseDTO:
            return JudgeResponseDTO(
                task_type=self.name,
                status="SUCCESS",
                verdict=len(request.context_text),
                latency_ms=round((time.perf_counter() - t_start) * 1000.0, 2),
                usage={"input_tokens": 1, "output_tokens": 0},
            )

    def _batch_char_count(
        q: Any,
        state_str: str,
        request: EvaluateRequestDTO,
        pipeline: JudgePipeline,
        target_model: str,
        custom_client: Optional[ZeroDecodeClient],
    ) -> Tuple[Dict[str, Any], Optional[UsageDict]]:
        req = JudgeRequestDTO.model_construct(task_type="char_count", context_text=state_str, labels=[])
        res = pipeline.judge(req, client=custom_client)
        return {"type": "char_count", "length": res.verdict, "status": res.status}, res.usage

    registry = TaskRegistry(include_builtins=True)
    registry.register(_LengthTaskHandler(), batch_handler=_batch_char_count)

    pipe = JudgePipeline(
        client=ZeroDecodeClient(backend=_MockBackend()),
        task_registry=registry,
    )
    dummy_eval_req = EvaluateRequestDTO(state="hello", questions={})
    answer, usage = pipe.task_registry.execute_batch_question(
        SimpleNamespace(type="char_count"),
        "hello",
        dummy_eval_req,
        pipe,
        "phi4-mini:latest",
        None,
    )
    assert answer == {"type": "char_count", "length": 5, "status": "SUCCESS"}
    assert usage == {"input_tokens": 1, "output_tokens": 0}
