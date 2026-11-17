"""JEV タスク種別レジストリモジュール

タスク種別ごとのリクエスト検証・推論実行・バッチ評価ハンドラを一元管理する。
"""

from dataclasses import dataclass
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from jev.dto import (
    MAX_CHOICE_LABELS,
    MIN_CHOICE_LABELS,
    ChoiceQuestionDTO,
    EvaluateRequestDTO,
    JudgeRequestDTO,
    JudgeResponseDTO,
    MultiLabelQuestionDTO,
    NoulQuestionDTO,
    ScoreQuestionDTO,
)
from jev.prompt_builder import PromptBuilder
from jev.result_mapper import ResultMapper
from jev.zero_decode import UsageDict, ZeroDecodeClient, merge_usage

if TYPE_CHECKING:
    from jev.pipeline import JudgePipeline

RequestValidator = Callable[[JudgeRequestDTO], None]
TaskExecutor = Callable[[JudgeRequestDTO, str, float, ZeroDecodeClient], JudgeResponseDTO]
BatchQuestionHandler = Callable[
    [Any, str, EvaluateRequestDTO, "JudgePipeline", str, Optional[ZeroDecodeClient]],
    Tuple[Dict[str, Any], Optional[UsageDict]],
]


def _noop_validator(request: JudgeRequestDTO) -> None:
    """追加のリクエスト検証が不要なタスク用の既定バリデータ"""
    return None


@runtime_checkable
class TaskHandler(Protocol):
    """タスクレジストリへ登録可能なタスクハンドラのプロトコル"""

    @property
    def name(self) -> str: ...

    def validate_request(self, request: JudgeRequestDTO) -> None: ...

    def execute(
        self,
        request: JudgeRequestDTO,
        model_name: str,
        t_start: float,
        client: ZeroDecodeClient,
    ) -> JudgeResponseDTO: ...


@dataclass(frozen=True)
class TaskDefinition:
    """タスク種別ごとの検証・実行・バッチ評価ハンドラをまとめた定義"""

    name: str
    execute: TaskExecutor
    validate_request: RequestValidator = _noop_validator
    batch_handler: Optional[BatchQuestionHandler] = None


# =====================================================================
# 組み込みタスクの検証・実行実装 (noul / choice / score / multilabel)
# =====================================================================


def _validate_choice_request(request: JudgeRequestDTO) -> None:
    count = len(request.labels)
    if not MIN_CHOICE_LABELS <= count <= MAX_CHOICE_LABELS:
        raise ValueError(f"choice task requires {MIN_CHOICE_LABELS} to {MAX_CHOICE_LABELS} labels (got {count})")


def _validate_multilabel_request(request: JudgeRequestDTO) -> None:
    if len(request.labels) == 0:
        raise ValueError("multilabel task requires at least 1 label")


def _execute_noul(
    request: JudgeRequestDTO,
    model_name: str,
    t_start: float,
    client: ZeroDecodeClient,
) -> JudgeResponseDTO:
    messages, target_tokens, _ = PromptBuilder.build_messages(request, model_name)
    logprobs, _, usage = client.forward_with_usage(model_name, messages, target_tokens=target_tokens)
    total_latency = (time.perf_counter() - t_start) * 1000.0
    res = ResultMapper.map_noul(logprobs, total_latency)
    res.usage = usage
    return res


def _execute_choice(
    request: JudgeRequestDTO,
    model_name: str,
    t_start: float,
    client: ZeroDecodeClient,
) -> JudgeResponseDTO:
    # 1. 通常 Forward
    messages_fwd, tokens_fwd, symbol_map_fwd = PromptBuilder.build_messages(request, model_name, swap=False)
    logprobs_fwd, _, usage_fwd = client.forward_with_usage(model_name, messages_fwd, target_tokens=tokens_fwd)

    logprobs_swap = None
    symbol_map_swap = None
    usages: List[Optional[UsageDict]] = [usage_fwd]

    # 2. A/Bスワップ検証 (Issue #4 仕様)
    if request.swap_verify and len(request.labels) >= 2:
        messages_swap, tokens_swap, symbol_map_swap = PromptBuilder.build_messages(request, model_name, swap=True)
        logprobs_swap, _, usage_swap = client.forward_with_usage(
            model_name, messages_swap, target_tokens=tokens_swap
        )
        usages.append(usage_swap)

    total_latency = (time.perf_counter() - t_start) * 1000.0
    res = ResultMapper.map_choice(
        logprobs_forward=logprobs_fwd,
        symbol_map_forward=symbol_map_fwd,
        latency_ms=total_latency,
        logprobs_swap=logprobs_swap,
        symbol_map_swap=symbol_map_swap,
    )
    res.usage = merge_usage(usages)
    return res


def _execute_score(
    request: JudgeRequestDTO,
    model_name: str,
    t_start: float,
    client: ZeroDecodeClient,
) -> JudgeResponseDTO:
    messages, target_tokens, _ = PromptBuilder.build_messages(request, model_name)
    logprobs, _, usage = client.forward_with_usage(model_name, messages, target_tokens=target_tokens)
    total_latency = (time.perf_counter() - t_start) * 1000.0
    res = ResultMapper.map_score(
        logprobs, total_latency, temperature=request.temperature, levels=request.score_levels
    )
    res.usage = usage
    return res


def _execute_multilabel(
    request: JudgeRequestDTO,
    model_name: str,
    t_start: float,
    client: ZeroDecodeClient,
) -> JudgeResponseDTO:
    labels = request.labels
    if not labels:
        raise ValueError("Multi-Label task requires non-empty request.labels")

    label_results = []
    usages: List[Optional[UsageDict]] = []
    for lbl in labels:
        messages, target_tokens, _ = PromptBuilder.build_messages(request, model_name, single_label_target=lbl)
        logprobs, _, usage = client.forward_with_usage(model_name, messages, target_tokens=target_tokens)
        label_results.append((lbl, logprobs))
        usages.append(usage)

    total_latency = (time.perf_counter() - t_start) * 1000.0
    res = ResultMapper.map_multilabel(
        label_results, total_latency, threshold=request.threshold, offset=request.offset
    )
    res.usage = merge_usage(usages)
    return res


# =====================================================================
# /api/v1/evaluate 用組み込みバッチ質問ハンドラ
# =====================================================================


def _evaluate_choice_question(
    q: ChoiceQuestionDTO,
    state_str: str,
    request: EvaluateRequestDTO,
    pipeline: "JudgePipeline",
    target_model: str,
    custom_client: Optional[ZeroDecodeClient],
) -> Tuple[Dict[str, Any], Optional[UsageDict]]:
    instructions = q.instructions or ""
    if isinstance(q.criteria, dict):
        labels = list(q.criteria.keys())
        rule_desc = instructions
        if any(q.criteria.values()):
            criteria_desc = "\n".join([f"{k}: {v}" for k, v in q.criteria.items() if v])
            rule_desc = f"{instructions}\n【各選択肢の定義】\n{criteria_desc}".strip()
    else:
        labels = list(q.criteria)
        rule_desc = instructions

    res = pipeline.judge_choice(
        context_text=state_str,
        labels=labels,
        rule_definition=rule_desc,
        swap_verify=request.swap_verify,
        model=target_model,
        client=custom_client,
    )

    probabilities = res.details.get("probabilities", {})
    answer = {
        "type": "choice",
        "choice": res.verdict,
        "confidence": res.confidence,
        "probabilities": probabilities,
        "status": res.status,
        "error_message": res.error_message,
    }
    return answer, res.usage


def _evaluate_score_question(
    q: ScoreQuestionDTO,
    state_str: str,
    request: EvaluateRequestDTO,
    pipeline: "JudgePipeline",
    target_model: str,
    custom_client: Optional[ZeroDecodeClient],
) -> Tuple[Dict[str, Any], Optional[UsageDict]]:
    instructions = q.instructions or ""
    rule_desc = instructions
    if q.criteria:
        if isinstance(q.criteria, list):
            crit_text = " | ".join([f"{i + 1}={lbl}" for i, lbl in enumerate(q.criteria)])
            rule_desc = f"{instructions}\n【段階定義】 {crit_text}".strip()
        elif isinstance(q.criteria, dict):
            crit_text = "\n".join([f"{k}: {v}" for k, v in q.criteria.items()])
            rule_desc = f"{instructions}\n【段階定義】\n{crit_text}".strip()

    res = pipeline.judge_score(
        context_text=state_str,
        rule_definition=rule_desc,
        temperature=request.temperature,
        model=target_model,
        client=custom_client,
        score_levels=q.levels,
    )

    distribution = res.details.get("distribution", {})
    answer = {
        "type": "score",
        "score": res.verdict,
        "confidence": res.confidence,
        "legend": distribution,
        "probabilities": distribution,
        "status": res.status,
        "error_message": res.error_message,
    }
    return answer, res.usage


def _evaluate_noul_question(
    q: NoulQuestionDTO,
    state_str: str,
    request: EvaluateRequestDTO,
    pipeline: "JudgePipeline",
    target_model: str,
    custom_client: Optional[ZeroDecodeClient],
) -> Tuple[Dict[str, Any], Optional[UsageDict]]:
    instructions = q.instructions or ""
    rule_desc = instructions
    if q.criteria:
        yes_desc = q.criteria.get("true", "")
        no_desc = q.criteria.get("false", "")
        rule_desc = f"{instructions}\n(Yes: {yes_desc}, No: {no_desc})".strip()

    res = pipeline.judge_noul(
        context_text=state_str,
        rule_definition=rule_desc,
        model=target_model,
        client=custom_client,
    )

    # ERROR / INCONCLUSIVE のときは中立値(0.5)を作らず、null で返す (Issue #21)
    ok = res.status == "SUCCESS"
    p_yes = res.details.get("prob_yes") if ok else None
    answer = {
        "type": "noul",
        "noul": round(p_yes, 4) if p_yes is not None else None,
        "verdict": res.verdict,
        "confidence": res.confidence,
        "margin": res.details.get("margin") if ok else None,
        "status": res.status,
        "error_message": res.error_message,
    }
    return answer, res.usage


def _evaluate_multilabel_question(
    q: MultiLabelQuestionDTO,
    state_str: str,
    request: EvaluateRequestDTO,
    pipeline: "JudgePipeline",
    target_model: str,
    custom_client: Optional[ZeroDecodeClient],
) -> Tuple[Dict[str, Any], Optional[UsageDict]]:
    instructions = q.instructions or ""
    if isinstance(q.criteria, dict):
        labels = list(q.criteria.keys())
        criteria_desc = "\n".join([f"{k}: {v}" for k, v in q.criteria.items() if v])
        rule_desc = f"{instructions}\n【ラベル定義】\n{criteria_desc}".strip()
    else:
        labels = list(q.criteria)
        rule_desc = instructions

    res = pipeline.judge_multilabel(
        context_text=state_str,
        labels=labels,
        rule_definition=rule_desc,
        model=target_model,
        client=custom_client,
        threshold=q.threshold,
    )

    answer = {
        "type": "multilabel",
        "matched_labels": res.verdict,
        "confidence": res.confidence,
        "probabilities": res.details.get("probabilities", {}),
        "margins": res.details.get("margins", {}),
        "status": res.status,
        "error_message": res.error_message,
    }
    return answer, res.usage


BUILTIN_TASKS: Tuple[TaskDefinition, ...] = (
    TaskDefinition(
        name="noul",
        execute=_execute_noul,
        batch_handler=_evaluate_noul_question,
    ),
    TaskDefinition(
        name="choice",
        validate_request=_validate_choice_request,
        execute=_execute_choice,
        batch_handler=_evaluate_choice_question,
    ),
    TaskDefinition(
        name="score",
        execute=_execute_score,
        batch_handler=_evaluate_score_question,
    ),
    TaskDefinition(
        name="multilabel",
        validate_request=_validate_multilabel_request,
        execute=_execute_multilabel,
        batch_handler=_evaluate_multilabel_question,
    ),
)


class TaskRegistry:
    """タスク定義（検証・実行・バッチ評価ハンドラ）のレジストリ"""

    def __init__(self, include_builtins: bool = True) -> None:
        self._tasks: Dict[str, TaskHandler] = {}
        self._batch_handlers: Dict[str, BatchQuestionHandler] = {}
        if include_builtins:
            self.register_builtins()

    def register_builtins(self) -> None:
        """組み込み 4 タスクおよび互換エイリアス（boolean）を登録する"""
        for task in BUILTIN_TASKS:
            self.register(task)
        # NoulQuestionDTO.type は "noul" | "boolean" の双方を受け付けるためエイリアスを登録
        self.register_batch_handler("boolean", _evaluate_noul_question)

    def register(
        self,
        task: TaskHandler,
        *,
        batch_handler: Optional[BatchQuestionHandler] = None,
        overwrite: bool = True,
    ) -> None:
        """タスクを登録する。batch_handler が指定されていればバッチ評価ハンドラも登録する。"""
        if not task.name or not isinstance(task.name, str):
            raise ValueError("Task name must be a non-empty string")
        if not overwrite and task.name in self._tasks:
            raise ValueError(f"Task '{task.name}' is already registered")
        self._tasks[task.name] = task
        resolved_batch_handler = batch_handler or getattr(task, "batch_handler", None)
        if resolved_batch_handler is not None:
            self._batch_handlers[task.name] = resolved_batch_handler

    def unregister(self, name: str) -> None:
        """指定名のタスク定義とバッチ評価ハンドラを削除する（主にテスト用）"""
        self._tasks.pop(name, None)
        self._batch_handlers.pop(name, None)

    def register_batch_handler(
        self,
        question_type: str,
        handler: BatchQuestionHandler,
        *,
        overwrite: bool = True,
    ) -> None:
        """/api/v1/evaluate 用の質問種別ハンドラを登録する"""
        if not question_type or not isinstance(question_type, str):
            raise ValueError("question_type must be a non-empty string")
        if not overwrite and question_type in self._batch_handlers:
            raise ValueError(f"Batch handler '{question_type}' is already registered")
        self._batch_handlers[question_type] = handler

    def has(self, name: str) -> bool:
        """タスクが登録済みかどうかを返す"""
        return name in self._tasks

    def list_tasks(self) -> Tuple[str, ...]:
        """登録済みタスク名の一覧を返す"""
        return tuple(self._tasks.keys())

    def get(self, name: str) -> TaskHandler:
        """登録済みタスクを取得する。未登録の場合は ValueError を送出する。"""
        if name not in self._tasks:
            raise ValueError(f"Unknown task_type: {name}")
        return self._tasks[name]

    def validate_request(self, request: JudgeRequestDTO) -> None:
        """リクエストの task_type 存在確認とタスク固有バリデーションを実行する"""
        task = self.get(request.task_type)
        task.validate_request(request)

    def execute(
        self,
        request: JudgeRequestDTO,
        model_name: str,
        t_start: float,
        client: ZeroDecodeClient,
    ) -> JudgeResponseDTO:
        """リクエストを該当タスクハンドラへ委譲して推論判定を実行する"""
        task = self.get(request.task_type)
        task.validate_request(request)
        return task.execute(request, model_name, t_start, client)

    def get_batch_handler(self, question_type: str) -> BatchQuestionHandler:
        """質問種別に対応するバッチ評価ハンドラを返す"""
        if question_type not in self._batch_handlers:
            raise ValueError(f"Unknown batch question type: {question_type}")
        return self._batch_handlers[question_type]

    def execute_batch_question(
        self,
        question: Any,
        state_str: str,
        request: EvaluateRequestDTO,
        pipeline: "JudgePipeline",
        target_model: str,
        custom_client: Optional[ZeroDecodeClient],
    ) -> Tuple[Dict[str, Any], Optional[UsageDict]]:
        """バッチ評価の1問分を質問種別レジストリ経由で実行する"""
        question_type = getattr(question, "type", None)
        if not isinstance(question_type, str):
            raise ValueError(f"Question object must have a string 'type' attribute: {question!r}")
        handler = self.get_batch_handler(question_type)
        return handler(question, state_str, request, pipeline, target_model, custom_client)


default_task_registry = TaskRegistry(include_builtins=True)
