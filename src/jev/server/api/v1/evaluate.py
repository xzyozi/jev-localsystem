"""JEV REST API Jev互換バッチ評価ルーターモジュール (v1 / 互換エイリアス)

TypeSafe Jev / OpenJev 風の 1リクエスト複数質問バッチ評価 API を提供します (Issue #14)。
完全互換ではなく、パス・Score の段数と値・confidence の定義などは上流と異なります。
"""

import json
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends

from jev import JudgePipeline
from jev.dto import (
    EvaluateRequestDTO,
    EvaluateResponseDTO,
    _empty_usage,
)
from jev.server.api.v1.judge import get_pipeline
from jev.zero_decode import OpenAIBackend, UsageDict, ZeroDecodeClient, is_cloud_model_name, merge_usage

router = APIRouter(tags=["JEV Batch Evaluation"])


def _format_state(state: Any) -> str:
    """state を文字列表現へ正規化する"""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, indent=2)


def _resolve_client(request: EvaluateRequestDTO) -> ZeroDecodeClient | None:
    """リクエスト情報から明示的なクラウドクライアントが必要かを判定・構築する (Issue #15)"""
    # api_key または base_url が明示指定されたか、モデル名がクラウド API の識別名らしい場合
    # (Ollama の name:tag 形式はクラウドとみなさない)
    if request.api_key or request.base_url or is_cloud_model_name(request.model):
        backend = OpenAIBackend(api_key=request.api_key, base_url=request.base_url)
        return ZeroDecodeClient(backend=backend)
    return None


@router.post(
    "/api/v1/evaluate",
    response_model=EvaluateResponseDTO,
    summary="Jev互換バッチ評価 (v1正規ルート)",
    description="stateと複数のtyped questionsを受け取り、各判定結果を一括返却します。",
)
def evaluate_batch(
    request: EvaluateRequestDTO,
    pipeline: JudgePipeline = Depends(get_pipeline),
) -> EvaluateResponseDTO:
    """単一のstateに対して複数の質問（choice, score, noul, multilabel）を順次評価します。"""
    t_start = time.perf_counter()
    state_str = _format_state(request.state)
    target_model = request.model or pipeline.default_model

    # クラウドクライアントの動的解決 (Issue #15)
    custom_client = _resolve_client(request)

    answers: Dict[str, Any] = {}
    usages: List[Optional[UsageDict]] = []

    for q_name, q in request.questions.items():
        answer, usage = pipeline.task_registry.execute_batch_question(
            q, state_str, request, pipeline, target_model, custom_client
        )
        answers[q_name] = answer
        usages.append(usage)

    total_latency_ms = (time.perf_counter() - t_start) * 1000.0

    return EvaluateResponseDTO(
        model=target_model,
        answers=answers,
        usage=merge_usage(usages) or _empty_usage(),
        meta={
            "mode": "zero_decode",
            "latency_ms": round(total_latency_ms, 2),
            "total_questions": len(request.questions),
        },
    )
