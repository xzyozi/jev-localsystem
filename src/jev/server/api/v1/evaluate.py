"""JEV REST API Jev互換バッチ評価ルーターモジュール (v1 / 互換エイリアス)

TypeSafe Jev / OpenJev と完全互換な 1リクエスト複数質問バッチ評価 API を提供します (Issue #14)。
"""

import json
import time
from typing import Any, Dict

from fastapi import APIRouter, Depends

from jev import JudgePipeline
from jev.dto import (
    ChoiceQuestionDTO,
    EvaluateRequestDTO,
    EvaluateResponseDTO,
    MultiLabelQuestionDTO,
    NoulQuestionDTO,
    ScoreQuestionDTO,
)
from jev.server.api.v1.judge import get_pipeline
from jev.zero_decode import OpenAIBackend, ZeroDecodeClient

router = APIRouter(tags=["JEV Batch Evaluation"])


def _format_state(state: Any) -> str:
    """state を文字列表現へ正規化する"""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, indent=2)


def _resolve_client(request: EvaluateRequestDTO) -> ZeroDecodeClient | None:
    """リクエスト情報から明示的なクラウドクライアントが必要かを判定・構築する (Issue #15)"""
    model_lower = (request.model or "").lower()
    is_cloud_model = any(
        model_lower.startswith(p)
        for p in ("gpt-", "o1-", "o3-", "text-embedding", "claude-", "gemini-")
    )

    # api_key または base_url が明示指定されたか、モデル名がクラウドプレフィックスの場合
    if request.api_key or request.base_url or is_cloud_model:
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

    for q_name, q in request.questions.items():
        instructions = q.instructions or ""


        if isinstance(q, ChoiceQuestionDTO):
            # labels の抽出
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
            answers[q_name] = {
                "type": "choice",
                "choice": res.verdict,
                "confidence": res.confidence,
                "probabilities": probabilities,
                "status": res.status,
            }

        elif isinstance(q, ScoreQuestionDTO):
            rule_desc = instructions
            if q.criteria:
                if isinstance(q.criteria, list):
                    crit_text = " | ".join([f"{i+1}={lbl}" for i, lbl in enumerate(q.criteria)])
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
            )

            distribution = res.details.get("distribution", {})
            answers[q_name] = {
                "type": "score",
                "score": res.verdict,
                "confidence": res.confidence,
                "legend": distribution,
                "probabilities": distribution,
                "status": res.status,
            }

        elif isinstance(q, NoulQuestionDTO):
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

            p_yes = res.details.get("prob_yes", 0.5)
            answers[q_name] = {
                "type": "noul",
                "noul": round(p_yes, 4),
                "verdict": res.verdict,
                "confidence": res.confidence,
                "margin": res.details.get("margin", 0.0),
                "status": res.status,
            }

        elif isinstance(q, MultiLabelQuestionDTO):
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
            )


            answers[q_name] = {
                "type": "multilabel",
                "matched_labels": res.verdict,
                "confidence": res.confidence,
                "probabilities": res.details.get("probabilities", {}),
                "margins": res.details.get("margins", {}),
                "status": res.status,
            }

    total_latency_ms = (time.perf_counter() - t_start) * 1000.0

    return EvaluateResponseDTO(
        model=target_model,
        answers=answers,
        usage={"input_tokens": None, "output_tokens": None},
        meta={
            "mode": "zero_decode",
            "latency_ms": round(total_latency_ms, 2),
            "total_questions": len(request.questions),
        },
    )
