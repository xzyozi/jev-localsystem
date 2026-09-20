"""JEV REST API 判定ルーターモジュール (v1)"""

from fastapi import APIRouter, Depends

from jev import JudgePipeline, JudgeRequestDTO, JudgeResponseDTO
from jev.server.config import settings
from jev.server.schemas import (
    ChoiceRequest,
    MultiLabelRequest,
    NoulRequest,
    ScoreRequest,
)

router = APIRouter(prefix="/api/v1", tags=["Judge"])

# シングルトンパイプライン
_pipeline_instance: JudgePipeline | None = None


def get_pipeline() -> JudgePipeline:
    """JudgePipeline インスタンスを取得する (依存性注入用)"""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = JudgePipeline(
            default_model=settings.default_model,
            lightweight_model=settings.lightweight_model,
        )
    return _pipeline_instance


@router.post("/judge", response_model=JudgeResponseDTO, summary="統合判定エンドポイント")
def execute_judge(
    request: JudgeRequestDTO,
    pipeline: JudgePipeline = Depends(get_pipeline),
) -> JudgeResponseDTO:
    """タスク種別（noul, choice, score, multilabel）を包括的に受け付け判定を実行します。"""
    return pipeline.judge(request)


@router.post("/noul", response_model=JudgeResponseDTO, summary="真偽判定 (Yes/No)")
def execute_noul(
    request: NoulRequest,
    pipeline: JudgePipeline = Depends(get_pipeline),
) -> JudgeResponseDTO:
    """提示された基準に対してテキストが条件を満たすか（Yes / No）を超高速に判定します。"""
    return pipeline.judge_noul(
        context_text=request.context_text,
        rule_definition=request.rule_definition,
        model=request.model,
    )


@router.post("/choice", response_model=JudgeResponseDTO, summary="単一選択 (分類)")
def execute_choice(
    request: ChoiceRequest,
    pipeline: JudgePipeline = Depends(get_pipeline),
) -> JudgeResponseDTO:
    """複数の選択肢ラベルから最適なものを1つ選定します。位置バイアス相殺（スワップ検証）に対応しています。"""
    return pipeline.judge_choice(
        context_text=request.context_text,
        labels=request.labels,
        rule_definition=request.rule_definition,
        swap_verify=request.swap_verify,
        model=request.model,
    )


@router.post("/score", response_model=JudgeResponseDTO, summary="段階評価 (1〜5連続値)")
def execute_score(
    request: ScoreRequest,
    pipeline: JudgePipeline = Depends(get_pipeline),
) -> JudgeResponseDTO:
    """テキストの品質や適合度を1〜5の確率加重平均による高精度な連続値スコア（期待値）として評価します。"""
    return pipeline.judge_score(
        context_text=request.context_text,
        rule_definition=request.rule_definition,
        temperature=request.temperature,
        model=request.model,
    )


@router.post("/multilabel", response_model=JudgeResponseDTO, summary="複数ラベル分類 (Sigmoid)")
def execute_multilabel(
    request: MultiLabelRequest,
    pipeline: JudgePipeline = Depends(get_pipeline),
) -> JudgeResponseDTO:
    """提示された複数のラベルそれぞれに対し、独立Sigmoid確率を算出して閾値を超えたものを抽出します。"""
    req = JudgeRequestDTO(
        task_type="multilabel",
        context_text=request.context_text,
        labels=request.labels,
        rule_definition=request.rule_definition,
        model=request.model,
    )
    return pipeline.judge(req)
