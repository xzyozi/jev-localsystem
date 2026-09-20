"""JEV (Judge & Evaluation via Vectorized-logits) DTO スキーマモジュール

詳細設計書 (JEV-DD-001) 第3章のインターフェースおよびDTO仕様に基づく型安全なデータモデル。
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator

TaskType = Literal["noul", "choice", "score", "multilabel"]
JudgeStatus = Literal["SUCCESS", "INCONCLUSIVE", "ERROR"]


class JudgeRequestDTO(BaseModel):
    """JEV パイプライン共通リクエスト DTO (JEV-DD-001 3.1)"""

    task_type: TaskType = Field(
        ...,
        description="判定タスク種別: 'noul' (真偽), 'choice' (単一選択), 'score' (段階評価), 'multilabel' (複数分類)",
    )
    context_text: str = Field(
        ...,
        description="判定対象テキスト。最大4,000トークン（約12,000文字以内）",
    )
    rule_definition: str = Field(
        default="",
        description="動的注入する判定基準・ルール定義（RAGチャンクやポリシー規程）",
    )
    labels: List[str] = Field(
        default_factory=list,
        description="ChoiceまたはMulti-Labelタスクで使用する選択肢・分類対象ラベルのリスト",
    )
    swap_verify: bool = Field(
        default=False,
        description="位置バイアス相殺のためのA/Bスワップ推論（2回実行照合）を行うか",
    )
    temperature: float = Field(
        default=1.0,
        description="Scoreタスク等で確率分布の平滑度を調整する温度パラメータ（>0.0）",
    )
    model: Optional[str] = Field(
        default=None,
        description="推論に使用するモデル識別名。省略時はパイプラインのデフォルト（Tier 1: qwen3:8b）",
    )

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if v <= 0.0:
            raise ValueError("temperature must be strictly greater than 0.0")
        return v

    @field_validator("labels")
    @classmethod
    def validate_labels(cls, v: List[str], info) -> List[str]:
        # choice や multilabel タスクでは labels の指定が推奨される
        task = info.data.get("task_type")
        if task in ("choice", "multilabel") and len(v) == 0:
            # バリデーションエラーにはせず、デフォルトラベルを後段で補完可能にするか、ここで空リストを許容
            pass
        return v


class NoulDetails(BaseModel):
    """Noul (真偽判定) 詳細計算データ (JEV-DD-001 3.3.A)"""

    margin: float = Field(..., description="YesとNoのLogit差分（正ならYes優位）")
    prob_yes: float = Field(..., description="Yesの正規化確率 (0.0 〜 1.0)")
    prob_no: float = Field(..., description="Noの正規化確率 (0.0 〜 1.0)")


class ChoiceDetails(BaseModel):
    """Choice (単一選択) 詳細計算データ (JEV-DD-001 3.3.B)"""

    symbol: str = Field(..., description="選択された記号 ('A', 'B', 'C' など)")
    is_consistent: bool = Field(default=True, description="スワップ検証で判定結果が整合したか")
    swap_verified: bool = Field(default=False, description="スワップ検証が実施されたか")
    probabilities: Dict[str, float] = Field(default_factory=dict, description="各選択肢ラベルの正規化確率分布")


class ScoreDetails(BaseModel):
    """Score (段階評価) 詳細計算データ (JEV-DD-001 3.3.C)"""

    distribution: Dict[str, float] = Field(..., description="各評価スケール ('1'〜'5') の正規化確率分布")
    most_likely: str = Field(..., description="最も確率が高かった評価スケール ('1'〜'5')")


class MultiLabelDetails(BaseModel):
    """Multi-Label (複数選択) 詳細計算データ (JEV-DD-001 3.3.D)"""

    margins: Dict[str, float] = Field(..., description="各ラベルの独立Logit差分 (Yes - No)")
    probabilities: Dict[str, float] = Field(..., description="各ラベルのSigmoid確率 (0.0 〜 1.0)")
    threshold: float = Field(default=0.5, description="ラベル採用判定のSigmoid確率閾値")
    offset: float = Field(default=0.0, description="Logit差分に対する動的オフセット")


class JudgeResponseDTO(BaseModel):
    """JEV パイプライン共通レスポンス DTO (JEV-DD-001 3.2)"""

    task_type: TaskType = Field(..., description="リクエストされた判定タスク種別")
    status: JudgeStatus = Field(..., description="判定結果ステータス: 'SUCCESS', 'INCONCLUSIVE', 'ERROR'")
    verdict: Any = Field(..., description="タスク別の確定判定値（Noul: str, Choice: str, Score: float, Multi-Label: List[str]）")
    latency_ms: float = Field(..., description="パイプライン全体の処理所要時間（ミリ秒）")
    confidence: Optional[float] = Field(default=None, description="判定の確信度 (0.0 〜 1.0)")
    details: Dict[str, Any] = Field(default_factory=dict, description="タスク別詳細計算データ")
    error_message: Optional[str] = Field(default=None, description="エラー発生時のエラーメッセージ")
