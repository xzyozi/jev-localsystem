"""JEV REST API リクエスト・レスポンススキーマ定義モジュール"""

from typing import List, Optional

from pydantic import BaseModel, Field

# ==========================================
# タスク別リクエストスキーマ
# ==========================================

class NoulRequest(BaseModel):
    """真偽判定 (Yes/No) リクエスト"""

    context_text: str = Field(..., description="判定対象テキスト（最大12,000文字 / 4,000トークン以内）")
    rule_definition: str = Field(default="", description="動的注入する判定基準・ルール（任意）")
    model: Optional[str] = Field(default=None, description="推論モデル名（省略時はTier 1主軸モデル）")

    model_config = {
        "json_schema_extra": {
            "example": {
                "context_text": "ユーザーが正しい二要素認証コードを入力してログインしました。",
                "rule_definition": "正常な認証アクティビティであるかを判定してください。",
            }
        }
    }


class ChoiceRequest(BaseModel):
    """単一選択 (分類) リクエスト"""

    context_text: str = Field(..., description="分類対象テキスト")
    labels: List[str] = Field(..., min_length=2, description="選択肢ラベルのリスト（2個以上）")
    rule_definition: str = Field(default="", description="分類ルール（任意）")
    swap_verify: bool = Field(default=True, description="位置バイアス相殺のためのA/Bスワップ推論を行うか")
    model: Optional[str] = Field(default=None, description="推論モデル名")

    model_config = {
        "json_schema_extra": {
            "example": {
                "context_text": "SELECT * FROM orders JOIN customers ON orders.cust_id = customers.id;",
                "labels": ["Database", "Frontend", "UI/UX Design"],
                "swap_verify": True,
            }
        }
    }


class ScoreRequest(BaseModel):
    """段階評価 (1〜5連続値スコア) リクエスト"""

    context_text: str = Field(..., description="評価対象テキスト")
    rule_definition: str = Field(default="", description="評価基準・採点基準（任意）")
    temperature: float = Field(default=1.0, gt=0.0, description="確率分布の平滑化温度パラメータ")
    model: Optional[str] = Field(default=None, description="推論モデル名")

    model_config = {
        "json_schema_extra": {
            "example": {
                "context_text": "ユニットテストのカバレッジが95%を超え、ドキュメントも完全に保守されています。",
                "rule_definition": "コードベースの健全性を1〜5で評価してください。",
                "temperature": 1.0,
            }
        }
    }


class MultiLabelRequest(BaseModel):
    """複数ラベル分類 (Sigmoid独立判定) リクエスト"""

    context_text: str = Field(..., description="分類対象テキスト")
    labels: List[str] = Field(..., min_length=1, description="判定対象ラベルのリスト（1個以上）")
    rule_definition: str = Field(default="", description="判定ルール（任意）")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="ラベル採用判定のSigmoid確率閾値")
    model: Optional[str] = Field(default=None, description="推論モデル名")

    model_config = {
        "json_schema_extra": {
            "example": {
                "context_text": "SQLインジェクション脆弱性を修正し、クエリキャッシュを導入しました。",
                "labels": ["Security", "Database", "Mobile"],
                "threshold": 0.5,
            }
        }
    }


# ==========================================
# システム運用・監視系レスポンススキーマ
# ==========================================

class HealthResponse(BaseModel):
    """ヘルスチェックレスポンス"""

    status: str = Field(default="ok", description="サーバー稼働状態")
    backend_connected: bool = Field(..., description="Ollama推論バックエンドへの接続可否")
    default_model: str = Field(..., description="現在設定されているTier 1主軸モデル")
    lightweight_model: str = Field(..., description="現在設定されているTier 2常駐モデル")


class VRAMMetricsResponse(BaseModel):
    """VRAMリソース・キュー稼働状態レスポンス"""

    waiting_queue_count: int = Field(..., description="現在セマフォ待ち状態のリクエスト数")
    is_running: bool = Field(..., description="現在GPU推論中フラグ")
    total_processed: int = Field(..., description="起動以降に直列完走した累計リクエスト数")


class ModelInfo(BaseModel):
    """モデル諸元情報"""

    tier: str = Field(..., description="モデル階層 (Tier 1 / Tier 2)")
    model_name: str = Field(..., description="モデル識別名")
    role: str = Field(..., description="運用用途")
    vram_usage: str = Field(..., description="概算VRAM専有量")


class ModelListResponse(BaseModel):
    """モデル一覧レスポンス"""

    models: List[ModelInfo] = Field(..., description="サポートモデル一覧")
