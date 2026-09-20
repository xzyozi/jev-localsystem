"""JEV REST API システム運用・監視・メトリクスルーターモジュール"""

import requests
from fastapi import APIRouter

from jev.server.config import settings
from jev.server.schemas import (
    HealthResponse,
    ModelInfo,
    ModelListResponse,
    VRAMMetricsResponse,
)
from jev.vram_manager import default_vram_manager

router = APIRouter(tags=["System"])


@router.get("/health", response_model=HealthResponse, summary="ヘルスチェック (死活・疎通監視)")
def health_check() -> HealthResponse:
    """サーバーの稼働状態およびOllama推論バックエンドへの疎通確認を行います。"""
    backend_ok = False
    try:
        # Ollama のベースURL (/api/tags または /) に疎通確認
        base_url = settings.backend_endpoint.split("/v1/")[0]
        resp = requests.get(f"{base_url}/api/tags", timeout=2.0)
        backend_ok = (resp.status_code == 200)
    except Exception:
        backend_ok = False

    return HealthResponse(
        status="ok" if backend_ok else "degraded",
        backend_connected=backend_ok,
        default_model=settings.default_model,
        lightweight_model=settings.lightweight_model,
    )


@router.get("/api/v1/vram/metrics", response_model=VRAMMetricsResponse, summary="VRAMリソース・キュー稼働監視")
def get_vram_metrics() -> VRAMMetricsResponse:
    """単一GPUリソース保護のための直列セマフォ稼働状況および累計処理リクエスト数を取得します。"""
    metrics = default_vram_manager.get_metrics()
    return VRAMMetricsResponse(
        waiting_queue_count=metrics["waiting_queue_count"],
        is_running=metrics["is_running"],
        total_processed=metrics["total_processed"],
    )


@router.get("/api/v1/models", response_model=ModelListResponse, summary="サポートモデル一覧・諸元カタログ")
def list_models() -> ModelListResponse:
    """JEV本番確定モデル階層アーキテクチャ（Tier 1 主軸 / Tier 2 軽量常駐）の一覧を返却します。"""
    return ModelListResponse(
        models=[
            ModelInfo(
                tier="Tier 1 (Primary / 主軸)",
                model_name=settings.default_model,
                role="高精度判定・多値分類・Sigmoid分離（分離度36.6pt最高値・思考抑制Prefill適用）",
                vram_usage="約 5.2 GB (100% GPU収容)",
            ),
            ModelInfo(
                tier="Tier 2 (Lightweight / 常駐)",
                model_name=settings.lightweight_model,
                role="高速常駐ゲートキーパー・二値判定・前処理（2.5GB省VRAM・分離度14.6pt）",
                vram_usage="約 2.5 GB (100% GPU収容)",
            ),
        ]
    )
