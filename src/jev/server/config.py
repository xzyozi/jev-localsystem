"""JEV REST API サーバー設定モジュール"""

import os

from pydantic import BaseModel, Field


class ServerSettings(BaseModel):
    """JEV REST API サーバー環境設定"""

    host: str = Field(default=os.environ.get("JEV_HOST", "0.0.0.0"), description="バインド先ホスト")
    port: int = Field(default=int(os.environ.get("JEV_PORT", "8000")), description="リッスンポート番号")
    cors_origins: list[str] = Field(
        default_factory=lambda: os.environ.get("JEV_CORS_ORIGINS", "*").split(","),
        description="CORS許可オリジン一覧",
    )
    default_model: str = Field(
        default=os.environ.get("JEV_DEFAULT_MODEL", "qwen3:8b"),
        description="本番主軸モデル (Tier 1 Primary)",
    )
    lightweight_model: str = Field(
        default=os.environ.get("JEV_LIGHTWEIGHT_MODEL", "phi4-mini:latest"),
        description="本番常駐軽量モデル (Tier 2 Lightweight)",
    )
    backend_endpoint: str = Field(
        default=os.environ.get("JEV_BACKEND_ENDPOINT", "http://localhost:11434/v1/chat/completions"),
        description="Ollama / OpenAI互換エンドポイントURL",
    )
    log_level: str = Field(default=os.environ.get("JEV_LOG_LEVEL", "info"), description="ログレベル")


settings = ServerSettings()
