"""JEV FastAPI アプリケーションファクトリおよび例外ハンドラーモジュール"""

import logging
import sys
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from jev.exceptions import (
    BackendConnectionError,
    InconclusiveVerdictError,
    JevError,
    PayloadTooLargeError,
    QueueTimeoutError,
    TokenNotFoundError,
)
from jev.server.api.v1 import judge_router, system_router
from jev.server.config import settings

logger = logging.getLogger("jev.server")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def create_app() -> FastAPI:
    """FastAPI アプリケーションインスタンスを生成・設定するファクトリ関数"""
    app = FastAPI(
        title="JEV: Judge & Evaluation via Vectorized-logits API",
        description=(
            "文章生成（デコードループ）を伴わず、特定トークンのLogit抽出と確率正規化により"
            "超高速・型安全な判定を行うローカル判定基盤 JEV の REST API サーバーです。"
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS ミドルウェア設定
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ==========================================
    # 例外ハンドラー登録（JEV内部例外 -> HTTPステータス）
    # ==========================================

    @app.exception_handler(PayloadTooLargeError)
    async def payload_too_large_handler(request: Request, exc: PayloadTooLargeError):
        logger.warning("HTTP 413 Payload Too Large: %s", exc.message)
        return JSONResponse(
            status_code=413,
            content={"detail": exc.message, "error_type": "PayloadTooLargeError"},
        )

    @app.exception_handler(QueueTimeoutError)
    async def queue_timeout_handler(request: Request, exc: QueueTimeoutError):
        logger.error("HTTP 504 Gateway Timeout: %s", exc.message)
        return JSONResponse(
            status_code=504,
            content={"detail": exc.message, "error_type": "QueueTimeoutError"},
        )

    @app.exception_handler(BackendConnectionError)
    async def backend_connection_handler(request: Request, exc: BackendConnectionError):
        logger.error("HTTP 502 Bad Gateway: %s", exc.message)
        return JSONResponse(
            status_code=502,
            content={"detail": exc.message, "error_type": "BackendConnectionError"},
        )

    @app.exception_handler(InconclusiveVerdictError)
    async def inconclusive_handler(request: Request, exc: InconclusiveVerdictError):
        return JSONResponse(
            status_code=422,
            content={"detail": exc.message, "error_type": "InconclusiveVerdictError"},
        )

    @app.exception_handler(JevError)
    async def jev_base_handler(request: Request, exc: JevError):
        logger.error("HTTP %d JevError: %s", exc.status_code, exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.message, "error_type": type(exc).__name__},
        )

    # ルーター登録
    app.include_router(judge_router)
    app.include_router(system_router)

    return app


app = create_app()
