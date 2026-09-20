"""JEV REST API v1 パッケージ"""

from jev.server.api.v1.judge import router as judge_router

__all__ = ["judge_router"]
