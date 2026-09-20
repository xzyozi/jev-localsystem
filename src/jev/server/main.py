"""JEV REST API サーバー起動エントリーポイント"""

import argparse

import uvicorn

from jev.server.config import settings


def parse_args():
    parser = argparse.ArgumentParser(description="JEV REST API サーバー起動CLI")
    parser.add_argument(
        "--host",
        default=settings.host,
        help=f"バインド先ホスト (デフォルト: {settings.host})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.port,
        help=f"リッスンポート番号 (デフォルト: {settings.port})",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="コード変更時の自動リロードを有効化 (開発用)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("==================================================")
    print("      JEV: Judge & Evaluation REST API Server     ")
    print("==================================================")
    print(f"Host: {args.host}:{args.port}")
    print(f"Tier 1 (Primary Model): {settings.default_model}")
    print(f"Tier 2 (Lightweight):   {settings.lightweight_model}")
    print(f"Swagger UI Docs:        http://localhost:{args.port}/docs")
    print("==================================================")

    uvicorn.run(
        "jev.server.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=settings.log_level,
    )


if __name__ == "__main__":
    main()
