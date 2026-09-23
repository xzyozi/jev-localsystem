import os

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--model",
        action="store",
        default=os.environ.get("JEV_TEST_MODEL", "qwen3:8b"),
        help="検証対象のモデル識別名 (デフォルト: qwen3:8b, 例: phi4-mini:latest)"
    )

@pytest.fixture(scope="session")
def target_model(request):
    return request.config.getoption("--model")
