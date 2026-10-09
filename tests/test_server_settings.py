"""サーバー設定とパイプライン生成のテスト (Issue #23)

環境変数は ServerSettings() の生成時に読み込まれること、
パイプラインのキャッシュを明示的に破棄・再生成できることを検証する。
Ollama は不要。
"""

import pytest

from jev.server.api.v1 import judge as judge_module
from jev.server.config import ServerSettings


def test_settings_read_environment_at_instantiation(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("JEV_DEFAULT_MODEL", "custom-model:1b")
    monkeypatch.setenv("JEV_PORT", "9001")
    monkeypatch.setenv("JEV_CORS_ORIGINS", "http://a.example,http://b.example")
    s = ServerSettings()
    assert s.default_model == "custom-model:1b"
    assert s.port == 9001
    assert s.cors_origins == ["http://a.example", "http://b.example"]


def test_settings_defaults_when_environment_is_empty(monkeypatch: pytest.MonkeyPatch):
    for key in ("JEV_HOST", "JEV_PORT", "JEV_CORS_ORIGINS", "JEV_DEFAULT_MODEL", "JEV_LIGHTWEIGHT_MODEL"):
        monkeypatch.delenv(key, raising=False)
    s = ServerSettings()
    assert s.default_model == "qwen3:8b"
    assert s.lightweight_model == "phi4-mini:latest"
    assert s.port == 8000
    assert s.cors_origins == ["*"]


def test_get_pipeline_is_cached_and_reset_recreates():
    judge_module.reset_pipeline()
    first = judge_module.get_pipeline()
    assert judge_module.get_pipeline() is first
    judge_module.reset_pipeline()
    second = judge_module.get_pipeline()
    assert second is not first
    judge_module.reset_pipeline()


def test_create_pipeline_does_not_cache():
    judge_module.reset_pipeline()
    a = judge_module.create_pipeline()
    b = judge_module.create_pipeline()
    assert a is not b
    assert a.default_model == judge_module.settings.default_model
