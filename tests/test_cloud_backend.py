"""JEV クラウド Zero-Decode バックエンド包括テストスイート (Issue #15)

OpenAI 互換バックエンド (OpenAIBackend) の正常系・異常系・Logprobs抽出を検証する。
"""

from unittest.mock import MagicMock, patch

import pytest

from jev.exceptions import BackendConnectionError
from jev.zero_decode import OllamaBackend, OpenAIBackend, ZeroDecodeClient


def test_ollama_backend_is_cloud():
    """OllamaBackend の is_cloud が False であること"""
    b = OllamaBackend()
    assert b.is_cloud is False


def test_openai_backend_is_cloud():
    """OpenAIBackend の is_cloud が True であること"""
    b = OpenAIBackend(api_key="test-key")
    assert b.is_cloud is True


def test_openai_backend_missing_api_key(monkeypatch):
    """APIキーが未設定の場合に BackendConnectionError が発生すること"""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENJEV_API_KEY", raising=False)
    monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    backend = OpenAIBackend(api_key="")
    with pytest.raises(BackendConnectionError) as exc:
        backend.forward("gpt-4o-mini", [{"role": "user", "content": "test"}])
    assert "OpenAI API key is missing" in str(exc.value)


@patch("requests.post")
def test_openai_backend_forward_success(mock_post):
    """OpenAI 互換レスポンスから Top-Logprobs が正常に抽出されること"""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "Yes"},
                "logprobs": {
                    "content": [
                        {
                            "token": "Yes",
                            "logprob": -0.05,
                            "top_logprobs": [
                                {"token": "Yes", "logprob": -0.05},
                                {"token": "No", "logprob": -3.2},
                                {"token": " Yes", "logprob": -4.5},
                            ],
                        }
                    ]
                },
            }
        ]
    }
    mock_post.return_value = mock_resp

    backend = OpenAIBackend(api_key="sk-test-mock-key", base_url="https://api.openai.com/v1")
    client = ZeroDecodeClient(backend=backend)

    logprobs, latency_ms = client.forward(
        "gpt-4o-mini",
        [{"role": "user", "content": "Say yes"}],
        top_logprobs=10,
    )

    assert client.is_cloud is True
    assert "Yes" in logprobs
    assert logprobs["Yes"] == -0.05
    assert logprobs["No"] == -3.2
    assert logprobs[" Yes"] == -4.5
    assert latency_ms > 0

    # 呼び出しパラメータの検証
    mock_post.assert_called_once()
    _, kwargs = mock_post.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer sk-test-mock-key"
    assert kwargs["json"]["max_tokens"] == 1
    assert kwargs["json"]["logprobs"] is True
    assert kwargs["json"]["model"] == "gpt-4o-mini"


@patch("requests.post")
def test_openai_backend_http_error(mock_post):
    """HTTP 401 / 500 等のエラー時に BackendConnectionError が発生すること"""
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.text = "Unauthorized"
    mock_post.return_value = mock_resp

    backend = OpenAIBackend(api_key="sk-invalid")
    with pytest.raises(BackendConnectionError) as exc:
        backend.forward("gpt-4o-mini", [{"role": "user", "content": "test"}])
    assert "returned HTTP 401" in str(exc.value)
