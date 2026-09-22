"""JEV パイプライン＆API クラウド統合テストスイート (Issue #15)

JudgePipeline および /api/v1/evaluate エンドポイントにおいて、
クラウド推論 (OpenAIBackend) 時に VRAM マネージャーがバイパスされ、
モック経由で全タスクが高速かつ安全に処理されることを検証する。
"""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
import pytest

from jev.dto import JudgeRequestDTO
from jev.pipeline import JudgePipeline
from jev.server.app import app
from jev.zero_decode import OpenAIBackend, ZeroDecodeClient


@pytest.fixture
def mock_cloud_client():
    """OpenAI API をモックした ZeroDecodeClient"""
    client = ZeroDecodeClient(backend=OpenAIBackend(api_key="sk-test-mock"))
    return client


def _make_mock_openai_response(token: str, logprob: float = -0.1):
    """OpenAI 形式のモックレスポンスを生成するヘルパー"""
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": token},
                "logprobs": {
                    "content": [
                        {
                            "token": token,
                            "logprob": logprob,
                            "top_logprobs": [
                                {"token": token, "logprob": logprob},
                                {"token": f" {token}", "logprob": logprob - 0.5},
                                {"token": "No", "logprob": -5.0},
                                {"token": "B", "logprob": -5.0},
                                {"token": "2", "logprob": -5.0},
                            ],
                        }
                    ]
                },
            }
        ]
    }


@patch("requests.post")
def test_pipeline_cloud_bypass_and_execution(mock_post):
    """JudgePipeline がクラウド推論時に VRAM ロックをバイパスして正常終了すること"""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = _make_mock_openai_response("Yes")
    mock_post.return_value = mock_resp

    cloud_client = ZeroDecodeClient(backend=OpenAIBackend(api_key="sk-test-mock"))
    pipeline = JudgePipeline(client=cloud_client)

    req = JudgeRequestDTO(
        task_type="noul",
        context_text="This is a test running on cloud backend without local GPU.",
        rule_definition="Verify cloud execution.",
        model="gpt-4o-mini",
    )

    res = pipeline.judge(req)
    assert res.status == "SUCCESS"
    assert res.verdict == "Yes"
    assert res.confidence > 0.5


@patch("requests.post")
def test_api_evaluate_cloud_model(mock_post):
    """POST /api/v1/evaluate にクラウドモデルと api_key を渡した際のエンドツーエンド検証"""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = _make_mock_openai_response("A")
    mock_post.return_value = mock_resp

    client = TestClient(app)
    payload = {
        "state": "The user reported an urgent account billing bug.",
        "questions": {
            "dept": {
                "type": "choice",
                "criteria": ["A", "B", "C"],
            },
            "urgent": {
                "type": "noul",
            },
        },
        "model": "gpt-4o-mini",
        "api_key": "sk-mock-eval-key",
    }

    resp = client.post("/api/v1/evaluate", json=payload)
    assert resp.status_code == 200, f"Error: {resp.text}"

    data = resp.json()
    assert data["model"] == "gpt-4o-mini"
    assert "dept" in data["answers"]
    assert "urgent" in data["answers"]
    assert data["meta"]["total_questions"] == 2
