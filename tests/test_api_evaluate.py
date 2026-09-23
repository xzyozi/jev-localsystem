"""JEV REST API Jev互換バッチ評価エンドポイント包括テストスイート (Issue #14)

POST /api/v1/evaluate および互換エイリアス POST /api/evaluate の正常系・異常系・Jevスキーマ互換性を検証する。
"""

from fastapi.testclient import TestClient
import pytest

from jev.server.app import app


@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient フィクスチャ"""
    return TestClient(app)


def test_evaluate_batch_v1_full_suite(client: TestClient, target_model: str):
    """POST /api/v1/evaluate の全タスク型混在バッチ評価テスト"""
    payload = {
        "state": "Charged twice again!! Second month in a row for the Pro plan.",
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {
                    "billing": "Charges, refunds, invoices",
                    "technical": "Bugs or product issues",
                    "other": "Doesn't fit",
                },
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent?",
                "criteria": ["Low", "Medium", "High", "Critical"],
            },
            "angry": {
                "type": "noul",
                "instructions": "Strong frustration or anger?",
            },
            "categories": {
                "type": "multilabel",
                "instructions": "Select all applicable categories",
                "criteria": ["PaymentIssue", "AccountAccess", "BugReport"],
            },
        },
        "model": target_model,
        "swap_verify": False,
    }

    resp = client.post("/api/v1/evaluate", json=payload)
    assert resp.status_code == 200, f"Failed: {resp.text}"

    data = resp.json()
    assert "model" in data
    assert "answers" in data
    assert "meta" in data
    assert data["meta"]["total_questions"] == 4
    assert data["meta"]["latency_ms"] > 0

    answers = data["answers"]

    # 1. choice 検証
    assert "department" in answers
    dept = answers["department"]
    assert dept["type"] == "choice"
    assert dept["choice"] in ("billing", "technical", "other")
    assert "probabilities" in dept
    assert dept["status"] == "SUCCESS"

    # 2. score 検証
    assert "urgency" in answers
    urgency = answers["urgency"]
    assert urgency["type"] == "score"
    assert isinstance(urgency["score"], (int, float))
    assert urgency["status"] == "SUCCESS"

    # 3. noul 検証
    assert "angry" in answers
    angry = answers["angry"]
    assert angry["type"] == "noul"
    assert angry["verdict"] in ("Yes", "No")
    assert 0.0 <= angry["noul"] <= 1.0

    # 4. multilabel 検証
    assert "categories" in answers
    cat = answers["categories"]
    assert cat["type"] == "multilabel"
    assert isinstance(cat["matched_labels"], list)


def test_evaluate_batch_alias_route(client: TestClient, target_model: str):
    """POST /api/evaluate 互換エイリアスルートの疎通・互換性テスト"""
    payload = {
        "state": "Hi, I have a question about my invoice.",
        "questions": {
            "is_billing": {
                "type": "noul",
                "instructions": "Is this inquiry related to billing?",
            },
            "dept": {
                "type": "choice",
                "criteria": ["Billing", "Support", "Sales"],
            },
        },
        "model": target_model,
    }

    resp = client.post("/api/evaluate", json=payload)
    assert resp.status_code == 200, f"Failed: {resp.text}"

    data = resp.json()
    assert "answers" in data
    assert "is_billing" in data["answers"]
    assert "dept" in data["answers"]
    assert data["meta"]["total_questions"] == 2


def test_evaluate_batch_validation_error(client: TestClient):
    """state や questions が欠落している不正リクエストに対する 422 バリデーションエラー検証"""
    # 1. state 欠落
    resp1 = client.post("/api/v1/evaluate", json={"questions": {}})
    assert resp1.status_code == 422

    # 2. questions 欠落
    resp2 = client.post("/api/v1/evaluate", json={"state": "Hello"})
    assert resp2.status_code == 422

    # 3. 不正な question type
    resp3 = client.post(
        "/api/v1/evaluate",
        json={
            "state": "Hello",
            "questions": {"q": {"type": "unknown_type"}},
        },
    )
    assert resp3.status_code == 422
