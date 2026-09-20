"""JEV REST API サーバー包括テストスイート

FastAPI TestClient を用いて全エンドポイントの正常系・異常系・HTTPステータスコードを検証する。
"""

from fastapi.testclient import TestClient
import pytest

from jev.server.app import app


@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient フィクスチャ"""
    return TestClient(app)


# ==========================================
# 1. 運用監視系エンドポイントテスト
# ==========================================

def test_api_health(client: TestClient):
    """GET /health の検証"""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("ok", "degraded")
    assert "backend_connected" in data
    assert data["default_model"] == "qwen3:8b"


def test_api_vram_metrics(client: TestClient):
    """GET /api/v1/vram/metrics の検証"""
    resp = client.get("/api/v1/vram/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "waiting_queue_count" in data
    assert "is_running" in data
    assert "total_processed" in data
    assert isinstance(data["waiting_queue_count"], int)


def test_api_models(client: TestClient):
    """GET /api/v1/models の検証"""
    resp = client.get("/api/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["models"]) >= 2
    tiers = [m["tier"] for m in data["models"]]
    assert any("Tier 1" in t for t in tiers)
    assert any("Tier 2" in t for t in tiers)


# ==========================================
# 2. 判定タスクエンドポイント正常系テスト (E2E)
# ==========================================

def test_api_noul(client: TestClient, target_model: str):
    """POST /api/v1/noul の検証"""
    payload = {
        "context_text": "2026年9月21日 バックアッププロセスが正常終了しました。",
        "rule_definition": "バックアップ成功かを判定してください。",
        "model": target_model,
    }
    resp = client.post("/api/v1/noul", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["task_type"] == "noul"
    assert data["status"] == "SUCCESS"
    assert data["verdict"] == "Yes"
    assert data["latency_ms"] > 0


def test_api_choice(client: TestClient, target_model: str):
    """POST /api/v1/choice の検証 (スワップ検証付き)"""
    payload = {
        "context_text": "def add(x, y): return x + y",
        "labels": ["Python", "SQL", "HTML"],
        "swap_verify": True,
        "model": target_model,
    }
    resp = client.post("/api/v1/choice", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["task_type"] == "choice"
    assert data["status"] == "SUCCESS"
    assert data["verdict"] == "Python"
    assert data["details"]["swap_verified"] is True


def test_api_score(client: TestClient, target_model: str):
    """POST /api/v1/score の検証"""
    payload = {
        "context_text": "完璧な設計と堅牢なテストカバレッジです。",
        "rule_definition": "品質を1〜5で評価してください。",
        "temperature": 1.0,
        "model": target_model,
    }
    resp = client.post("/api/v1/score", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["task_type"] == "score"
    assert data["status"] == "SUCCESS"
    assert 1.0 <= data["verdict"] <= 5.0


def test_api_multilabel(client: TestClient, target_model: str):
    """POST /api/v1/multilabel の検証"""
    payload = {
        "context_text": "XSSおよびSQLインジェクション対策のサニタイズ処理を追加しました。",
        "labels": ["Security", "Database", "Design"],
        "threshold": 0.5,
        "model": target_model,
    }
    resp = client.post("/api/v1/multilabel", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["task_type"] == "multilabel"
    assert data["status"] == "SUCCESS"
    assert isinstance(data["verdict"], list)
    assert "Security" in data["verdict"]


def test_api_judge_unified(client: TestClient, target_model: str):
    """POST /api/v1/judge (統合エンドポイント) の検証"""
    payload = {
        "task_type": "noul",
        "context_text": "侵入検知システムが不正なパケットを遮断しました。",
        "rule_definition": "セキュリティインシデント検知であるかを判定してください。",
        "model": target_model,
    }
    resp = client.post("/api/v1/judge", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["task_type"] == "noul"
    assert data["status"] == "SUCCESS"
    assert data["verdict"] == "Yes"


# ==========================================
# 3. 異常系・HTTPステータスコードテスト
# ==========================================

def test_api_payload_too_large_413(client: TestClient):
    """コンテキスト長がハードリミット超過（12,000文字超）の場合に HTTP 413 が返却されること"""
    payload = {
        "context_text": "A" * 12001,
        "rule_definition": "テスト",
    }
    resp = client.post("/api/v1/noul", json=payload)
    assert resp.status_code == 413
    data = resp.json()
    assert data["error_type"] == "PayloadTooLargeError"
    assert "hard limit" in data["detail"]


def test_api_validation_error_422(client: TestClient):
    """選択肢ラベル不足（min_length < 2）などのバリデーション不正で HTTP 422 が返却されること"""
    # Choiceで選択肢が1つのみ
    payload_invalid_choice = {
        "context_text": "テスト",
        "labels": ["OnlyOne"],
    }
    resp = client.post("/api/v1/choice", json=payload_invalid_choice)
    assert resp.status_code == 422
