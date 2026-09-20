import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.verify_qwen3_full_suite import (
    query_qwen3,
    extract_token_logprob,
    run_test_1_position_swap,
    run_test_2_context_decay,
    run_test_3_sequential_queue_stress
)

def test_qwen3_thinking_suppress_and_single_forward():
    """Qwen3:8b が空思考タグPrefillにより単一Forward（<200ms）でLogit抽出できるかを検証"""
    prompt = "Is Tokyo the capital of Japan?\nA) Yes\nB) No\nAnswer:"
    # 初回ウォームアップ (Cold Start 吸収)
    query_qwen3([{"role": "user", "content": "warmup"}])
    res, latency_ms = query_qwen3([{"role": "user", "content": prompt}])
    
    assert "error" not in res, f"Query error: {res.get('error')}"
    assert "choices" in res and len(res["choices"]) > 0
    
    choice = res["choices"][0]
    logprobs = choice.get("logprobs")
    assert logprobs is not None, "Logprobs should not be None"
    
    items = logprobs["content"][0].get("top_logprobs", [])
    log_a = extract_token_logprob(items, "A")
    log_b = extract_token_logprob(items, "B")
    
    assert log_a > log_b, f"Expected A > B for Tokyo capital, got A={log_a}, B={log_b}"
    assert latency_ms < 200.0, f"Latency {latency_ms}ms exceeded 200ms threshold"

def test_qwen3_japanese_position_swap():
    """日本語ビジネス・コンプライアンス規程におけるA/Bスワップ一貫率が80%以上であることを検証"""
    result = run_test_1_position_swap()
    assert result["consistency_rate"] >= 80.0, f"Consistency rate {result['consistency_rate']}% below 80%"

def test_qwen3_context_decay():
    """日本語長文コンテキストにおいてSoft Limit（6000文字/~2000T）まで完全正解マージンが維持されるかを検証"""
    result = run_test_2_context_decay()
    # 6000文字（Soft Limit）までの4スケールがすべて正解であることを検証
    soft_limit_results = result["scales"][:4]
    assert all(r["verdict_correct"] for r in soft_limit_results), "Failed to maintain positive margin up to Soft Limit"
    # 12000文字（Hard Limit）ではマージンが減衰することを確認
    hard_limit_result = result["scales"][4]
    assert hard_limit_result["margin_pt"] <= 1.0, "Expected margin decay at Hard Limit"

def test_qwen3_sequential_queue():
    """直列キューイングにより10件並行リクエストがOOMゼロで100%完走することを検証"""
    result = run_test_3_sequential_queue_stress()
    assert result["all_success"] is True, "Some requests failed in sequential queue stress test"
