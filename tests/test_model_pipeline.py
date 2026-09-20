import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.verify_model_pipeline_suite import (
    run_test_token_and_latency,
    run_test_position_swap,
    run_test_multilabel_separation,
    run_test_context_decay,
    run_test_score_calibration,
    run_test_sequential_queue
)

def test_token_binding_and_latency(target_model):
    """[Issue #1, #2, #3] 空白対数和、思考抑制Prefill、および単一Forwardレイテンシ(<200ms)の検証"""
    res = run_test_token_and_latency(target_model)
    assert res["passed"] is True, f"Token binding or latency test failed for {target_model}"
    assert res["logprob_a"] > res["logprob_b"], "Tokyo capital question expected A > B"
    assert res["latency_ms"] < 200.0, f"Latency {res['latency_ms']:.1f}ms exceeded 200ms"

def test_position_swap_consistency(target_model):
    """[Issue #4] 日本語ビジネス・法務規程におけるA/Bスワップ位置バイアス整合率(>=75%)の検証"""
    res = run_test_position_swap(target_model)
    assert res["consistency_rate"] >= 75.0, f"Consistency rate {res['consistency_rate']:.1f}% below 75%"

def test_multilabel_sigmoid_separation(target_model):
    """[Issue #5] マルチラベルSigmoid判定における正負分離度ギャップ(>=6.0pt)の検証"""
    res = run_test_multilabel_separation(target_model)
    assert res["passed"] is True, f"Margin delta {res['margin_delta_pt']:.2f}pt below 6.0pt for {target_model}"

def test_context_decay_limits(target_model):
    """[Issue #6] 日本語長文コンテキストのSoft Limit(2000T)正解保持とHard Limit(4000T)減衰境界の検証"""
    res = run_test_context_decay(target_model)
    assert res["passed"] is True, f"Context decay limit test failed for {target_model}"

def test_score_calibration_monotonicity(target_model):
    """[Issue #7] 5段階品質評価Scoreタスクにおける期待値の完全単調減少性(High > Mid > Low)の検証"""
    res = run_test_score_calibration(target_model)
    assert res["monotonic"] is True, f"Score expectation failed monotonicity for {target_model}"

def test_sequential_queue_concurrency(target_model):
    """[Issue #8] 直列キュー排他制御による並行リクエスト投入時のOOMゼロ・全件完走の検証"""
    res = run_test_sequential_queue(target_model, count=5)
    assert res["all_success"] is True, f"Sequential queue concurrency test failed for {target_model}"
