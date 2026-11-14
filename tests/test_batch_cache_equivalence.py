"""[Layer 1] バッチ連続推論（KVキャッシュあり）vs 単発推論（キャッシュリセット）等価性テスト (Issue #64)."""

import pytest

from jev import JudgePipeline
from scripts.verify_batch_cache_equivalence import (
    build_context,
    run_evaluate_equivalence,
    run_multilabel_pipeline_equivalence,
)

pytestmark = pytest.mark.ollama

_SCALES = [
    ("short_1000c", 1000),
    ("medium_3000c", 3000),
]


@pytest.fixture
def warmup_after_cache_reset(target_model: str):
    """キャッシュリセットを伴う検証後にモデルをウォーム状態へ戻し、後続テストの初回遅延を防ぐ。"""
    yield
    JudgePipeline(default_model=target_model).judge_noul("1+1=2", model=target_model)


@pytest.mark.parametrize(("state_label", "chars"), _SCALES)
def test_evaluate_batch_vs_single_equivalence(
    target_model: str,
    warmup_after_cache_reset: None,
    state_label: str,
    chars: int,
) -> None:
    """`/api/v1/evaluate` の混合バッチ推論と単発リセット推論の等価性を検証する。"""
    result = run_evaluate_equivalence(target_model, state_label, build_context(chars))
    cmp = result["comparisons"]

    assert cmp["choice"]["verdict_match"] is True
    assert cmp["choice"]["max_prob_diff"] <= 0.01

    assert cmp["score"]["score_diff"] <= 0.05
    assert cmp["score"]["max_prob_diff"] <= 0.01

    assert cmp["noul"]["verdict_match"] is True
    assert cmp["noul"]["prob_diff"] is not None
    assert cmp["noul"]["prob_diff"] <= 0.01

    assert cmp["multilabel"]["matched_match"] is True
    assert cmp["multilabel"]["max_prob_diff"] <= 0.01
    assert cmp["multilabel"]["max_margin_diff"] <= 0.05


@pytest.mark.parametrize(("state_label", "chars"), _SCALES)
def test_pipeline_multilabel_batch_vs_single_equivalence(
    target_model: str,
    warmup_after_cache_reset: None,
    state_label: str,
    chars: int,
) -> None:
    """`JudgePipeline.judge_multilabel` の連続実行と単発リセット実行の等価性を検証する。"""
    result = run_multilabel_pipeline_equivalence(target_model, state_label, build_context(chars))

    assert result["matched_match"] is True
    assert result["max_prob_diff"] <= 0.01
    assert result["max_margin_diff"] <= 0.05
