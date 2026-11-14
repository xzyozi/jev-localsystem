"""[Layer 2] モデル能力向けメタモルフィック評価テスト (Issue #65)."""

import pytest

from jev import JudgePipeline
from scripts.verify_model_metamorphic import (
    run_choice_cyclic_eval,
    run_noise_robustness_eval,
    run_polarity_flip_eval,
)

pytestmark = pytest.mark.ollama


def test_polarity_flip_consistency(target_model: str) -> None:
    """Noul タスクにおける肯定・否定ペアの極性反転一貫性率を検証する。"""
    pipeline = JudgePipeline(default_model=target_model)
    summary = run_polarity_flip_eval(pipeline, target_model)

    assert summary["flip_consistency_rate"] >= 80.0
    assert summary["both_correct_rate"] >= 80.0
    assert summary["margin_sign_flip_rate"] >= 80.0


def test_choice_cyclic_permutation_consistency(target_model: str) -> None:
    """3〜4択 Choice タスクにおける巡回置換不変性率と正答率を検証する。"""
    pipeline = JudgePipeline(default_model=target_model)
    summary = run_choice_cyclic_eval(pipeline, target_model)

    assert summary["cyclic_consistency_rate"] >= 75.0
    assert summary["shift_accuracy_rate"] >= 80.0


def test_noise_robustness_consistency(target_model: str) -> None:
    """軽微な表記ゆれ・無関係ノイズに対する判定維持率と信頼度安定性を検証する。"""
    pipeline = JudgePipeline(default_model=target_model)
    summary = run_noise_robustness_eval(pipeline, target_model)

    assert summary["verdict_preservation_rate"] >= 80.0
    assert summary["max_confidence_delta"] <= 0.1
