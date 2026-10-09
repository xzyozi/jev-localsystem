"""モデルプロファイル・バックエンド契約のテスト (Issue #23)

- モデル名ごとの Prefill / 思考モデル判定は、外出し前の PromptBuilder の挙動と同一
- 利用者定義のプロファイルを追加できる
- pipeline が target_tokens をバックエンドへ渡す（既存の forward のみ実装のバックエンドも動作する）

Ollama / FastAPI には依存しない。
"""

from typing import Dict, List, Optional, Sequence, Tuple

import pytest

from jev import JudgePipeline, PromptBuilder, ZeroDecodeClient
from jev.model_profiles import (
    PREFILL_THINK_TAG,
    PREFILL_THOUGHT_TAG,
    ModelProfile,
    clear_custom_model_profiles,
    is_thinking,
    register_model_profile,
    resolve_prefill,
)
from jev.zero_decode import InferenceBackend


@pytest.fixture(autouse=True)
def _reset_profiles():
    clear_custom_model_profiles()
    yield
    clear_custom_model_profiles()


# ==========================================
# 1. 外出し前と同一の判定結果（特性テスト）
# ==========================================

@pytest.mark.parametrize(
    "model,prefill,thinking",
    [
        ("qwen3:8b", PREFILL_THINK_TAG, True),
        ("Qwen3-14B-Instruct", PREFILL_THINK_TAG, True),
        ("deepseek-r1:14b", PREFILL_THOUGHT_TAG, True),
        ("DeepSeek-R1-Distill", PREFILL_THOUGHT_TAG, True),
        ("gemma3:12b", PREFILL_THOUGHT_TAG, False),
        ("phi4-mini:latest", None, False),
        ("qwen2.5-coder:14b", None, False),
        ("gpt-4o-mini", None, False),
        ("", None, False),
    ],
)
def test_builtin_profiles_match_previous_behavior(model: str, prefill: Optional[str], thinking: bool):
    assert resolve_prefill(model) == prefill
    assert is_thinking(model) is thinking
    # PromptBuilder の公開メソッドも同じ結果を返す（後方互換）
    assert PromptBuilder.get_prefill_string(model) == prefill
    assert PromptBuilder.is_thinking_model(model) is thinking


def test_prompt_builder_constants_are_preserved():
    assert PromptBuilder.PREFILL_THINK_TAG == "<think>\n\n</think>\n"
    assert PromptBuilder.PREFILL_THOUGHT_TAG == "<thought>\n\n</thought>\n"
    assert PromptBuilder.THINKING_MODELS == ("qwen3", "deepseek-r1")


# ==========================================
# 2. 利用者定義プロファイル
# ==========================================

def test_registered_profile_is_used_and_overrides_builtin():
    register_model_profile(ModelProfile(name="my-model", match=("my-model",), thinking=True, prefill="<x></x>"))
    assert resolve_prefill("my-model:7b") == "<x></x>"
    assert is_thinking("my-model:7b") is True
    # 組み込みを上書きできる
    register_model_profile(ModelProfile(name="qwen3-nothink", match=("qwen3",), prefill=None))
    assert resolve_prefill("qwen3:8b") is None


def test_clear_custom_profiles_restores_builtin():
    register_model_profile(ModelProfile(name="qwen3-nothink", match=("qwen3",), prefill=None))
    clear_custom_model_profiles()
    assert resolve_prefill("qwen3:8b") == PREFILL_THINK_TAG


# ==========================================
# 3. バックエンド契約: target_tokens の受け渡し
# ==========================================

class _LegacyBackend(InferenceBackend):
    """forward のみ実装した既存形式のバックエンド（target_tokens を知らない）"""

    def __init__(self) -> None:
        self.forward_calls = 0

    @property
    def is_cloud(self) -> bool:
        return False

    def forward(
        self, model: str, messages: List[Dict[str, str]], top_logprobs: int = 10
    ) -> Tuple[Dict[str, float], float]:
        self.forward_calls += 1
        return {"Yes": -0.05, "No": -3.0, "A": -0.1, "B": -3.0, "1": -0.1, "2": -3.0, "3": -3.0}, 1.0


class _ConstrainedBackend(_LegacyBackend):
    """forward_constrained をオーバーライドして target_tokens を記録するバックエンド"""

    def __init__(self) -> None:
        super().__init__()
        self.targets: List[List[str]] = []

    def forward_constrained(
        self,
        model: str,
        messages: List[Dict[str, str]],
        target_tokens: Sequence[str],
        top_logprobs: int = 10,
    ) -> Tuple[Dict[str, float], float]:
        self.targets.append(list(target_tokens))
        return self.forward(model, messages, top_logprobs=top_logprobs)


def test_legacy_backend_still_works_through_pipeline():
    backend = _LegacyBackend()
    pipe = JudgePipeline(client=ZeroDecodeClient(backend=backend))
    assert pipe.judge_noul("x", "基準").status == "SUCCESS"
    assert backend.forward_calls == 1


def test_pipeline_passes_target_tokens_for_each_task():
    backend = _ConstrainedBackend()
    pipe = JudgePipeline(client=ZeroDecodeClient(backend=backend))

    pipe.judge_noul("x", "基準")
    assert backend.targets[-1] == ["Yes", "No", " Yes", " No"]

    pipe.judge_choice("x", labels=["a", "b"])
    assert backend.targets[-1] == ["A", " A", "B", " B"]

    pipe.judge_score("x", score_levels=3)
    assert backend.targets[-1] == ["1", " 1", "2", " 2", "3", " 3"]

    before = len(backend.targets)
    pipe.judge_multilabel("x", labels=["l1", "l2"])
    assert len(backend.targets) == before + 2
    assert backend.targets[-1] == ["Yes", "No", " Yes", " No"]


def test_choice_swap_passes_target_tokens_for_both_orders():
    backend = _ConstrainedBackend()
    pipe = JudgePipeline(client=ZeroDecodeClient(backend=backend))
    pipe.judge_choice("x", labels=["a", "b"], swap_verify=True)
    assert len(backend.targets) == 2


def test_client_without_target_tokens_calls_plain_forward():
    backend = _ConstrainedBackend()
    client = ZeroDecodeClient(backend=backend)
    client.forward("m", [{"role": "user", "content": "x"}])
    assert backend.targets == []
    assert backend.forward_calls == 1
