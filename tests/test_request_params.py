"""リクエスト指定値の検証・反映テスト (Issue #19)

- Choice の選択肢数（2〜8）と Multi-Label のラベル数を DTO で検証する（黙った切り捨ての防止）
- Multi-Label の threshold / offset が判定ロジックに反映される
- Score の段数（score_levels）がプロンプトと集計に反映される

Ollama / FastAPI には依存しない（偽バックエンドのみ）。
"""

import math
from typing import Dict, List, Tuple

from pydantic import ValidationError
import pytest

from jev import JudgePipeline, JudgeRequestDTO, PromptBuilder, ZeroDecodeClient
from jev.dto import ChoiceQuestionDTO, MultiLabelQuestionDTO, ScoreQuestionDTO
from jev.zero_decode import InferenceBackend


class _RecordingBackend(InferenceBackend):
    """固定の logprobs を返し、送信されたメッセージを記録する偽バックエンド"""

    def __init__(self, logprobs: Dict[str, float]):
        self._logprobs = logprobs
        self.calls: List[List[Dict[str, str]]] = []

    @property
    def is_cloud(self) -> bool:
        return False

    def forward(
        self, model: str, messages: List[Dict[str, str]], top_logprobs: int = 10
    ) -> Tuple[Dict[str, float], float]:
        self.calls.append(messages)
        return self._logprobs, 1.0


def _pipeline(logprobs: Dict[str, float]) -> Tuple[JudgePipeline, _RecordingBackend]:
    backend = _RecordingBackend(logprobs)
    return JudgePipeline(client=ZeroDecodeClient(backend=backend)), backend


# ==========================================
# 1. ラベル数の検証
# ==========================================

@pytest.mark.parametrize("count", [0, 1, 9, 10])
def test_choice_label_count_out_of_range_is_rejected(count: int):
    with pytest.raises(ValidationError):
        JudgeRequestDTO(task_type="choice", context_text="x", labels=[f"L{i}" for i in range(count)])


@pytest.mark.parametrize("count", [2, 8])
def test_choice_label_count_in_range_is_accepted(count: int):
    req = JudgeRequestDTO(task_type="choice", context_text="x", labels=[f"L{i}" for i in range(count)])
    assert len(req.labels) == count


def test_multilabel_requires_at_least_one_label():
    with pytest.raises(ValidationError):
        JudgeRequestDTO(task_type="multilabel", context_text="x", labels=[])


def test_noul_and_score_do_not_require_labels():
    JudgeRequestDTO(task_type="noul", context_text="x")
    JudgeRequestDTO(task_type="score", context_text="x")


def test_prompt_builder_never_truncates_labels_silently():
    """DTO 検証を迂回した場合でも、9個以上のラベルを黙って切り捨てずエラーにする"""
    req = JudgeRequestDTO.model_construct(
        task_type="choice", context_text="x", labels=[f"L{i}" for i in range(9)], rule_definition="",
    )
    with pytest.raises(ValueError):
        PromptBuilder.build_messages(req, "phi4-mini")


def test_choice_question_dto_size_limits():
    with pytest.raises(ValidationError):
        ChoiceQuestionDTO(type="choice", criteria=[f"L{i}" for i in range(9)])
    with pytest.raises(ValidationError):
        ChoiceQuestionDTO(type="choice", criteria={f"L{i}": "" for i in range(9)})
    with pytest.raises(ValidationError):
        ChoiceQuestionDTO(type="choice", criteria=["only-one"])
    assert ChoiceQuestionDTO(type="choice", criteria=["a", "b"]).criteria == ["a", "b"]


def test_multilabel_question_dto_limits():
    with pytest.raises(ValidationError):
        MultiLabelQuestionDTO(type="multilabel", criteria=[])
    with pytest.raises(ValidationError):
        MultiLabelQuestionDTO(type="multilabel", criteria=["a"], threshold=1.5)


# ==========================================
# 2. threshold / offset の反映
# ==========================================

_P70 = {"Yes": math.log(0.7), "No": math.log(0.3)}  # Sigmoid 確率 0.7


def test_multilabel_threshold_is_applied():
    pipe, _ = _pipeline(_P70)
    low = pipe.judge_multilabel("x", labels=["Security"], threshold=0.5)
    high = pipe.judge_multilabel("x", labels=["Security"], threshold=0.9)
    assert low.verdict == ["Security"]
    assert high.verdict == []
    assert high.details["threshold"] == 0.9


def test_multilabel_default_threshold_is_unchanged():
    pipe, _ = _pipeline(_P70)
    res = pipe.judge_multilabel("x", labels=["Security"])
    assert res.verdict == ["Security"]
    assert res.details["threshold"] == 0.5


def test_multilabel_offset_is_applied():
    pipe, _ = _pipeline(_P70)
    res = pipe.judge_multilabel("x", labels=["Security"], offset=-2.0)
    assert res.verdict == []
    assert res.details["offset"] == -2.0


def test_threshold_range_is_validated():
    with pytest.raises(ValidationError):
        JudgeRequestDTO(task_type="multilabel", context_text="x", labels=["a"], threshold=1.5)


# ==========================================
# 3. Score の段数
# ==========================================

def test_score_default_prompt_is_unchanged_five_levels():
    req = JudgeRequestDTO(task_type="score", context_text="本文")
    messages, targets, mapping = PromptBuilder.build_messages(req, "phi4-mini")
    system = messages[0]["content"]
    assert "1〜5の5段階" in system
    assert "5: 極めて高い / 完璧" in system
    assert messages[1]["content"].endswith("評価スコア (1-5):")
    assert list(mapping) == ["1", "2", "3", "4", "5"]
    assert "5" in targets and " 5" in targets


def test_score_levels_are_reflected_in_prompt_and_mapping():
    req = JudgeRequestDTO(task_type="score", context_text="本文", score_levels=3)
    messages, targets, mapping = PromptBuilder.build_messages(req, "phi4-mini")
    assert "1〜3の3段階" in messages[0]["content"]
    assert messages[1]["content"].endswith("評価スコア (1-3):")
    assert list(mapping) == ["1", "2", "3"]
    assert "4" not in targets


def test_score_with_three_levels_uses_three_level_distribution():
    logprobs = {"1": math.log(0.1), "2": math.log(0.2), "3": math.log(0.7)}
    pipe, backend = _pipeline(logprobs)
    res = pipe.judge_score("本文", score_levels=3)
    assert res.status == "SUCCESS"
    assert list(res.details["distribution"]) == ["1", "2", "3"]
    assert res.verdict == pytest.approx(2.6, abs=1e-3)  # 1*0.1 + 2*0.2 + 3*0.7
    assert "1〜3の3段階" in backend.calls[0][0]["content"]


def test_score_mass_check_uses_requested_levels():
    """3段階の依頼に対して '4' '5' だけが出る分布は、対象外トークンとして INCONCLUSIVE になる"""
    pipe, _ = _pipeline({"4": -0.1, "5": -2.0})
    res = pipe.judge_score("本文", score_levels=3)
    assert res.status == "INCONCLUSIVE"


@pytest.mark.parametrize("levels", [1, 10])
def test_score_levels_range_is_validated(levels: int):
    with pytest.raises(ValidationError):
        JudgeRequestDTO(task_type="score", context_text="x", score_levels=levels)


def test_score_question_dto_levels():
    assert ScoreQuestionDTO(type="score").levels == 5
    assert ScoreQuestionDTO(type="score", criteria=[]).levels == 5
    assert ScoreQuestionDTO(type="score", criteria=["Low", "Medium", "High", "Critical"]).levels == 4
    assert ScoreQuestionDTO(type="score", criteria={"a": "x", "b": "y", "c": "z"}).levels == 3
    with pytest.raises(ValidationError):
        ScoreQuestionDTO(type="score", criteria=["only-one"])
    with pytest.raises(ValidationError):
        ScoreQuestionDTO(type="score", criteria=[str(i) for i in range(10)])
