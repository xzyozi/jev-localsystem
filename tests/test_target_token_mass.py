"""対象トークン確率質量チェックのテストスイート

モデルが想定外のトークン（前置き文など）を出そうとして、対象トークンが
Top-Logprobs に現れない場合に、偽の SUCCESS ではなく INCONCLUSIVE を返すことを検証する。
Ollama / FastAPI には依存しない（偽バックエンドとログ確率の直接入力のみ）。
"""

import math
from typing import Dict, List, Tuple

import pytest

from jev import JudgePipeline, ResultMapper, ZeroDecodeClient
from jev.exceptions import BackendConnectionError
from jev.zero_decode import InferenceBackend

# 対象トークンを一切含まない分布（"The" 等の前置きを出そうとしているケース）
OFF_TARGET = {"The": -0.1, "I": -2.5}


class _FakeBackend(InferenceBackend):
    """固定の logprobs を返す偽バックエンド"""

    def __init__(self, logprobs: Dict[str, float]):
        self._logprobs = logprobs

    @property
    def is_cloud(self) -> bool:
        return False

    def forward(
        self, model: str, messages: List[Dict[str, str]], top_logprobs: int = 10
    ) -> Tuple[Dict[str, float], float]:
        return self._logprobs, 1.0


def _pipeline(logprobs: Dict[str, float]) -> JudgePipeline:
    return JudgePipeline(client=ZeroDecodeClient(backend=_FakeBackend(logprobs)))


# ==========================================
# get_target_mass
# ==========================================

def test_get_target_mass_sums_space_variants():
    logprobs = {"Yes": math.log(0.5), " Yes": math.log(0.2), "No": math.log(0.1)}
    mass = ResultMapper.get_target_mass(logprobs, ("Yes", "No"))
    assert mass == pytest.approx(0.8, abs=1e-4)


# ==========================================
# Noul
# ==========================================

def test_noul_off_target_is_inconclusive():
    res = ResultMapper.map_noul(OFF_TARGET, 5.0)
    assert res.status == "INCONCLUSIVE"
    assert res.verdict is None
    assert res.confidence is None
    assert "Target tokens not found" in (res.error_message or "")


def test_noul_normal_stays_success():
    res = ResultMapper.map_noul({"Yes": -0.05, "No": -3.2}, 5.0)
    assert res.status == "SUCCESS"
    assert res.verdict == "Yes"


def test_noul_min_target_mass_is_overridable():
    # 質量は約 0.0067 (No のみ)。下限 0.0 なら従来どおり SUCCESS となる
    res = ResultMapper.map_noul({"No": -5.0}, 5.0, min_target_mass=0.0)
    assert res.status == "SUCCESS"
    assert res.verdict == "No"


# ==========================================
# Choice
# ==========================================

def test_choice_off_target_is_inconclusive():
    res = ResultMapper.map_choice(OFF_TARGET, {"A": "Python", "B": "Go"}, 5.0)
    assert res.status == "INCONCLUSIVE"
    assert res.verdict is None


def test_choice_swap_side_off_target_is_inconclusive():
    fwd = {"A": -0.1, "B": -3.0}
    res = ResultMapper.map_choice(
        fwd,
        {"A": "Python", "B": "Go"},
        5.0,
        logprobs_swap=OFF_TARGET,
        symbol_map_swap={"A": "Go", "B": "Python"},
    )
    assert res.status == "INCONCLUSIVE"
    assert "swap mass" in (res.error_message or "")


def test_choice_normal_stays_success():
    fwd = {"A": -0.1, "B": -3.0}
    res = ResultMapper.map_choice(fwd, {"A": "Python", "B": "Go"}, 5.0)
    assert res.status == "SUCCESS"
    assert res.verdict == "Python"


# ==========================================
# Score
# ==========================================

def test_score_off_target_is_inconclusive():
    res = ResultMapper.map_score(OFF_TARGET, 5.0)
    assert res.status == "INCONCLUSIVE"
    assert res.verdict is None


def test_score_normal_stays_success():
    res = ResultMapper.map_score({"1": -5.0, "2": -4.0, "3": -3.0, "4": -1.0, "5": -0.2}, 5.0)
    assert res.status == "SUCCESS"
    assert res.verdict > 4.0


# ==========================================
# Multi-Label
# ==========================================

def test_multilabel_one_label_off_target_is_inconclusive():
    res = ResultMapper.map_multilabel(
        [("Security", {"Yes": -0.05, "No": -3.2}), ("Database", OFF_TARGET)],
        5.0,
    )
    assert res.status == "INCONCLUSIVE"
    assert res.verdict is None
    assert "Database" in (res.error_message or "")
    # 確率データ自体は details に残る（呼び出し側のデバッグ用）
    assert "Security" in res.details["probabilities"]


def test_multilabel_normal_stays_success():
    res = ResultMapper.map_multilabel(
        [("Security", {"Yes": -0.05, "No": -3.2}), ("UI", {"Yes": -4.0, "No": -0.02})],
        5.0,
    )
    assert res.status == "SUCCESS"
    assert res.verdict == ["Security"]


# ==========================================
# パイプライン経由（外部インターフェースは DTO のまま）
# ==========================================

def test_pipeline_noul_off_target_returns_inconclusive_dto():
    res = _pipeline(OFF_TARGET).judge_noul("テスト", "基準")
    assert res.status == "INCONCLUSIVE"
    assert res.task_type == "noul"


def test_pipeline_choice_off_target_returns_inconclusive_dto():
    res = _pipeline(OFF_TARGET).judge_choice("テスト", labels=["X", "Y"], swap_verify=True)
    assert res.status == "INCONCLUSIVE"


def test_pipeline_score_and_multilabel_off_target_return_inconclusive_dto():
    pipe = _pipeline(OFF_TARGET)
    assert pipe.judge_score("テスト").status == "INCONCLUSIVE"
    assert pipe.judge_multilabel("テスト", labels=["X"]).status == "INCONCLUSIVE"


def test_pipeline_normal_noul_is_unchanged():
    res = _pipeline({"Yes": -0.05, "No": -3.2}).judge_noul("テスト", "基準")
    assert res.status == "SUCCESS"
    assert res.verdict == "Yes"


def test_pipeline_backend_error_still_error_status():
    """バックエンド障害は従来どおり ERROR（今回の変更で挙動を変えていないことの確認）"""

    class _Down(_FakeBackend):
        def forward(self, model, messages, top_logprobs=10):
            raise BackendConnectionError("down")

    pipe = JudgePipeline(client=ZeroDecodeClient(backend=_Down({})))
    assert pipe.judge_noul("テスト", "基準").status == "ERROR"
