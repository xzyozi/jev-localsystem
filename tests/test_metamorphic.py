"""決定論的コアロジックのメタモルフィックテストスイート (Issue #61)

Ollama / FastAPI に依存せず、ResultMapper および PromptBuilder が満たすべき
数学的・構造的不変性（対称性・単調性・スケーリング不変性・順列不変性・
State-First プレフィックス不変性）を検証する。
"""

import itertools
import math
from typing import Dict, List, Tuple

import pytest

from jev import JudgePipeline, JudgeRequestDTO, PromptBuilder, ResultMapper, ZeroDecodeClient
from jev.zero_decode import InferenceBackend


def _split_prob_with_space(symbol: str, total_prob: float, exact_ratio: float) -> Dict[str, float]:
    """合計確率 total_prob を無空白トークンと空白付きトークンに分配した対数確率辞書を返す。"""
    p_exact = max(total_prob * exact_ratio, 1e-12)
    p_space = max(total_prob * (1.0 - exact_ratio), 1e-12)
    res: Dict[str, float] = {}
    if exact_ratio > 0.0:
        res[symbol] = math.log(p_exact)
    if exact_ratio < 1.0:
        res[f" {symbol}"] = math.log(p_space)
    return res


# ==============================================================================
# 1. Noul: 対称性・空白バリアント等価性・対数確率シフト不変性
# ==============================================================================


@pytest.mark.parametrize(
    "p_yes,p_no",
    [
        (0.75, 0.15),
        (0.85, 0.05),
        (0.55, 0.35),
        (0.20, 0.65),
    ],
)
def test_noul_label_swap_symmetry(p_yes: float, p_no: float):
    """Yes と No の確率を入れ替えると、verdict が反転し margin の符号が反転して confidence は不変になる。"""
    orig_lp = {"Yes": math.log(p_yes), "No": math.log(p_no)}
    swapped_lp = {"Yes": math.log(p_no), "No": math.log(p_yes)}

    r_orig = ResultMapper.map_noul(orig_lp, latency_ms=10.0)
    r_swap = ResultMapper.map_noul(swapped_lp, latency_ms=10.0)

    assert r_orig.status == "SUCCESS"
    assert r_swap.status == "SUCCESS"
    assert r_orig.verdict != r_swap.verdict
    assert r_orig.confidence == pytest.approx(r_swap.confidence, abs=1e-4)
    assert r_orig.details is not None and r_swap.details is not None
    assert r_orig.details["margin"] == pytest.approx(-r_swap.details["margin"], abs=1e-4)
    assert r_orig.details["prob_yes"] == pytest.approx(r_swap.details["prob_no"], abs=1e-4)
    assert r_orig.details["prob_no"] == pytest.approx(r_swap.details["prob_yes"], abs=1e-4)


@pytest.mark.parametrize("ratio_yes", [0.0, 0.25, 0.5, 0.8, 1.0])
@pytest.mark.parametrize("ratio_no", [0.0, 0.3, 0.7, 1.0])
def test_noul_space_variant_allocation_invariance(ratio_yes: float, ratio_no: float):
    """'Yes'/' Yes' および 'No'/' No' 間の確率配分を変えても合成後の判定・確率・マージンは不変。"""
    p_yes, p_no = 0.68, 0.22
    base = ResultMapper.map_noul({"Yes": math.log(p_yes), "No": math.log(p_no)}, latency_ms=5.0)

    split_lp: Dict[str, float] = {}
    split_lp.update(_split_prob_with_space("Yes", p_yes, ratio_yes))
    split_lp.update(_split_prob_with_space("No", p_no, ratio_no))
    variant = ResultMapper.map_noul(split_lp, latency_ms=5.0)

    assert variant.status == base.status == "SUCCESS"
    assert variant.verdict == base.verdict
    assert variant.confidence == pytest.approx(base.confidence, abs=1e-4)
    assert variant.details == base.details


@pytest.mark.parametrize("shift", [-1.5, -0.7, -0.2, 0.0])
def test_noul_constant_logprob_shift_invariance(shift: float):
    """対象トークンすべてに共通定数を加算しても（質量>=MIN_TARGET_MASSの範囲で）正規化結果は不変。"""
    base_lp = {"Yes": math.log(0.7), "No": math.log(0.25)}
    shifted_lp = {tok: lp + shift for tok, lp in base_lp.items()}

    r_base = ResultMapper.map_noul(base_lp, latency_ms=5.0)
    r_shift = ResultMapper.map_noul(shifted_lp, latency_ms=5.0)

    assert r_base.status == r_shift.status == "SUCCESS"
    assert r_base.verdict == r_shift.verdict
    assert r_base.confidence == pytest.approx(r_shift.confidence, abs=1e-4)
    assert r_base.details == r_shift.details


# ==============================================================================
# 2. Choice: 多選択肢の順列不変性 & 対数確率シフト不変性
# ==============================================================================


@pytest.mark.parametrize("num_choices", [2, 3, 4, 6])
def test_choice_permutation_invariance(num_choices: int):
    """2〜6択の選択肢順序を任意の順列に並び替えても、勝者ラベルと各ラベル確率は不変。"""
    labels = [f"Label_{i}" for i in range(num_choices)]
    # 明確に異なる確率を割り当てる（合計 0.9 程度）
    probs_by_label = {lbl: 0.45 * (0.5**idx) for idx, lbl in enumerate(labels)}
    symbols = PromptBuilder.CHOICE_SYMBOLS[:num_choices]

    base_map = dict(zip(symbols, labels))
    base_lp = {sym: math.log(probs_by_label[lbl]) for sym, lbl in base_map.items()}
    base_res = ResultMapper.map_choice(base_lp, base_map, latency_ms=5.0)
    assert base_res.status == "SUCCESS"

    # 代表的な順列（逆順、巡回シフト、偶数奇数入替）で検証
    candidates = [
        list(reversed(labels)),
        labels[1:] + labels[:1],
        labels[-1:] + labels[:-1],
    ]
    if num_choices <= 4:
        candidates = [list(p) for p in itertools.permutations(labels)]

    for perm in candidates:
        perm_map = dict(zip(symbols, perm))
        perm_lp = {sym: math.log(probs_by_label[lbl]) for sym, lbl in perm_map.items()}
        perm_res = ResultMapper.map_choice(perm_lp, perm_map, latency_ms=5.0)

        assert perm_res.status == "SUCCESS"
        assert perm_res.verdict == base_res.verdict
        assert perm_res.confidence == pytest.approx(base_res.confidence, abs=1e-4)
        assert perm_res.details is not None and base_res.details is not None
        assert perm_res.details["probabilities"] == base_res.details["probabilities"]


def test_choice_constant_logprob_shift_invariance():
    """Choice の全対象シンボルに同一の対数定数を加算しても正規化確率・勝者は不変。"""
    symbol_map = {"A": "Alpha", "B": "Beta", "C": "Gamma"}
    base_lp = {"A": math.log(0.5), "B": math.log(0.3), "C": math.log(0.1)}
    shifted_lp = {k: v - 1.2 for k, v in base_lp.items()}  # 合計質量 ~ 0.27 >= 0.1

    r_base = ResultMapper.map_choice(base_lp, symbol_map, latency_ms=5.0)
    r_shift = ResultMapper.map_choice(shifted_lp, symbol_map, latency_ms=5.0)

    assert r_base.status == r_shift.status == "SUCCESS"
    assert r_base.verdict == r_shift.verdict == "Alpha"
    assert r_base.confidence == pytest.approx(r_shift.confidence, abs=1e-4)
    assert r_base.details is not None and r_shift.details is not None
    assert r_base.details["probabilities"] == r_shift.details["probabilities"]


# ==============================================================================
# 3. Score: 鏡像反転対称性・質量シフト単調性・温度平滑化挙動
# ==============================================================================


@pytest.mark.parametrize("levels", [3, 5, 7])
def test_score_mirror_inversion_symmetry(levels: int):
    """スコア 1..levels の分布を鏡像反転（k <-> levels+1-k）すると期待値 E' = (levels+1) - E を満たす。"""
    # 非対称な確率分布を構成（合計 0.9）
    weights = [float(i + 1) ** 2 for i in range(levels)]
    norm = 0.9 / sum(weights)
    probs = {str(i + 1): w * norm for i, w in enumerate(weights)}
    mirrored_probs = {str(levels - i): w * norm for i, w in enumerate(weights)}

    lp_orig = {k: math.log(v) for k, v in probs.items()}
    lp_mirror = {k: math.log(v) for k, v in mirrored_probs.items()}

    r_orig = ResultMapper.map_score(lp_orig, latency_ms=5.0, levels=levels)
    r_mirror = ResultMapper.map_score(lp_mirror, latency_ms=5.0, levels=levels)

    assert r_orig.status == "SUCCESS" and r_mirror.status == "SUCCESS"
    assert r_orig.verdict + r_mirror.verdict == pytest.approx(levels + 1, abs=2e-3)


def test_score_upward_probability_shift_monotonicity():
    """より高い評価段階へ確率質量を移すと期待値が単調増加する。"""
    low_bias = {"1": math.log(0.4), "2": math.log(0.3), "3": math.log(0.15), "4": math.log(0.04), "5": math.log(0.01)}
    mid_bias = {"1": math.log(0.1), "2": math.log(0.2), "3": math.log(0.4), "4": math.log(0.2), "5": math.log(0.1)}
    high_bias = {"1": math.log(0.01), "2": math.log(0.04), "3": math.log(0.15), "4": math.log(0.3), "5": math.log(0.4)}

    r_low = ResultMapper.map_score(low_bias, latency_ms=5.0)
    r_mid = ResultMapper.map_score(mid_bias, latency_ms=5.0)
    r_high = ResultMapper.map_score(high_bias, latency_ms=5.0)

    assert r_low.verdict < r_mid.verdict < r_high.verdict


@pytest.mark.parametrize("levels", [3, 5])
def test_score_temperature_smoothing_moves_toward_midpoint(levels: int):
    """temperature を上げるほど分布が平坦化され、期待値が一様平均 (levels+1)/2 に近づく。"""
    # 最高点側へ偏った分布
    probs = {str(i): 0.05 for i in range(1, levels)}
    probs[str(levels)] = 0.75
    lp = {k: math.log(v) for k, v in probs.items()}

    midpoint = (levels + 1) / 2.0
    temps = [0.5, 1.0, 2.0, 10.0]
    distances = []
    for t in temps:
        res = ResultMapper.map_score(lp, latency_ms=5.0, temperature=t, levels=levels)
        assert res.status == "SUCCESS"
        distances.append(abs(res.verdict - midpoint))

    assert all(distances[i] >= distances[i + 1] - 1e-3 for i in range(len(distances) - 1))


# ==============================================================================
# 4. Multi-Label: offset/threshold 単調性・ラベル順不変性
# ==============================================================================


def test_multilabel_offset_and_threshold_monotonicity():
    """offset 増加で確率は単調非減少、threshold 増加で採用ラベル集合は単調縮小（部分集合）になる。"""
    label_results = [
        ("Security", {"Yes": math.log(0.85), "No": math.log(0.10)}),
        ("Network", {"Yes": math.log(0.55), "No": math.log(0.40)}),
        ("Database", {"Yes": math.log(0.30), "No": math.log(0.65)}),
        ("Frontend", {"Yes": math.log(0.10), "No": math.log(0.85)}),
    ]

    offsets = [-2.0, 0.0, 2.0]
    prev_probs: Dict[str, float] = {}
    prev_verdict: set[str] = set()
    for off in offsets:
        res = ResultMapper.map_multilabel(label_results, latency_ms=5.0, threshold=0.5, offset=off)
        assert res.status == "SUCCESS"
        assert res.details is not None
        probs = res.details["probabilities"]
        if prev_probs:
            for lbl in probs:
                assert probs[lbl] >= prev_probs[lbl]
            assert prev_verdict.issubset(set(res.verdict))
        prev_probs = probs
        prev_verdict = set(res.verdict)

    thresholds = [0.2, 0.5, 0.8]
    prev_set = None
    for th in thresholds:
        res = ResultMapper.map_multilabel(label_results, latency_ms=5.0, threshold=th, offset=0.0)
        assert res.status == "SUCCESS"
        cur_set = set(res.verdict)
        if prev_set is not None:
            assert cur_set.issubset(prev_set)
        prev_set = cur_set


def test_multilabel_label_order_invariance():
    """入力ラベルの順番を入れ替えても、各ラベルの margin・prob・採用ラベル集合・confidence は不変。"""
    label_results = [
        ("Security", {"Yes": math.log(0.80), "No": math.log(0.15)}),
        ("Network", {"Yes": math.log(0.55), "No": math.log(0.40)}),
        ("Database", {"Yes": math.log(0.20), "No": math.log(0.75)}),
    ]
    r_orig = ResultMapper.map_multilabel(label_results, latency_ms=5.0)
    r_rev = ResultMapper.map_multilabel(list(reversed(label_results)), latency_ms=5.0)

    assert r_orig.status == r_rev.status == "SUCCESS"
    assert set(r_orig.verdict) == set(r_rev.verdict)
    assert r_orig.confidence == pytest.approx(r_rev.confidence, abs=1e-4)
    assert r_orig.details is not None and r_rev.details is not None
    assert r_orig.details["margins"] == r_rev.details["margins"]
    assert r_orig.details["probabilities"] == r_rev.details["probabilities"]


# ==============================================================================
# 5. PromptBuilder: State-First プレフィックス不変性 (Issue #22 契約)
# ==============================================================================


@pytest.mark.parametrize("model_name", ["qwen3:8b", "phi4-mini:latest"])
def test_prompt_builder_state_first_prefix_invariance(model_name: str):
    """同一 context_text なら rule/labels/swap/target_label を変えても user メッセージ先頭が一致する。"""
    shared_context = "第1条 本システムはすべてのアクセスに多要素認証を必須とする。"
    expected_prefix = f"【対象テキスト】\n{shared_context}"

    # Noul: rule_definition を変更しても先頭共通
    m_noul_1, _, _ = PromptBuilder.build_messages(
        JudgeRequestDTO(task_type="noul", context_text=shared_context, rule_definition="質問A"),
        model_name,
    )
    m_noul_2, _, _ = PromptBuilder.build_messages(
        JudgeRequestDTO(task_type="noul", context_text=shared_context, rule_definition="質問B"),
        model_name,
    )
    assert m_noul_1[0] == m_noul_2[0]
    assert m_noul_1[1]["content"].startswith(expected_prefix)
    assert m_noul_2[1]["content"].startswith(expected_prefix)

    # Choice: swap=False と swap=True で先頭共通
    req_choice = JudgeRequestDTO(
        task_type="choice",
        context_text=shared_context,
        rule_definition="最も適切な選択肢を選べ",
        labels=["案1", "案2", "案3"],
    )
    m_choice_fwd, _, _ = PromptBuilder.build_messages(req_choice, model_name, swap=False)
    m_choice_rev, _, _ = PromptBuilder.build_messages(req_choice, model_name, swap=True)
    assert m_choice_fwd[0] == m_choice_rev[0]
    assert m_choice_fwd[1]["content"].startswith(f"{expected_prefix}\n\n【分類基準】\n最も適切な選択肢を選べ")
    assert m_choice_rev[1]["content"].startswith(f"{expected_prefix}\n\n【分類基準】\n最も適切な選択肢を選べ")

    # Multi-Label: target_label が変わっても rule_definition まで完全一致
    req_ml = JudgeRequestDTO(
        task_type="multilabel",
        context_text=shared_context,
        rule_definition="障害カテゴリを分類せよ",
        labels=["Security", "Network", "Database"],
    )
    ml_prefixes = []
    for lbl in req_ml.labels:
        msgs, _, _ = PromptBuilder.build_messages(req_ml, model_name, single_label_target=lbl)
        assert msgs[1]["content"].startswith(f"{expected_prefix}\n\n【判定ルール】\n障害カテゴリを分類せよ")
        ml_prefixes.append(msgs[1]["content"].split("【評価カテゴリ】")[0])
    assert len(set(ml_prefixes)) == 1


# ==============================================================================
# 6. JudgePipeline E2E (偽バックエンド): 順列・順序不変性
# ==============================================================================


class _LabelAwareFakeBackend(InferenceBackend):
    """プロンプト内の選択肢定義を読み、特定ターゲットラベルを高確率にする偽バックエンド。"""

    def __init__(self, preferred_label: str):
        self._preferred_label = preferred_label

    @property
    def is_cloud(self) -> bool:
        return True  # VRAM/モデル接続チェックをバイパスして純粋にロジックを検証

    def forward(
        self, model: str, messages: List[Dict[str, str]], top_logprobs: int = 10
    ) -> Tuple[Dict[str, float], float]:
        user_msg = messages[1]["content"]
        # "【選択肢】" セクションから "- A: <label>" を解析
        if "【選択肢】" in user_msg:
            options_block = user_msg.split("【選択肢】\n", 1)[1].split("\n\n", 1)[0]
            out: Dict[str, float] = {}
            for line in options_block.splitlines():
                if line.startswith("- ") and ": " in line:
                    sym, lbl = line[2:].split(": ", 1)
                    out[sym] = math.log(0.80) if lbl == self._preferred_label else math.log(0.05)
            return out, 1.0
        # Multi-Label 判定
        if "カテゴリ「Security」" in user_msg:
            return {"Yes": math.log(0.90), "No": math.log(0.08)}, 1.0
        return {"Yes": math.log(0.10), "No": math.log(0.85)}, 1.0


@pytest.mark.parametrize(
    "labels_perm",
    [
        ["Opt1", "Opt2", "Opt3"],
        ["Opt2", "Opt3", "Opt1"],
        ["Opt3", "Opt1", "Opt2"],
    ],
)
def test_pipeline_choice_permutation_invariance(labels_perm: List[str]):
    """パイプライン全体で Choice 選択肢順列・swap_verify を通しても勝者と確率が一致する。"""
    pipeline = JudgePipeline(client=ZeroDecodeClient(backend=_LabelAwareFakeBackend("Opt2")))
    res = pipeline.judge_choice(
        context_text="評価対象コンテキスト",
        labels=labels_perm,
        swap_verify=True,
    )
    assert res.status == "SUCCESS"
    assert res.verdict == "Opt2"
    assert res.details is not None
    assert res.details["is_consistent"] is True
    assert res.details["probabilities"]["Opt2"] == pytest.approx(0.80 / 0.90, abs=1e-4)
