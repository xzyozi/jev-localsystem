"""JEV (Judge & Evaluation via Vectorized-logits) コアパイプライン包括テストスイート

詳細設計書 (JEV-DD-001) の全仕様、契約、例外、および実機PoC検証知見を検証する。
- 単体テスト: DTO検証、リミッター契約、セマフォ排他制御、Prefill注入、確率正規化、スワップ不一致検知
- 実機統合テスト: Ollama 実機による4大判定タスク (Noul, Choice, Score, Multi-Label)
"""

import math
import threading

import pytest

from jev import (
    JudgePipeline,
    JudgeRequestDTO,
    PayloadTooLargeError,
    PromptBuilder,
    QueueTimeoutError,
    ResultMapper,
    VRAMManager,
)

# ==========================================
# 1. DTO & Validation テスト
# ==========================================

def test_dto_validation():
    """リクエストDTOのバリデーション契約の検証"""
    req = JudgeRequestDTO(
        task_type="noul",
        context_text="テストコンテキスト",
        temperature=1.0,
    )
    assert req.task_type == "noul"
    assert req.temperature == 1.0

    # temperature <= 0.0 は無効
    with pytest.raises(ValueError):
        JudgeRequestDTO(
            task_type="noul",
            context_text="テスト",
            temperature=0.0,
        )


# ==========================================
# 2. VRAM Manager & リミッター契約テスト
# ==========================================

def test_vram_manager_hard_limit():
    """ハードリミット (12,000文字 / 4,000T) 超過時に PayloadTooLargeError を即座に送出すること"""
    vm = VRAMManager(hard_char_limit=100)
    with pytest.raises(PayloadTooLargeError):
        vm.validate_payload_limits("X" * 101)


def test_vram_manager_concurrency_and_timeout():
    """直列セマフォにより同時実行が排他制御され、待機タイムアウト時に QueueTimeoutError となること"""
    vm = VRAMManager(default_timeout_sec=0.1)

    with vm.acquire():
        timed_out = False

        def worker():
            nonlocal timed_out
            try:
                with vm.acquire(timeout=0.05):
                    pass
            except QueueTimeoutError:
                timed_out = True

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        assert timed_out is True


# ==========================================
# 3. Prompt Builder & Prefill テスト
# ==========================================

def test_prompt_builder_thinking_prefill():
    """qwen3等の思考モデルに対して思考完了タグ Prefill が assistant ロールに注入されること"""
    req = JudgeRequestDTO(task_type="noul", context_text="テスト")
    messages, targets, _ = PromptBuilder.build_messages(req, "qwen3:8b")

    assert len(messages) == 3
    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == "<think>\n\n</think>\n"
    assert "Yes" in targets and " Yes" in targets


def test_prompt_builder_choice_swap():
    """Choiceタスクで swap=True の場合に選択肢の対応が反転すること"""
    req = JudgeRequestDTO(task_type="choice", context_text="コード", labels=["Python", "Go"])
    _, _, map_fwd = PromptBuilder.build_messages(req, "phi4-mini:latest", swap=False)
    _, _, map_swp = PromptBuilder.build_messages(req, "phi4-mini:latest", swap=True)

    assert map_fwd["A"] == "Python" and map_fwd["B"] == "Go"
    assert map_swp["A"] == "Go" and map_swp["B"] == "Python"


# ==========================================
# 4. Result Mapper & 確率計算テスト
# ==========================================

def test_result_mapper_space_variant_sum():
    """空白バリアント対数和 P('A') + P(' A') が数学的に正しく合算されること (Issue #1)"""
    # log(0.5) ≒ -0.6931, log(0.3) ≒ -1.20397 -> 合計 0.8
    logprobs = {"A": math.log(0.5), " A": math.log(0.3)}
    prob = ResultMapper.get_token_prob(logprobs, "A")
    assert pytest.approx(prob, 1e-4) == 0.8


def test_result_mapper_choice_swap_inconclusive():
    """位置バイアスでスワップ照合結果が不一致の場合、INCONCLUSIVE ステータスを返すこと (Issue #4)"""
    lp_fwd = {"A": -0.1, "B": -3.0}
    map_fwd = {"A": "First", "B": "Second"}

    # スワップ側でも位置バイアスにより先頭の 'A'（=Second）を選んでしまい不一致
    lp_swp = {"A": -0.1, "B": -3.0}
    map_swp = {"A": "Second", "B": "First"}

    res = ResultMapper.map_choice(lp_fwd, map_fwd, latency_ms=10.0, logprobs_swap=lp_swp, symbol_map_swap=map_swp)
    assert res.status == "INCONCLUSIVE"
    assert res.verdict is None
    assert res.details["is_consistent"] is False


def test_result_mapper_score_calibration():
    """Scoreタスクの確率加重期待値キャリブレーションが単調性を維持すること (Issue #7)"""
    # 5寄り
    lp_high = {"1": -5.0, "2": -4.0, "3": -3.0, "4": -1.0, "5": -0.2}
    res_high = ResultMapper.map_score(lp_high, 5.0)

    # 1寄り
    lp_low = {"1": -0.2, "2": -1.0, "3": -3.0, "4": -4.0, "5": -5.0}
    res_low = ResultMapper.map_score(lp_low, 5.0)

    assert res_high.verdict > 4.0
    assert res_low.verdict < 2.0
    assert res_high.verdict > res_low.verdict


# ==========================================
# 5. 実機 Ollama 統合テスト (E2E)
# ==========================================

def test_pipeline_e2e_real_model(target_model: str):
    """実際のローカル推論バックエンド（Ollama）を用いたパイプライン統合検証"""
    pipeline = JudgePipeline(default_model=target_model)

    # A. Noul
    res_noul = pipeline.judge_noul(
        context_text="2026年9月20日にシステムバックアップが正常完了しました。",
        rule_definition="バックアップが成功したかどうかを判定してください。",
        model=target_model,
    )
    assert res_noul.status == "SUCCESS"
    assert res_noul.verdict == "Yes"
    assert res_noul.latency_ms > 0

    # B. Choice (スワップ検証付き)
    res_choice = pipeline.judge_choice(
        context_text="def add(a, b): return a + b",
        labels=["Python", "SQL", "HTML"],
        swap_verify=True,
        model=target_model,
    )
    assert res_choice.status == "SUCCESS"
    assert res_choice.verdict == "Python"
    assert res_choice.details["swap_verified"] is True

    # C. Score
    res_score = pipeline.judge_score(
        context_text="優れたドキュメントと堅牢なテスト、見通しの良い設計です。",
        rule_definition="コード品質を1〜5で評価してください。",
        model=target_model,
    )
    assert res_score.status == "SUCCESS"
    assert 1.0 <= res_score.verdict <= 5.0
    assert res_score.details["most_likely"] in ["1", "2", "3", "4", "5"]

    # D. Multi-Label
    res_ml = pipeline.judge_multilabel(
        context_text="SQLインジェクション脆弱性を修正し、データベース接続プールを最適化しました。",
        labels=["Security", "Database", "UI/UX"],
        model=target_model,
    )
    assert res_ml.status == "SUCCESS"
    assert "Security" in res_ml.verdict or "Database" in res_ml.verdict
    assert "UI/UX" not in res_ml.verdict
