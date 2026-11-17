"""プロンプト言語・記号テンプレートのテスト (Issue #71)

- 既定テンプレート ('ja') と従来 PromptBuilder ハードコード出力のバイト単位等価性（4タスク + swap）
- 英語テンプレート ('en') の State-First 構造（Prefix Caching 不変性）、4タスク生成、JudgePipeline 統合動作
- カスタム choice_symbols（有効ケース / 重複・空・記号不足・ラベル超過の検証）
- register_prompt_template / resolve_prompt_template / clear_custom_prompt_templates の登録・解決・リセット

Ollama / FastAPI には依存しない（偽バックエンドのみ）。
"""

from dataclasses import replace
import math
from typing import Dict, List, Sequence, Tuple

from pydantic import ValidationError
import pytest

from jev import (
    JudgePipeline,
    JudgeRequestDTO,
    PromptBuilder,
    PromptTemplate,
    ZeroDecodeClient,
    clear_custom_prompt_templates,
    register_prompt_template,
    resolve_prompt_template,
)
from jev.prompt_templates import ENGLISH_TEMPLATE, JAPANESE_TEMPLATE
from jev.zero_decode import InferenceBackend


@pytest.fixture(autouse=True)
def _reset_templates():
    clear_custom_prompt_templates()
    yield
    clear_custom_prompt_templates()


class _RecordingBackend(InferenceBackend):
    """プロンプト・target_tokens を記録し、固定対数確率を返す偽バックエンド"""

    def __init__(self, logprobs: Dict[str, float]) -> None:
        self._logprobs = logprobs
        self.calls: List[Tuple[str, List[Dict[str, str]], List[str]]] = []

    @property
    def is_cloud(self) -> bool:
        return False

    def forward(
        self, model: str, messages: List[Dict[str, str]], top_logprobs: int = 10
    ) -> Tuple[Dict[str, float], float]:
        self.calls.append((model, messages, []))
        return self._logprobs, 1.0

    def forward_constrained(
        self,
        model: str,
        messages: List[Dict[str, str]],
        target_tokens: Sequence[str],
        top_logprobs: int = 10,
    ) -> Tuple[Dict[str, float], float]:
        self.calls.append((model, messages, list(target_tokens)))
        return self._logprobs, 1.0


# ==========================================
# 1. 既定テンプレートと従来出力のバイト単位等価性
# ==========================================


@pytest.mark.parametrize("rule", ["", "個人情報の有無を判定する"])
@pytest.mark.parametrize("model", ["phi4-mini:latest", "qwen3:8b"])
def test_default_noul_prompt_byte_identical(rule: str, model: str) -> None:
    req = JudgeRequestDTO(task_type="noul", context_text="テスト本文", rule_definition=rule)
    messages, targets, mapping = PromptBuilder.build_messages(req, model)

    expected_system = (
        "あなたは超高速な二値判定器です。提示された基準に基づいて、対象テキストが条件に合致するか判定してください。\n"
        "説明や理由は一切不要です。必ず 'Yes' または 'No' のどちらか1単語のみを出力してください。"
    )
    rule_part = f"\n\n【判定基準】\n{rule}" if rule else ""
    expected_user = f"【対象テキスト】\nテスト本文{rule_part}\n\n判定結果 (Yes / No):"
    expected_messages = [
        {"role": "system", "content": expected_system},
        {"role": "user", "content": expected_user},
    ]
    if model == "qwen3:8b":
        expected_messages.append({"role": "assistant", "content": "<think>\n\n</think>\n"})

    assert messages == expected_messages
    assert targets == ["Yes", "No", " Yes", " No"]
    assert mapping == {"Yes": "Yes", "No": "No"}


@pytest.mark.parametrize("swap", [False, True])
@pytest.mark.parametrize("rule", ["", "言語を分類せよ"])
def test_default_choice_prompt_byte_identical(swap: bool, rule: str) -> None:
    labels = ["Python", "Go", "Rust"]
    req = JudgeRequestDTO(task_type="choice", context_text="fn main() {}", rule_definition=rule, labels=labels)
    messages, targets, mapping = PromptBuilder.build_messages(req, "phi4-mini:latest", swap=swap)

    display_labels = list(reversed(labels)) if swap else labels
    expected_map = {sym: lbl for sym, lbl in zip(["A", "B", "C"], display_labels)}
    options_text = "\n".join(f"- {sym}: {lbl}" for sym, lbl in expected_map.items())
    expected_system = (
        "あなたは高精度な単一選択分類器です。提示されたテキストがどの選択肢に最も合致するか判定してください。\n"
        "説明や挨拶は一切出力せず、必ず選択肢のアルファベット記号（A, B, C...）の1文字のみを出力してください。"
    )
    rule_part = f"\n\n【分類基準】\n{rule}" if rule else ""
    expected_user = (
        f"【対象テキスト】\nfn main() {{}}{rule_part}\n\n"
        f"【選択肢】\n{options_text}\n\n"
        "判定記号 (A, B, C):"
    )

    assert messages == [
        {"role": "system", "content": expected_system},
        {"role": "user", "content": expected_user},
    ]
    assert targets == ["A", " A", "B", " B", "C", " C"]
    assert mapping == expected_map


@pytest.mark.parametrize("levels", [3, 5, 7])
@pytest.mark.parametrize("rule", ["", "正確性を評価せよ"])
def test_default_score_prompt_byte_identical(levels: int, rule: str) -> None:
    req = JudgeRequestDTO(task_type="score", context_text="回答文", rule_definition=rule, score_levels=levels)
    messages, targets, mapping = PromptBuilder.build_messages(req, "phi4-mini:latest")

    if levels == 5:
        expected_system = (
            "あなたは厳密な段階評価判定器です。提示された基準に基づいて、対象テキストの品質や適合度を1〜5の5段階で評価してください。\n"
            "説明は一切出力せず、必ず '1', '2', '3', '4', '5' のいずれか1文字の数字のみを出力してください。\n"
            "1: 極めて低い / 不適合\n"
            "2: 低い / やや不十分\n"
            "3: 標準 / 合格水準\n"
            "4: 高い / 良好\n"
            "5: 極めて高い / 完璧"
        )
    else:
        sym_list = ", ".join(f"'{i}'" for i in range(1, levels + 1))
        expected_system = (
            f"あなたは厳密な段階評価判定器です。提示された基準に基づいて、対象テキストの品質や適合度を1〜{levels}の{levels}段階で評価してください。\n"
            f"説明は一切出力せず、必ず {sym_list} のいずれか1文字の数字のみを出力してください。\n"
            "1: 最も低い / 不適合\n"
            f"{levels}: 最も高い / 完璧\n"
            "（途中の段階は基準の定義に従い、数値が大きいほど評価が高い）"
        )

    rule_part = f"\n\n【評価基準】\n{rule}" if rule else ""
    expected_user = f"【対象テキスト】\n回答文{rule_part}\n\n評価スコア (1-{levels}):"
    expected_targets: List[str] = []
    for i in range(1, levels + 1):
        expected_targets.extend([str(i), f" {i}"])
    expected_mapping = {str(i): str(i) for i in range(1, levels + 1)}

    assert PromptBuilder.build_score_system_prompt(levels) == expected_system
    assert messages == [
        {"role": "system", "content": expected_system},
        {"role": "user", "content": expected_user},
    ]
    assert targets == expected_targets
    assert mapping == expected_mapping


@pytest.mark.parametrize("target", [None, "Security"])
@pytest.mark.parametrize("rule", ["", "該当ドメインを判定せよ"])
def test_default_multilabel_prompt_byte_identical(target: str | None, rule: str) -> None:
    req = JudgeRequestDTO(
        task_type="multilabel",
        context_text="SQLインジェクション対策",
        rule_definition=rule,
        labels=["Security", "Performance"],
    )
    messages, targets, mapping = PromptBuilder.build_messages(
        req, "phi4-mini:latest", single_label_target=target
    )

    label_to_eval = target if target else "Security"
    expected_system = (
        "あなたは専門分野分類器です。提示されたテキストが指定のカテゴリ・ラベルに関連しているか判定してください。\n"
        "説明は一切出力せず、関連している場合は 'Yes'、関連していない場合は 'No' の1単語のみを出力してください。"
    )
    rule_part = f"\n\n【判定ルール】\n{rule}" if rule else ""
    expected_user = (
        f"【対象テキスト】\nSQLインジェクション対策{rule_part}\n\n"
        f"【評価カテゴリ】\n{label_to_eval}\n\n"
        f"カテゴリ「{label_to_eval}」に該当しますか？ (Yes / No):"
    )

    assert messages == [
        {"role": "system", "content": expected_system},
        {"role": "user", "content": expected_user},
    ]
    assert targets == ["Yes", "No", " Yes", " No"]
    assert mapping == {"Yes": "Yes", "No": "No"}


# ==========================================
# 2. 英語テンプレート ('en')・State-First 構造・Pipeline 実行
# ==========================================


def test_english_template_all_tasks_and_state_first_prefix() -> None:
    context = "SHARED_LONG_STATE_CONTEXT_BLOCK"

    noul_req = JudgeRequestDTO(
        task_type="noul", context_text=context, rule_definition="Must be polite.", prompt_template="en"
    )
    noul_msgs, _, _ = PromptBuilder.build_messages(noul_req, "phi4-mini:latest")
    assert noul_msgs[1]["content"].startswith(f"{ENGLISH_TEMPLATE.context_header}\n{context}")
    assert "[Decision Criteria]\nMust be polite." in noul_msgs[1]["content"]
    assert noul_msgs[1]["content"].index(context) < noul_msgs[1]["content"].index("Must be polite.")
    assert "Verdict (Yes / No):" in noul_msgs[1]["content"]

    choice_req = JudgeRequestDTO(
        task_type="choice",
        context_text=context,
        rule_definition="Pick the language.",
        labels=["Python", "Rust"],
        prompt_template="en",
    )
    fwd_msgs, _, fwd_map = PromptBuilder.build_messages(choice_req, "phi4-mini:latest", swap=False)
    swp_msgs, _, swp_map = PromptBuilder.build_messages(choice_req, "phi4-mini:latest", swap=True)
    assert fwd_msgs[0]["content"] == swp_msgs[0]["content"]
    assert fwd_msgs[1]["content"].startswith(f"{ENGLISH_TEMPLATE.context_header}\n{context}")
    expected_shared_prefix = (
        f"{ENGLISH_TEMPLATE.context_header}\n{context}\n\n"
        "[Classification Criteria]\nPick the language.\n\n"
        "[Options]\n"
    )
    assert fwd_msgs[1]["content"].startswith(expected_shared_prefix)
    assert swp_msgs[1]["content"].startswith(expected_shared_prefix)
    assert fwd_map == {"A": "Python", "B": "Rust"}
    assert swp_map == {"A": "Rust", "B": "Python"}

    score_req = JudgeRequestDTO(
        task_type="score", context_text=context, rule_definition="Rate clarity.", score_levels=4, prompt_template="en"
    )
    score_msgs, score_targets, _ = PromptBuilder.build_messages(score_req, "phi4-mini:latest")
    assert score_msgs[1]["content"].startswith(f"{ENGLISH_TEMPLATE.context_header}\n{context}")
    assert "4-point scale" in score_msgs[0]["content"]
    assert "Evaluation Score (1-4):" in score_msgs[1]["content"]
    assert score_targets == ["1", " 1", "2", " 2", "3", " 3", "4", " 4"]

    ml_req = JudgeRequestDTO(
        task_type="multilabel",
        context_text=context,
        rule_definition="Classify domain.",
        labels=["Security", "Performance"],
        prompt_template="en",
    )
    ml_msgs_1, _, _ = PromptBuilder.build_messages(ml_req, "phi4-mini:latest", single_label_target="Security")
    ml_msgs_2, _, _ = PromptBuilder.build_messages(ml_req, "phi4-mini:latest", single_label_target="Performance")
    ml_prefix = (
        f"{ENGLISH_TEMPLATE.context_header}\n{context}\n\n"
        "[Classification Rules]\nClassify domain.\n\n"
        "[Target Category]\n"
    )
    assert ml_msgs_1[0] == ml_msgs_2[0]
    assert ml_msgs_1[1]["content"].startswith(ml_prefix)
    assert ml_msgs_2[1]["content"].startswith(ml_prefix)
    assert "Does the text belong to category 'Security'? (Yes / No):" in ml_msgs_1[1]["content"]
    assert "Does the text belong to category 'Performance'? (Yes / No):" in ml_msgs_2[1]["content"]


def test_pipeline_executes_with_english_and_request_override_templates() -> None:
    backend = _RecordingBackend(
        {
            "Yes": math.log(0.9),
            "No": math.log(0.1),
            "A": math.log(0.85),
            "B": math.log(0.15),
            "1": math.log(0.8),
            "2": math.log(0.2),
            "3": -5.0,
            "4": -5.0,
            "5": -5.0,
            "4_unused": -10.0,
        }
    )
    pipe = JudgePipeline(
        default_model="phi4-mini:latest",
        client=ZeroDecodeClient(backend=backend),
        default_prompt_template="en",
    )

    noul_res = pipe.judge_noul("Hello world")
    assert noul_res.status == "SUCCESS"
    assert noul_res.verdict == "Yes"
    assert backend.calls[-1][1][1]["content"].startswith("[Target Text]\nHello world")

    choice_res = pipe.judge_choice("Pick A", labels=["First", "Second"])
    assert choice_res.status == "SUCCESS"
    assert choice_res.verdict == "First"
    assert "[Options]" in backend.calls[-1][1][1]["content"]

    score_res = pipe.judge_score("Good text")
    assert score_res.status == "SUCCESS"
    assert "Evaluation Score (1-5):" in backend.calls[-1][1][1]["content"]

    ml_res = pipe.judge_multilabel("Security patch", labels=["Security", "UI"])
    assert ml_res.status == "SUCCESS"
    assert ml_res.verdict == ["Security", "UI"]
    assert "[Target Category]" in backend.calls[-1][1][1]["content"]

    # リクエスト単位で default_prompt_template ('en') を 'ja' で上書きできる
    override_req = JudgeRequestDTO(task_type="noul", context_text="日本語テスト", prompt_template="ja")
    pipe.judge(override_req)
    assert backend.calls[-1][1][1]["content"].startswith("【対象テキスト】\n日本語テスト")


# ==========================================
# 3. カスタム choice_symbols テスト
# ==========================================


def test_custom_choice_symbols_in_dto_prompt_and_pipeline() -> None:
    req = JudgeRequestDTO(
        task_type="choice",
        context_text="Pick option",
        labels=["Alpha", "Beta", "Gamma"],
        choice_symbols=["1", "2", "3"],
        prompt_template="en",
    )
    msgs, targets, mapping = PromptBuilder.build_messages(req, "phi4-mini:latest")
    assert "- 1: Alpha\n- 2: Beta\n- 3: Gamma" in msgs[1]["content"]
    assert "Selected Symbol (1, 2, 3):" in msgs[1]["content"]
    assert targets == ["1", " 1", "2", " 2", "3", " 3"]
    assert mapping == {"1": "Alpha", "2": "Beta", "3": "Gamma"}

    # パイプライン実行（スワップあり）でもカスタム記号が機能する
    backend = _RecordingBackend({"2": math.log(0.9), "1": math.log(0.05), "3": math.log(0.05)})
    pipe = JudgePipeline(default_model="phi4-mini:latest", client=ZeroDecodeClient(backend=backend))
    res = pipe.judge(req)
    assert res.status == "SUCCESS"
    assert res.verdict == "Beta"
    assert res.details["symbol"] == "2"


def test_custom_choice_symbols_allows_more_than_eight_labels() -> None:
    symbols_10 = [str(i) for i in range(10)]
    labels_9 = [f"Label-{i}" for i in range(9)]
    req = JudgeRequestDTO(
        task_type="choice",
        context_text="Many choices",
        labels=labels_9,
        choice_symbols=symbols_10,
    )
    _, targets, mapping = PromptBuilder.build_messages(req, "phi4-mini:latest")
    assert len(mapping) == 9
    assert mapping["8"] == "Label-8"
    assert targets == [tok for i in range(9) for tok in (str(i), f" {i}")]


@pytest.mark.parametrize(
    "bad_symbols",
    [
        [],
        ["A"],
        ["A", "A", "B"],
        ["A", ""],
        ["A", "   "],
    ],
)
def test_invalid_choice_symbols_rejected(bad_symbols: List[str]) -> None:
    with pytest.raises(ValidationError):
        JudgeRequestDTO(
            task_type="choice",
            context_text="x",
            labels=["L1", "L2"],
            choice_symbols=bad_symbols,
        )

    req = JudgeRequestDTO(task_type="choice", context_text="x", labels=["L1", "L2"])
    with pytest.raises(ValueError):
        PromptBuilder.build_messages(req, "phi4-mini:latest", choice_symbols=bad_symbols)


def test_choice_labels_exceeding_custom_symbols_rejected() -> None:
    with pytest.raises(ValidationError):
        JudgeRequestDTO(
            task_type="choice",
            context_text="x",
            labels=["L1", "L2", "L3", "L4"],
            choice_symbols=["X", "Y", "Z"],
        )

    req = JudgeRequestDTO(task_type="choice", context_text="x", labels=["L1", "L2", "L3", "L4"])
    with pytest.raises(ValueError):
        PromptBuilder.build_messages(req, "phi4-mini:latest", choice_symbols=["X", "Y", "Z"])


def test_per_call_choice_symbols_overrides_request_and_template() -> None:
    req = JudgeRequestDTO(
        task_type="choice",
        context_text="Override test",
        labels=["One", "Two"],
        choice_symbols=["X", "Y"],
    )
    msgs, targets, mapping = PromptBuilder.build_messages(
        req, "phi4-mini:latest", choice_symbols=["1", "2"]
    )
    assert "- 1: One\n- 2: Two" in msgs[1]["content"]
    assert targets == ["1", " 1", "2", " 2"]
    assert mapping == {"1": "One", "2": "Two"}


# ==========================================
# 4. テンプレート登録・解決・クリア
# ==========================================


def test_register_resolve_and_clear_custom_prompt_templates() -> None:
    assert isinstance(JAPANESE_TEMPLATE, PromptTemplate)
    custom_fr = replace(
        ENGLISH_TEMPLATE,
        name="fr",
        context_header="[Texte cible]",
        noul_answer_prompt="Verdict (Yes / No):",
    )
    register_prompt_template(custom_fr)
    resolved = resolve_prompt_template("fr")
    assert resolved.context_header == "[Texte cible]"

    req = JudgeRequestDTO(task_type="noul", context_text="Bonjour", prompt_template="fr")
    msgs, _, _ = PromptBuilder.build_messages(req, "phi4-mini:latest")
    assert msgs[1]["content"].startswith("[Texte cible]\nBonjour")

    # 組み込み名 ('ja') を上書き登録でき、後勝ちになる
    custom_ja_1 = replace(JAPANESE_TEMPLATE, context_header="【入力1】")
    custom_ja_2 = replace(JAPANESE_TEMPLATE, context_header="【入力2】")
    register_prompt_template(custom_ja_1)
    assert resolve_prompt_template("ja").context_header == "【入力1】"
    register_prompt_template(custom_ja_2)
    assert resolve_prompt_template("ja").context_header == "【入力2】"
    assert resolve_prompt_template(None).context_header == "【入力2】"

    clear_custom_prompt_templates()
    assert resolve_prompt_template("ja") == JAPANESE_TEMPLATE
    assert resolve_prompt_template(None) == JAPANESE_TEMPLATE
    with pytest.raises(ValueError):
        resolve_prompt_template("fr")


def test_register_empty_name_and_unknown_template_errors() -> None:
    with pytest.raises(ValueError):
        register_prompt_template(replace(ENGLISH_TEMPLATE, name="  "))

    with pytest.raises(ValueError):
        resolve_prompt_template("nonexistent-template")
