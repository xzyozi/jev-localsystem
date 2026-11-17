"""JEV プロンプト言語・記号テンプレートモジュール

プロンプト文言（言語）および Choice 選択肢記号をデータとして外出しし、
既定の日本語テンプレート (`JAPANESE_TEMPLATE`, name="ja")・英語テンプレート (`ENGLISH_TEMPLATE`, name="en")・
利用者登録テンプレートを統一的に解決する。

State-First 構造（対象テキスト `context_text` をルール定義や選択肢より先に配置し、
Prefix Caching を有効に保つ構造）を全テンプレートで維持する。
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple, Union

from jev.dto import DEFAULT_SCORE_LEVELS

DEFAULT_CHOICE_SYMBOLS: Tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G", "H")


@dataclass(frozen=True)
class PromptTemplate:
    """4大タスク (Noul, Choice, Score, Multi-Label) のプロンプト言語・記号テンプレート"""

    name: str
    # State-First 構造を維持するため、全タスクで対象テキスト見出しを先頭に配置する
    context_header: str

    # Noul (真偽判定)
    noul_system_prompt: str
    noul_rule_header: str
    noul_answer_prompt: str

    # Choice (単一選択)
    choice_system_prompt: str
    choice_rule_header: str
    choice_options_header: str
    choice_answer_prefix: str

    # Score (段階評価)
    score_system_prompt_5: str
    score_system_prompt_template: str
    score_rule_header: str
    score_answer_prefix: str

    # Multi-Label (複数分類)
    multilabel_system_prompt: str
    multilabel_rule_header: str
    multilabel_category_header: str
    multilabel_question_template: str
    multilabel_default_label: str = "汎用"

    # Choice の既定選択肢記号
    choice_symbols: Tuple[str, ...] = DEFAULT_CHOICE_SYMBOLS

    def build_score_system_prompt(self, levels: int) -> str:
        """段数に応じた Score 用 system prompt を返す（5段階は固定ルーブリック）"""
        if levels == DEFAULT_SCORE_LEVELS:
            return self.score_system_prompt_5
        symbols = ", ".join(f"'{i}'" for i in range(1, levels + 1))
        return self.score_system_prompt_template.format(levels=levels, symbols=symbols)


# 既定の日本語テンプレート（従来の PromptBuilder ハードコード文言とバイト単位で一致）
JAPANESE_TEMPLATE = PromptTemplate(
    name="ja",
    context_header="【対象テキスト】",
    noul_system_prompt=(
        "あなたは超高速な二値判定器です。提示された基準に基づいて、対象テキストが条件に合致するか判定してください。\n"
        "説明や理由は一切不要です。必ず 'Yes' または 'No' のどちらか1単語のみを出力してください。"
    ),
    noul_rule_header="【判定基準】",
    noul_answer_prompt="判定結果 (Yes / No):",
    choice_system_prompt=(
        "あなたは高精度な単一選択分類器です。提示されたテキストがどの選択肢に最も合致するか判定してください。\n"
        "説明や挨拶は一切出力せず、必ず選択肢のアルファベット記号（A, B, C...）の1文字のみを出力してください。"
    ),
    choice_rule_header="【分類基準】",
    choice_options_header="【選択肢】",
    choice_answer_prefix="判定記号",
    score_system_prompt_5=(
        "あなたは厳密な段階評価判定器です。提示された基準に基づいて、対象テキストの品質や適合度を1〜5の5段階で評価してください。\n"
        "説明は一切出力せず、必ず '1', '2', '3', '4', '5' のいずれか1文字の数字のみを出力してください。\n"
        "1: 極めて低い / 不適合\n"
        "2: 低い / やや不十分\n"
        "3: 標準 / 合格水準\n"
        "4: 高い / 良好\n"
        "5: 極めて高い / 完璧"
    ),
    score_system_prompt_template=(
        "あなたは厳密な段階評価判定器です。提示された基準に基づいて、対象テキストの品質や適合度を1〜{levels}の{levels}段階で評価してください。\n"
        "説明は一切出力せず、必ず {symbols} のいずれか1文字の数字のみを出力してください。\n"
        "1: 最も低い / 不適合\n"
        "{levels}: 最も高い / 完璧\n"
        "（途中の段階は基準の定義に従い、数値が大きいほど評価が高い）"
    ),
    score_rule_header="【評価基準】",
    score_answer_prefix="評価スコア",
    multilabel_system_prompt=(
        "あなたは専門分野分類器です。提示されたテキストが指定のカテゴリ・ラベルに関連しているか判定してください。\n"
        "説明は一切出力せず、関連している場合は 'Yes'、関連していない場合は 'No' の1単語のみを出力してください。"
    ),
    multilabel_rule_header="【判定ルール】",
    multilabel_category_header="【評価カテゴリ】",
    multilabel_question_template="カテゴリ「{label}」に該当しますか？ (Yes / No):",
    multilabel_default_label="汎用",
    choice_symbols=DEFAULT_CHOICE_SYMBOLS,
)

# 英語テンプレート (State-First 構造)
ENGLISH_TEMPLATE = PromptTemplate(
    name="en",
    context_header="[Target Text]",
    noul_system_prompt=(
        "You are an ultra-fast binary classifier. Determine whether the target text satisfies the given criteria.\n"
        "Do not output any explanation or reason. Output only a single word: 'Yes' or 'No'."
    ),
    noul_rule_header="[Decision Criteria]",
    noul_answer_prompt="Verdict (Yes / No):",
    choice_system_prompt=(
        "You are a high-precision single-choice classifier. Determine which option best matches the given text.\n"
        "Do not output any explanation or greeting. Output only the single symbol of the matching option."
    ),
    choice_rule_header="[Classification Criteria]",
    choice_options_header="[Options]",
    choice_answer_prefix="Selected Symbol",
    score_system_prompt_5=(
        "You are a strict graded evaluation judge. Evaluate the quality or relevance of the target text "
        "on a 5-point scale from 1 to 5 based on the given criteria.\n"
        "Do not output any explanation. Output only a single digit from '1', '2', '3', '4', '5'.\n"
        "1: Very poor / Non-compliant\n"
        "2: Poor / Somewhat insufficient\n"
        "3: Standard / Acceptable\n"
        "4: High / Good\n"
        "5: Very high / Excellent"
    ),
    score_system_prompt_template=(
        "You are a strict graded evaluation judge. Evaluate the quality or relevance of the target text "
        "on a {levels}-point scale from 1 to {levels} based on the given criteria.\n"
        "Do not output any explanation. Output only a single digit from {symbols}.\n"
        "1: Lowest / Non-compliant\n"
        "{levels}: Highest / Excellent\n"
        "(Intermediate levels follow the criteria definition; higher numbers indicate higher quality.)"
    ),
    score_rule_header="[Evaluation Criteria]",
    score_answer_prefix="Evaluation Score",
    multilabel_system_prompt=(
        "You are a domain classification judge. Determine whether the given text is relevant to the specified "
        "category/label.\n"
        "Do not output any explanation. Output only 'Yes' if relevant, or 'No' if not relevant."
    ),
    multilabel_rule_header="[Classification Rules]",
    multilabel_category_header="[Target Category]",
    multilabel_question_template="Does the text belong to category '{label}'? (Yes / No):",
    multilabel_default_label="General",
    choice_symbols=DEFAULT_CHOICE_SYMBOLS,
)

DEFAULT_TEMPLATE_NAME = "ja"

_BUILTIN_TEMPLATES: Dict[str, PromptTemplate] = {
    JAPANESE_TEMPLATE.name: JAPANESE_TEMPLATE,
    ENGLISH_TEMPLATE.name: ENGLISH_TEMPLATE,
}

# 利用者が register_prompt_template() で追加したテンプレート（組み込みより優先して評価する）
_custom_templates: List[PromptTemplate] = []


def register_prompt_template(template: PromptTemplate) -> None:
    """テンプレートを追加する。組み込みより優先され、後から追加したものが先に評価される。"""
    if not template.name or not template.name.strip():
        raise ValueError("PromptTemplate.name must be non-empty")
    _custom_templates.insert(0, template)


def clear_custom_prompt_templates() -> None:
    """register_prompt_template() で追加したテンプレートをすべて削除する（主にテスト用）。"""
    _custom_templates.clear()


_default_template_ctx: ContextVar[Optional[Union[str, PromptTemplate]]] = ContextVar(
    "_default_template_ctx", default=None
)


@contextmanager
def scoped_default_prompt_template(
    template: Optional[Union[str, PromptTemplate]],
) -> Iterator[None]:
    """実行スコープ内の既定プロンプトテンプレートを一時指定する（スレッド/非同期安全）。"""
    if template is None:
        yield
        return
    token = _default_template_ctx.set(template)
    try:
        yield
    finally:
        _default_template_ctx.reset(token)


def resolve_prompt_template(template: Optional[Union[str, PromptTemplate]] = None) -> PromptTemplate:
    """テンプレート名または PromptTemplate インスタンスを解決して返す（省略時は 'ja'）。"""
    effective = _default_template_ctx.get() if template is None else template
    if isinstance(effective, PromptTemplate):
        return effective
    target_name = DEFAULT_TEMPLATE_NAME if effective is None else effective
    if not isinstance(target_name, str):
        raise TypeError(f"template must be str, PromptTemplate, or None (got {type(effective).__name__})")
    for custom in _custom_templates:
        if custom.name == target_name:
            return custom
    if target_name in _BUILTIN_TEMPLATES:
        return _BUILTIN_TEMPLATES[target_name]
    available = sorted({*_BUILTIN_TEMPLATES.keys(), *(t.name for t in _custom_templates)})
    raise ValueError(f"Unknown prompt template: {target_name!r} (available: {', '.join(available)})")
