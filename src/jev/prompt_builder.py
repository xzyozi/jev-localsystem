"""JEV (Judge & Evaluation via Vectorized-logits) Prompt Builder モジュール

詳細設計書 (JEV-DD-001) 2.2節および各実機PoC検証成果に基づく。
- 思考モデルPrompt Prefill注入 (Issue #2, #9)
- 4大タスク (Noul, Choice, Score, Multi-Label) の型安全なプロンプト構築
- A/Bスワップ推論プロンプト生成 (Issue #4)
- 空白バリアント対応のターゲットトークン一覧管理 (Issue #1)
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union

from jev.dto import MIN_CHOICE_LABELS, JudgeRequestDTO
from jev.model_profiles import PREFILL_THINK_TAG, PREFILL_THOUGHT_TAG, is_thinking, resolve_prefill
from jev.prompt_templates import (
    DEFAULT_CHOICE_SYMBOLS,
    JAPANESE_TEMPLATE,
    PromptTemplate,
    resolve_prompt_template,
)


def _assemble_messages(
    system_prompt: str, user_prompt: str, prefill: Optional[str]
) -> List[Dict[str, str]]:
    """system / user メッセージと任意のアシスタント Prefill を組み立てる"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if prefill:
        messages.append({"role": "assistant", "content": prefill})
    return messages


class PromptBuilder:
    """JEV プロンプト構築およびPrefill注入クラス (JEV-DD-001 2.2)"""

    # 後方互換のための定数。モデル別の判定は jev.model_profiles（ModelProfile）に集約している。
    # THINKING_MODELS は組み込みの思考モデルのスナップショットで、判定には使わない。
    THINKING_MODELS = ("qwen3", "deepseek-r1")
    PREFILL_THINK_TAG = PREFILL_THINK_TAG
    PREFILL_THOUGHT_TAG = PREFILL_THOUGHT_TAG

    # Choiceタスクで使用する標準記号（後方互換のため維持。実体は jev.prompt_templates に集約）
    CHOICE_SYMBOLS = list(DEFAULT_CHOICE_SYMBOLS)
    SCORE_SYMBOLS = ["1", "2", "3", "4", "5"]  # 既定（5段階）。段数は JudgeRequestDTO.score_levels で変わる

    # 5段階（検証済みの既定）のルーブリック。文言は実機検証時から変更しない
    _SCORE_RUBRIC_5 = JAPANESE_TEMPLATE.score_system_prompt_5

    @classmethod
    def build_score_system_prompt(
        cls, levels: int, template: Optional[Union[str, PromptTemplate]] = None
    ) -> str:
        """段数に応じた Score 用 system prompt を返す（5段階は検証済みの固定文言）"""
        return resolve_prompt_template(template).build_score_system_prompt(levels)

    @classmethod
    def is_thinking_model(cls, model_name: str) -> bool:
        """モデル名から推論思考（CoT）モデルであるかを判定する（jev.model_profiles に委譲）"""
        return is_thinking(model_name)

    @classmethod
    def get_prefill_string(cls, model_name: str) -> Optional[str]:
        """思考抑制用のPrefill文字列を取得する (Issue #2, #9 実証成果。jev.model_profiles に委譲)"""
        return resolve_prefill(model_name)

    @classmethod
    def build_messages(
        cls,
        request: JudgeRequestDTO,
        model_name: str,
        swap: bool = False,
        single_label_target: Optional[str] = None,
        template: Optional[Union[str, PromptTemplate]] = None,
        choice_symbols: Optional[Sequence[str]] = None,
    ) -> Tuple[List[Dict[str, str]], List[str], Dict[str, str]]:
        """リクエストに応じたプロンプト（チャットメッセージ列）と対象トークンリストを生成する

        Args:
            request: リクエストDTO
            model_name: 推論対象モデル名
            swap: Choiceタスクにおける選択肢順序の反転フラグ (Issue #4)
            single_label_target: Multi-Labelタスクで特定ラベルを個別評価する場合のラベル名 (Issue #5)
            template: 使用するテンプレート名または PromptTemplate（省略時は request.prompt_template -> 'ja'）
            choice_symbols: Choiceタスク用のカスタム選択肢記号（省略時は request.choice_symbols -> テンプレート既定）

        Returns:
            Tuple[messages, target_tokens, symbol_to_label_mapping]
        """
        task = request.task_type
        prefill = cls.get_prefill_string(model_name)
        resolved_template = resolve_prompt_template(
            template if template is not None else getattr(request, "prompt_template", None)
        )

        if task == "noul":
            return cls._build_noul_prompt(request, prefill, template=resolved_template)
        elif task == "choice":
            effective_symbols = (
                choice_symbols if choice_symbols is not None else getattr(request, "choice_symbols", None)
            )
            return cls._build_choice_prompt(
                request, prefill, swap=swap, template=resolved_template, choice_symbols=effective_symbols
            )
        elif task == "score":
            return cls._build_score_prompt(request, prefill, template=resolved_template)
        elif task == "multilabel":
            return cls._build_multilabel_prompt(
                request, prefill, target_label=single_label_target, template=resolved_template
            )
        else:
            raise ValueError(f"Unsupported task_type: {task}")

    @classmethod
    def _build_noul_prompt(
        cls,
        request: JudgeRequestDTO,
        prefill: Optional[str],
        template: Optional[PromptTemplate] = None,
    ) -> Tuple[List[Dict[str, str]], List[str], Dict[str, str]]:
        tpl = template or resolve_prompt_template(None)
        rule_part = f"\n\n{tpl.noul_rule_header}\n{request.rule_definition}" if request.rule_definition else ""
        user_prompt = f"{tpl.context_header}\n{request.context_text}{rule_part}\n\n{tpl.noul_answer_prompt}"

        messages = _assemble_messages(tpl.noul_system_prompt, user_prompt, prefill)
        # 空白バリアントを含む対象トークン (Issue #1 仕様)
        target_tokens = ["Yes", "No", " Yes", " No"]
        mapping = {"Yes": "Yes", "No": "No"}
        return messages, target_tokens, mapping

    @classmethod
    def _build_choice_prompt(
        cls,
        request: JudgeRequestDTO,
        prefill: Optional[str],
        swap: bool = False,
        template: Optional[PromptTemplate] = None,
        choice_symbols: Optional[Sequence[str]] = None,
    ) -> Tuple[List[Dict[str, str]], List[str], Dict[str, str]]:
        tpl = template or resolve_prompt_template(None)
        raw_symbols = choice_symbols if choice_symbols is not None else tpl.choice_symbols
        available_symbols = JudgeRequestDTO.validate_choice_symbols(list(raw_symbols))
        assert available_symbols is not None

        labels = list(request.labels)
        if len(labels) < MIN_CHOICE_LABELS:
            raise ValueError(f"Choice task requires at least {MIN_CHOICE_LABELS} labels in request.labels")
        if len(labels) > len(available_symbols):
            # DTO 検証を通らない経路でも、ラベルを黙って切り捨てない
            raise ValueError(
                f"Choice task supports at most {len(available_symbols)} labels for the active choice_symbols "
                f"(got {len(labels)})"
            )

        # スワップ時は選択肢の順序を反転 (Issue #4 仕様)
        display_labels = list(reversed(labels)) if swap else labels

        symbols = available_symbols[: len(display_labels)]
        symbol_to_label = {sym: lbl for sym, lbl in zip(symbols, display_labels)}

        options_text = "\n".join([f"- {sym}: {lbl}" for sym, lbl in symbol_to_label.items()])

        rule_part = f"\n\n{tpl.choice_rule_header}\n{request.rule_definition}" if request.rule_definition else ""
        user_prompt = (
            f"{tpl.context_header}\n{request.context_text}{rule_part}\n\n"
            f"{tpl.choice_options_header}\n{options_text}\n\n"
            f"{tpl.choice_answer_prefix} ({', '.join(symbols)}):"
        )

        messages = _assemble_messages(tpl.choice_system_prompt, user_prompt, prefill)

        # 空白バリアントを含む対象トークン
        target_tokens = []
        for sym in symbols:
            target_tokens.extend([sym, f" {sym}"])

        return messages, target_tokens, symbol_to_label

    @classmethod
    def _build_score_prompt(
        cls,
        request: JudgeRequestDTO,
        prefill: Optional[str],
        template: Optional[PromptTemplate] = None,
    ) -> Tuple[List[Dict[str, str]], List[str], Dict[str, str]]:
        tpl = template or resolve_prompt_template(None)
        levels = request.score_levels
        system_prompt = tpl.build_score_system_prompt(levels)
        rule_part = f"\n\n{tpl.score_rule_header}\n{request.rule_definition}" if request.rule_definition else ""
        user_prompt = (
            f"{tpl.context_header}\n{request.context_text}{rule_part}\n\n{tpl.score_answer_prefix} (1-{levels}):"
        )

        messages = _assemble_messages(system_prompt, user_prompt, prefill)

        score_symbols = [str(i) for i in range(1, levels + 1)]
        target_tokens = []
        for sym in score_symbols:
            target_tokens.extend([sym, f" {sym}"])

        mapping = {sym: sym for sym in score_symbols}
        return messages, target_tokens, mapping

    @classmethod
    def _build_multilabel_prompt(
        cls,
        request: JudgeRequestDTO,
        prefill: Optional[str],
        target_label: Optional[str] = None,
        template: Optional[PromptTemplate] = None,
    ) -> Tuple[List[Dict[str, str]], List[str], Dict[str, str]]:
        """Multi-Labelタスク向け独立Sigmoidプロンプト (Issue #5 実証仕様: 分離度10.76〜36.6pt)"""
        tpl = template or resolve_prompt_template(None)
        label_to_eval = (
            target_label
            if target_label
            else (request.labels[0] if request.labels else tpl.multilabel_default_label)
        )

        rule_part = f"\n\n{tpl.multilabel_rule_header}\n{request.rule_definition}" if request.rule_definition else ""
        question = tpl.multilabel_question_template.format(label=label_to_eval)
        user_prompt = (
            f"{tpl.context_header}\n{request.context_text}{rule_part}\n\n"
            f"{tpl.multilabel_category_header}\n{label_to_eval}\n\n"
            f"{question}"
        )

        messages = _assemble_messages(tpl.multilabel_system_prompt, user_prompt, prefill)

        target_tokens = ["Yes", "No", " Yes", " No"]
        mapping = {"Yes": "Yes", "No": "No"}
        return messages, target_tokens, mapping
