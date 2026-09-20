"""JEV (Judge & Evaluation via Vectorized-logits) Prompt Builder モジュール

詳細設計書 (JEV-DD-001) 2.2節および各実機PoC検証成果に基づく。
- 思考モデルPrompt Prefill注入 (Issue #2, #9)
- 4大タスク (Noul, Choice, Score, Multi-Label) の型安全なプロンプト構築
- A/Bスワップ推論プロンプト生成 (Issue #4)
- 空白バリアント対応のターゲットトークン一覧管理 (Issue #1)
"""

from typing import Dict, List, Optional, Tuple

from jev.dto import JudgeRequestDTO


class PromptBuilder:
    """JEV プロンプト構築およびPrefill注入クラス (JEV-DD-001 2.2)"""

    # 思考モデルの判定シグネチャと注入タグ
    THINKING_MODELS = ("qwen3", "deepseek-r1")
    PREFILL_THINK_TAG = "<think>\n\n</think>\n"
    PREFILL_THOUGHT_TAG = "<thought>\n\n</thought>\n"

    # Choiceタスクで使用する標準記号
    CHOICE_SYMBOLS = ["A", "B", "C", "D", "E", "F", "G", "H"]
    SCORE_SYMBOLS = ["1", "2", "3", "4", "5"]

    @classmethod
    def is_thinking_model(cls, model_name: str) -> bool:
        """モデル名から推論思考（CoT）モデルであるかを判定する"""
        name_lower = model_name.lower()
        return any(tm in name_lower for tm in cls.THINKING_MODELS)

    @classmethod
    def get_prefill_string(cls, model_name: str) -> Optional[str]:
        """思考抑制用のPrefill文字列を取得する (Issue #2, #9 実証成果)"""
        name_lower = model_name.lower()
        if "qwen3" in name_lower:
            return cls.PREFILL_THINK_TAG
        if "deepseek-r1" in name_lower or "gemma" in name_lower:
            return cls.PREFILL_THOUGHT_TAG
        return None

    @classmethod
    def build_messages(
        cls,
        request: JudgeRequestDTO,
        model_name: str,
        swap: bool = False,
        single_label_target: Optional[str] = None,
    ) -> Tuple[List[Dict[str, str]], List[str], Dict[str, str]]:
        """リクエストに応じたプロンプト（チャットメッセージ列）と対象トークンリストを生成する

        Args:
            request: リクエストDTO
            model_name: 推論対象モデル名
            swap: Choiceタスクにおける選択肢順序の反転フラグ (Issue #4)
            single_label_target: Multi-Labelタスクで特定ラベルを個別評価する場合のラベル名 (Issue #5)

        Returns:
            Tuple[messages, target_tokens, symbol_to_label_mapping]
        """
        task = request.task_type
        prefill = cls.get_prefill_string(model_name)

        if task == "noul":
            return cls._build_noul_prompt(request, prefill)
        elif task == "choice":
            return cls._build_choice_prompt(request, prefill, swap=swap)
        elif task == "score":
            return cls._build_score_prompt(request, prefill)
        elif task == "multilabel":
            return cls._build_multilabel_prompt(request, prefill, target_label=single_label_target)
        else:
            raise ValueError(f"Unsupported task_type: {task}")

    @classmethod
    def _build_noul_prompt(
        cls, request: JudgeRequestDTO, prefill: Optional[str]
    ) -> Tuple[List[Dict[str, str]], List[str], Dict[str, str]]:
        system_prompt = (
            "あなたは超高速な二値判定器です。提示された基準に基づいて、対象テキストが条件に合致するか判定してください。\n"
            "説明や理由は一切不要です。必ず 'Yes' または 'No' のどちらか1単語のみを出力してください。"
        )
        rule_part = f"\n【判定基準】\n{request.rule_definition}\n" if request.rule_definition else ""
        user_prompt = f"{rule_part}\n【対象テキスト】\n{request.context_text}\n\n判定結果 (Yes / No):"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if prefill:
            messages.append({"role": "assistant", "content": prefill})

        # 空白バリアントを含む対象トークン (Issue #1 仕様)
        target_tokens = ["Yes", "No", " Yes", " No"]
        mapping = {"Yes": "Yes", "No": "No"}
        return messages, target_tokens, mapping

    @classmethod
    def _build_choice_prompt(
        cls, request: JudgeRequestDTO, prefill: Optional[str], swap: bool = False
    ) -> Tuple[List[Dict[str, str]], List[str], Dict[str, str]]:
        labels = list(request.labels)
        if not labels:
            raise ValueError("Choice task requires at least 2 labels in request.labels")

        # スワップ時は選択肢の順序を反転 (Issue #4 仕様)
        display_labels = list(reversed(labels)) if swap else labels

        symbols = cls.CHOICE_SYMBOLS[: len(display_labels)]
        symbol_to_label = {sym: lbl for sym, lbl in zip(symbols, display_labels)}

        options_text = "\n".join([f"- {sym}: {lbl}" for sym, lbl in symbol_to_label.items()])

        system_prompt = (
            "あなたは高精度な単一選択分類器です。提示されたテキストがどの選択肢に最も合致するか判定してください。\n"
            "説明や挨拶は一切出力せず、必ず選択肢のアルファベット記号（A, B, C...）の1文字のみを出力してください。"
        )
        rule_part = f"\n【分類基準】\n{request.rule_definition}\n" if request.rule_definition else ""
        user_prompt = (
            f"{rule_part}\n【選択肢】\n{options_text}\n\n"
            f"【対象テキスト】\n{request.context_text}\n\n"
            f"判定記号 ({', '.join(symbols)}):"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if prefill:
            messages.append({"role": "assistant", "content": prefill})

        # 空白バリアントを含む対象トークン
        target_tokens = []
        for sym in symbols:
            target_tokens.extend([sym, f" {sym}"])

        return messages, target_tokens, symbol_to_label

    @classmethod
    def _build_score_prompt(
        cls, request: JudgeRequestDTO, prefill: Optional[str]
    ) -> Tuple[List[Dict[str, str]], List[str], Dict[str, str]]:
        system_prompt = (
            "あなたは厳密な段階評価判定器です。提示された基準に基づいて、対象テキストの品質や適合度を1〜5の5段階で評価してください。\n"
            "説明は一切出力せず、必ず '1', '2', '3', '4', '5' のいずれか1文字の数字のみを出力してください。\n"
            "1: 極めて低い / 不適合\n"
            "2: 低い / やや不十分\n"
            "3: 標準 / 合格水準\n"
            "4: 高い / 良好\n"
            "5: 極めて高い / 完璧"
        )
        rule_part = f"\n【評価基準】\n{request.rule_definition}\n" if request.rule_definition else ""
        user_prompt = f"{rule_part}\n【対象テキスト】\n{request.context_text}\n\n評価スコア (1-5):"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if prefill:
            messages.append({"role": "assistant", "content": prefill})

        target_tokens = []
        for sym in cls.SCORE_SYMBOLS:
            target_tokens.extend([sym, f" {sym}"])

        mapping = {sym: sym for sym in cls.SCORE_SYMBOLS}
        return messages, target_tokens, mapping

    @classmethod
    def _build_multilabel_prompt(
        cls, request: JudgeRequestDTO, prefill: Optional[str], target_label: Optional[str] = None
    ) -> Tuple[List[Dict[str, str]], List[str], Dict[str, str]]:
        """Multi-Labelタスク向け独立Sigmoidプロンプト (Issue #5 実証仕様: 分離度10.76〜36.6pt)"""
        label_to_eval = target_label if target_label else (request.labels[0] if request.labels else "汎用")

        system_prompt = (
            "あなたは専門分野分類器です。提示されたテキストが指定のカテゴリ・ラベルに関連しているか判定してください。\n"
            "説明は一切出力せず、関連している場合は 'Yes'、関連していない場合は 'No' の1単語のみを出力してください。"
        )
        rule_part = f"\n【判定ルール】\n{request.rule_definition}\n" if request.rule_definition else ""
        user_prompt = (
            f"{rule_part}\n【評価カテゴリ】\n{label_to_eval}\n\n"
            f"【対象テキスト】\n{request.context_text}\n\n"
            f"カテゴリ「{label_to_eval}」に該当しますか？ (Yes / No):"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if prefill:
            messages.append({"role": "assistant", "content": prefill})

        target_tokens = ["Yes", "No", " Yes", " No"]
        mapping = {"Yes": "Yes", "No": "No"}
        return messages, target_tokens, mapping
