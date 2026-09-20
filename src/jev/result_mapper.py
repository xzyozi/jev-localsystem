"""JEV (Judge & Evaluation via Vectorized-logits) Result Mapper モジュール

詳細設計書 (JEV-DD-001) 2.4節および全実機PoC検証成果に基づく。
1. 空白バリアント対数和（P('A') + P(' A')）による確率計算 (Issue #1)
2. Noul: Yes/No Logit 差分、Margin、確信度
3. Choice: Softmax確率正規化、A/Bスワップ位置バイアス相殺・引き分け検知 (Issue #4)
4. Score: 1〜5スケールの確率加重連続値期待値キャリブレーション (Issue #7)
5. Multi-Label: 独立Sigmoid確率算出と閾値フィルタリング (Issue #5)
"""

import math
from typing import Any, Dict, List, Optional, Tuple

from jev.dto import (
    ChoiceDetails,
    JudgeResponseDTO,
    MultiLabelDetails,
    NoulDetails,
    ScoreDetails,
)


class ResultMapper:
    """Zero-Decode 推論結果（Logprobs）を型安全な DTO へ変換・マッピングするクラス (JEV-DD-001 2.4)"""

    # トークンがTop-10に不在の場合の安全な極小対数確率（exp(-20.0) ≒ 2.06e-9）
    FALLBACK_LOGPROB = -20.0

    @classmethod
    def get_token_prob(cls, logprobs: Dict[str, float], symbol: str) -> float:
        """空白バリアント対数和による確率を計算する (Issue #1 仕様)

        P(symbol) = exp(logprob(symbol)) + exp(logprob(' ' + symbol))
        """
        exact_lp = logprobs.get(symbol, cls.FALLBACK_LOGPROB)
        space_lp = logprobs.get(f" {symbol}", cls.FALLBACK_LOGPROB)
        return math.exp(exact_lp) + math.exp(space_lp)

    @classmethod
    def get_token_logprob_sum(cls, logprobs: Dict[str, float], symbol: str) -> float:
        """空白バリアントの合成対数確率（log(P(sym) + P(' ' + sym))）を計算する"""
        p = cls.get_token_prob(logprobs, symbol)
        return math.log(max(p, 1e-12))

    @classmethod
    def map_noul(
        cls,
        logprobs: Dict[str, float],
        latency_ms: float,
    ) -> JudgeResponseDTO:
        """Noul (真偽判定) 結果のマッピング (Issue #1, #3)"""
        p_yes = cls.get_token_prob(logprobs, "Yes")
        p_no = cls.get_token_prob(logprobs, "No")

        sum_p = p_yes + p_no
        if sum_p > 0:
            norm_yes = p_yes / sum_p
            norm_no = p_no / sum_p
        else:
            norm_yes = norm_no = 0.5

        margin = math.log(max(norm_yes, 1e-12)) - math.log(max(norm_no, 1e-12))
        verdict = "Yes" if norm_yes >= norm_no else "No"
        confidence = max(norm_yes, norm_no)

        details = NoulDetails(
            margin=round(margin, 4),
            prob_yes=round(norm_yes, 4),
            prob_no=round(norm_no, 4),
        )

        return JudgeResponseDTO(
            task_type="noul",
            status="SUCCESS",
            verdict=verdict,
            latency_ms=round(latency_ms, 2),
            confidence=round(confidence, 4),
            details=details.model_dump(),
        )

    @classmethod
    def map_choice(
        cls,
        logprobs_forward: Dict[str, float],
        symbol_map_forward: Dict[str, str],
        latency_ms: float,
        logprobs_swap: Optional[Dict[str, float]] = None,
        symbol_map_swap: Optional[Dict[str, str]] = None,
    ) -> JudgeResponseDTO:
        """Choice (単一選択) 結果のマッピングおよびスワップ照合 (Issue #4)"""
        # 1. Forward 側の確率計算
        probs_raw = {sym: cls.get_token_prob(logprobs_forward, sym) for sym in symbol_map_forward}
        total_p = sum(probs_raw.values())
        probs_norm = {sym: (p / total_p if total_p > 0 else 1.0 / len(probs_raw)) for sym, p in probs_raw.items()}

        best_sym = max(probs_norm, key=probs_norm.get)  # type: ignore
        best_label = symbol_map_forward[best_sym]
        confidence = probs_norm[best_sym]

        label_probabilities = {symbol_map_forward[sym]: round(p, 4) for sym, p in probs_norm.items()}

        # 2. スワップ検証の照合
        if logprobs_swap is not None and symbol_map_swap is not None:
            swap_probs_raw = {sym: cls.get_token_prob(logprobs_swap, sym) for sym in symbol_map_swap}
            best_swap_sym = max(swap_probs_raw, key=swap_probs_raw.get)  # type: ignore
            best_swap_label = symbol_map_swap[best_swap_sym]

            is_consistent = (best_label == best_swap_label)
            if not is_consistent:
                # 位置バイアスにより判定が割れた場合: 偽判定を下さず引き分け検知 (Issue #4)
                details = ChoiceDetails(
                    symbol=best_sym,
                    is_consistent=False,
                    swap_verified=True,
                    probabilities=label_probabilities,
                )
                return JudgeResponseDTO(
                    task_type="choice",
                    status="INCONCLUSIVE",
                    verdict=None,
                    latency_ms=round(latency_ms, 2),
                    confidence=0.5,
                    details=details.model_dump(),
                    error_message="Swap verification inconclusive: conflicting verdicts between original and swapped order",
                )
            else:
                details = ChoiceDetails(
                    symbol=best_sym,
                    is_consistent=True,
                    swap_verified=True,
                    probabilities=label_probabilities,
                )
        else:
            details = ChoiceDetails(
                symbol=best_sym,
                is_consistent=True,
                swap_verified=False,
                probabilities=label_probabilities,
            )

        return JudgeResponseDTO(
            task_type="choice",
            status="SUCCESS",
            verdict=best_label,
            latency_ms=round(latency_ms, 2),
            confidence=round(confidence, 4),
            details=details.model_dump(),
        )

    @classmethod
    def map_score(
        cls,
        logprobs: Dict[str, float],
        latency_ms: float,
        temperature: float = 1.0,
    ) -> JudgeResponseDTO:
        """Score (段階評価) 確率加重平均による連続値期待値キャリブレーション (Issue #7)"""
        scales = ["1", "2", "3", "4", "5"]

        # 温度パラメータを適用した確率計算
        log_probs = {}
        for s in scales:
            p = cls.get_token_prob(logprobs, s)
            lp = math.log(max(p, 1e-12))
            log_probs[s] = lp / temperature

        # Softmax 正規化
        max_lp = max(log_probs.values())
        exp_vals = {s: math.exp(lp - max_lp) for s, lp in log_probs.items()}
        sum_exp = sum(exp_vals.values())
        probs = {s: (ev / sum_exp) for s, ev in exp_vals.items()}

        # 連続値期待値算出: E = Σ (s * P(s))
        expected_score = sum(int(s) * probs[s] for s in scales)
        most_likely = max(probs, key=probs.get)  # type: ignore

        distribution = {s: round(probs[s], 4) for s in scales}
        details = ScoreDetails(
            distribution=distribution,
            most_likely=most_likely,
        )

        return JudgeResponseDTO(
            task_type="score",
            status="SUCCESS",
            verdict=round(expected_score, 3),
            latency_ms=round(latency_ms, 2),
            confidence=round(probs[most_likely], 4),
            details=details.model_dump(),
        )

    @classmethod
    def map_multilabel(
        cls,
        label_results: List[Tuple[str, Dict[str, float]]],
        latency_ms: float,
        threshold: float = 0.5,
        offset: float = 0.0,
    ) -> JudgeResponseDTO:
        """Multi-Label (複数選択) 独立Sigmoid確率と閾値判定 (Issue #5)

        Args:
            label_results: List of (label_name, logprobs_dict)
            latency_ms: 所要時間
            threshold: 採用判定のSigmoid確率閾値 (デフォルト 0.5)
            offset: Logit差分オフセット
        """
        margins = {}
        probabilities = {}
        matched_labels = []

        for label, logprobs in label_results:
            lp_yes = cls.get_token_logprob_sum(logprobs, "Yes")
            lp_no = cls.get_token_logprob_sum(logprobs, "No")

            diff = lp_yes - lp_no + offset
            margins[label] = round(diff, 3)

            # Sigmoid: 1 / (1 + exp(-diff))
            # オーバーフロー防止
            if diff > 40:
                prob = 1.0
            elif diff < -40:
                prob = 0.0
            else:
                prob = 1.0 / (1.0 + math.exp(-diff))

            probabilities[label] = round(prob, 4)

            if prob >= threshold:
                matched_labels.append(label)

        details = MultiLabelDetails(
            margins=margins,
            probabilities=probabilities,
            threshold=threshold,
            offset=offset,
        )

        return JudgeResponseDTO(
            task_type="multilabel",
            status="SUCCESS",
            verdict=matched_labels,
            latency_ms=round(latency_ms, 2),
            confidence=round(max(probabilities.values()) if probabilities else 0.0, 4),
            details=details.model_dump(),
        )
