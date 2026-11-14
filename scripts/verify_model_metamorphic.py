"""[Layer 2] モデル能力向けメタモルフィック評価スイート (Issue #65).

極性反転（Polarity Flip）・3択以上 Choice 巡回置換不変性・軽微な表記ゆれ/無関係ノイズ耐性を
実機モデルに対して計測し、一貫性率（Consistency Rate）や信頼度/マージン変動量を出力する。
"""

import json
import os
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from jev import JudgePipeline

POLARITY_CASES: List[Dict[str, str]] = [
    {
        "id": "MFA_Mandatory",
        "context": "本番クラウド環境へのログインには、多要素認証（MFA）を必ず実施しなければならない。",
        "pos_rule": "対象テキストは「本番環境へのログインに多要素認証（MFA）が必須である」と定めていますか？",
        "neg_rule": "対象テキストは「本番環境へのログインで多要素認証（MFA）を省略してよい」と定めていますか？",
    },
    {
        "id": "Shared_Account_Prohibition",
        "context": "システムアカウントを複数名で共有することは禁止されており、個人ごとに固有IDを割り当てる。",
        "pos_rule": "複数名でのシステムアカウント共有は禁止されていますか？",
        "neg_rule": "複数名で同じシステムアカウントを共有して運用することが許可されていますか？",
    },
    {
        "id": "Audit_Log_Retention",
        "context": "すべてのアクセス監査ログは改ざん不可能なストレージへ暗号化して7年間保存する。",
        "pos_rule": "アクセス監査ログは暗号化して7年間保存する必要がありますか？",
        "neg_rule": "アクセス監査ログは保存せず即時に破棄することが定められていますか？",
    },
    {
        "id": "PII_External_Transfer",
        "context": "顧客の個人情報は、暗号化されていない公開インターネット経由で外部送信してはならない。",
        "pos_rule": "暗号化されていない経路での顧客個人情報の外部送信は禁止されていますか？",
        "neg_rule": "顧客の個人情報を暗号化なしで外部へ送信しても問題ありませんか？",
    },
    {
        "id": "Backup_Completed",
        "context": "週次メンテナンス前のデータベースバックアップ取得は完了している。書込制限も解除済みである。",
        "pos_rule": "データベースの事前バックアップ取得は完了していますか？",
        "neg_rule": "データベースの事前バックアップ取得はまだ未実施の状態ですか？",
    },
]

CHOICE_CYCLIC_CASES: List[Dict[str, Any]] = [
    {
        "id": "Auth_Policy_4Choice",
        "context": (
            "当社システムでは個人ごとに固有IDを付与し、多要素認証(MFA)と16文字以上のパスワードを必須とする。"
            "共有アカウント利用は禁止する。"
        ),
        "rule": "規程の内容に合致する認証方針を1つ選んでください。",
        "labels": [
            "多要素認証と個人IDを必須とする",
            "共有アカウントの利用を推奨する",
            "パスワードは4桁の数字のみとする",
            "社外から未認証でアクセスを許可する",
        ],
        "ground_truth": "多要素認証と個人IDを必須とする",
    },
    {
        "id": "Incident_SLA_4Choice",
        "context": (
            "本番DB接続プールの枯渇やTCPタイムアウトを検知した場合、"
            "30秒以内に自動フェイルオーバーを実行し、オンコール担当へ通知する。"
        ),
        "rule": "障害検知時の自動フェイルオーバー目標時間として正しいものを選んでください。",
        "labels": [
            "30秒以内",
            "1時間以内",
            "翌営業日まで",
            "自動復旧は行わない",
        ],
        "ground_truth": "30秒以内",
    },
    {
        "id": "Log_Retention_3Choice",
        "context": (
            "アクセスログおよび操作履歴は改ざん検知可能なストレージへ最低7年間保管し、"
            "毎月の監査レビュー対象とする。"
        ),
        "rule": "ログ保管期間として規程に定められているものを選んでください。",
        "labels": [
            "最低7年間保管",
            "24時間後に削除",
            "保管義務なし",
        ],
        "ground_truth": "最低7年間保管",
    },
    {
        "id": "Deployment_Window_3Choice",
        "context": "本番リリースは毎週火曜・木曜の10:00〜16:00に実施し、金曜夕方および休前日のリリースは原則禁止する。",
        "rule": "本番リリースが原則として禁止されている時間帯を選んでください。",
        "labels": [
            "金曜夕方および休前日",
            "火曜日の午前中",
            "木曜日の午後",
        ],
        "ground_truth": "金曜夕方および休前日",
    },
]

NOISE_BASE_CASES: List[Dict[str, Any]] = [
    {
        "id": "Noise_Noul_MFA",
        "task_type": "noul",
        "context": (
            "第2条（アクセス制御）本番環境への接続には多要素認証（MFA）を必須とする。"
            "アカウントの共有は禁止し、退職時は当日に権限を無効化する。"
        ),
        "rule": "本番環境への接続には多要素認証（MFA）が必須ですか？",
    },
    {
        "id": "Noise_Noul_Retention",
        "task_type": "noul",
        "context": (
            "第3条（監査ログ）本番データベースへのすべてのクエリは記録され、"
            "改ざん防止ストレージに最低7年間暗号化して保存する。"
        ),
        "rule": "監査ログを保存せず破棄してもよいと記載されていますか？",
    },
    {
        "id": "Noise_Choice_Failover",
        "task_type": "choice",
        "context": (
            "障害対応手順: DB接続プール枯渇またはTCPタイムアウトを検知した場合、"
            "30秒以内に自動フェイルオーバーを実施し、管理チームへ即時アラートを送信する。"
        ),
        "rule": "障害検知時に実施されるフェイルオーバーの目標時間を選んでください。",
        "labels": [
            "30秒以内",
            "30分以内",
            "翌日対応",
        ],
    },
    {
        "id": "Noise_Choice_Auth",
        "task_type": "choice",
        "context": (
            "パスワード規程: パスワードは最低16文字以上とし、"
            "英大文字・英小文字・数字・特殊記号を含めなければならない。"
        ),
        "rule": "規程で定められているパスワードの最小文字数を選んでください。",
        "labels": [
            "16文字以上",
            "8文字以上",
            "制限なし",
        ],
    },
]

_IRRELEVANT_TAIL = (
    "\n\n【補足メモ】文書管理番号 REF-2026-084。"
    "本館3階会議室の空調点検スケジュールおよび社員食堂の週替わりメニュー案内とは無関係な参考付記です。"
)


def build_perturbations(text: str) -> Dict[str, str]:
    """テキストの意味を変えない軽微な表記ゆれ・無関係ノイズ変種を生成する。"""
    whitespace_width = text.replace("（", " ( ").replace("）", " ) ").replace("、", "、　")
    newline_format = text.replace("。", "。\n\n").strip()
    irrelevant_tail_note = text + _IRRELEVANT_TAIL
    return {
        "whitespace_width": whitespace_width,
        "newline_format": newline_format,
        "irrelevant_tail_note": irrelevant_tail_note,
    }


def _extract_margin(details: Dict[str, Any] | None) -> float | None:
    if not details:
        return None
    val = details.get("margin")
    return float(val) if isinstance(val, (int, float)) else None


def run_polarity_flip_eval(pipeline: JudgePipeline, model: str) -> Dict[str, Any]:
    """Noul タスクにおける肯定・否定ペアの極性反転一貫性を計測する。"""
    print(f"\n[Layer 2 - 1] 極性反転 (Polarity Flip) ({model})")
    rows: List[Dict[str, Any]] = []
    flipped_count = 0
    both_correct_count = 0
    margin_sign_flipped_count = 0

    for case in POLARITY_CASES:
        pos_res = pipeline.judge_noul(case["context"], rule_definition=case["pos_rule"], model=model)
        neg_res = pipeline.judge_noul(case["context"], rule_definition=case["neg_rule"], model=model)

        pos_v = pos_res.verdict
        neg_v = neg_res.verdict
        pos_m = _extract_margin(pos_res.details)
        neg_m = _extract_margin(neg_res.details)

        flipped = {pos_v, neg_v} == {"Yes", "No"}
        both_correct = pos_v == "Yes" and neg_v == "No"
        margin_sign_flipped = pos_m is not None and neg_m is not None and (pos_m * neg_m < 0)

        if flipped:
            flipped_count += 1
        if both_correct:
            both_correct_count += 1
        if margin_sign_flipped:
            margin_sign_flipped_count += 1

        print(
            f"  [{case['id']}] Pos={pos_v}({pos_m}) | Neg={neg_v}({neg_m}) | "
            f"Flipped={flipped} | BothCorrect={both_correct}"
        )
        rows.append(
            {
                "id": case["id"],
                "pos_status": pos_res.status,
                "neg_status": neg_res.status,
                "pos_verdict": pos_v,
                "neg_verdict": neg_v,
                "pos_margin": pos_m,
                "neg_margin": neg_m,
                "flipped": flipped,
                "both_correct": both_correct,
                "margin_sign_flipped": margin_sign_flipped,
            }
        )

    total = len(POLARITY_CASES)
    summary = {
        "total_pairs": total,
        "flip_consistency_rate": round(flipped_count / total * 100.0, 2),
        "both_correct_rate": round(both_correct_count / total * 100.0, 2),
        "margin_sign_flip_rate": round(margin_sign_flipped_count / total * 100.0, 2),
        "cases": rows,
    }
    print(
        f"  => Flip Rate: {summary['flip_consistency_rate']}% | "
        f"Both Correct: {summary['both_correct_rate']}% | "
        f"Margin Sign Flip: {summary['margin_sign_flip_rate']}%"
    )
    return summary


def run_choice_cyclic_eval(pipeline: JudgePipeline, model: str) -> Dict[str, Any]:
    """3〜4択 Choice タスクでの巡回置換不変性を計測する。"""
    print(f"\n[Layer 2 - 2] 3〜4択 Choice 巡回置換不変性 ({model})")
    rows: List[Dict[str, Any]] = []
    invariant_cases = 0
    correct_invariant_cases = 0
    total_shifts = 0
    correct_shifts = 0

    for case in CHOICE_CYCLIC_CASES:
        labels: List[str] = list(case["labels"])
        n = len(labels)
        shift_verdicts: List[str | None] = []
        shift_statuses: List[str] = []

        for s in range(n):
            shifted_labels = labels[s:] + labels[:s]
            res = pipeline.judge_choice(
                context_text=case["context"],
                labels=shifted_labels,
                rule_definition=case["rule"],
                swap_verify=False,
                model=model,
            )
            shift_verdicts.append(res.verdict if isinstance(res.verdict, str) else None)
            shift_statuses.append(res.status)
            total_shifts += 1
            if res.status == "SUCCESS" and res.verdict == case["ground_truth"]:
                correct_shifts += 1

        all_success = all(st == "SUCCESS" for st in shift_statuses)
        cyclic_invariant = all_success and len(set(shift_verdicts)) == 1
        correct_invariant = cyclic_invariant and shift_verdicts[0] == case["ground_truth"]

        if cyclic_invariant:
            invariant_cases += 1
        if correct_invariant:
            correct_invariant_cases += 1

        print(
            f"  [{case['id']} ({n}択)] Verdicts={shift_verdicts} | "
            f"Invariant={cyclic_invariant} | GroundTruthMatch={correct_invariant}"
        )
        rows.append(
            {
                "id": case["id"],
                "num_choices": n,
                "ground_truth": case["ground_truth"],
                "shift_statuses": shift_statuses,
                "shift_verdicts": shift_verdicts,
                "cyclic_invariant": cyclic_invariant,
                "correct_invariant": correct_invariant,
            }
        )

    total_cases = len(CHOICE_CYCLIC_CASES)
    summary = {
        "total_cases": total_cases,
        "total_shifts": total_shifts,
        "cyclic_consistency_rate": round(invariant_cases / total_cases * 100.0, 2),
        "correct_cyclic_rate": round(correct_invariant_cases / total_cases * 100.0, 2),
        "shift_accuracy_rate": round(correct_shifts / total_shifts * 100.0, 2),
        "cases": rows,
    }
    print(
        f"  => Cyclic Consistency: {summary['cyclic_consistency_rate']}% | "
        f"Correct Cyclic: {summary['correct_cyclic_rate']}% | "
        f"Shift Accuracy: {summary['shift_accuracy_rate']}%"
    )
    return summary


def run_noise_robustness_eval(pipeline: JudgePipeline, model: str) -> Dict[str, Any]:
    """軽微な表記ゆれ・無関係ノイズに対する判定維持率と信頼度/マージン変動量を計測する。"""
    print(f"\n[Layer 2 - 3] 表記ゆれ・無関係ノイズ耐性 ({model})")
    rows: List[Dict[str, Any]] = []
    total_variants = 0
    preserved_variants = 0
    conf_deltas: List[float] = []
    margin_deltas: List[float] = []

    for case in NOISE_BASE_CASES:
        if case["task_type"] == "noul":
            base_res = pipeline.judge_noul(case["context"], rule_definition=case["rule"], model=model)
        else:
            base_res = pipeline.judge_choice(
                case["context"],
                labels=case["labels"],
                rule_definition=case["rule"],
                swap_verify=False,
                model=model,
            )

        base_conf = base_res.confidence if base_res.confidence is not None else 0.0
        base_margin = _extract_margin(base_res.details)
        variants = build_perturbations(case["context"])
        variant_rows: Dict[str, Any] = {}

        for v_name, v_text in variants.items():
            if case["task_type"] == "noul":
                res = pipeline.judge_noul(v_text, rule_definition=case["rule"], model=model)
            else:
                res = pipeline.judge_choice(
                    v_text,
                    labels=case["labels"],
                    rule_definition=case["rule"],
                    swap_verify=False,
                    model=model,
                )

            preserved = res.status == base_res.status and res.verdict == base_res.verdict
            total_variants += 1
            if preserved:
                preserved_variants += 1

            conf = res.confidence if res.confidence is not None else 0.0
            conf_delta = round(abs(conf - base_conf), 5)
            conf_deltas.append(conf_delta)

            m = _extract_margin(res.details)
            m_delta = round(abs(m - base_margin), 4) if (m is not None and base_margin is not None) else None
            if m_delta is not None:
                margin_deltas.append(m_delta)

            variant_rows[v_name] = {
                "status": res.status,
                "verdict": res.verdict,
                "preserved": preserved,
                "confidence": res.confidence,
                "conf_delta": conf_delta,
                "margin": m,
                "margin_delta": m_delta,
            }

        case_preserved = all(v["preserved"] for v in variant_rows.values())
        print(
            f"  [{case['id']}] Base={base_res.verdict}({base_conf}) | "
            f"AllPreserved={case_preserved} | Variants={ {k: v['verdict'] for k, v in variant_rows.items()} }"
        )
        rows.append(
            {
                "id": case["id"],
                "task_type": case["task_type"],
                "base_status": base_res.status,
                "base_verdict": base_res.verdict,
                "base_confidence": base_res.confidence,
                "base_margin": base_margin,
                "all_preserved": case_preserved,
                "variants": variant_rows,
            }
        )

    summary = {
        "total_variants": total_variants,
        "verdict_preservation_rate": round(preserved_variants / total_variants * 100.0, 2),
        "max_confidence_delta": round(max(conf_deltas) if conf_deltas else 0.0, 5),
        "avg_confidence_delta": round(sum(conf_deltas) / len(conf_deltas) if conf_deltas else 0.0, 5),
        "max_margin_delta": round(max(margin_deltas) if margin_deltas else 0.0, 4),
        "cases": rows,
    }
    print(
        f"  => Verdict Preservation: {summary['verdict_preservation_rate']}% | "
        f"MaxConfDelta: {summary['max_confidence_delta']} | MaxMarginDelta: {summary['max_margin_delta']}"
    )
    return summary


def run_model_metamorphic_suite(model: str) -> Dict[str, Any]:
    """指定モデルに対して Layer 2 メタモルフィック評価スイート全体を実行する。"""
    pipeline = JudgePipeline(default_model=model)
    return {
        "model": model,
        "polarity_flip": run_polarity_flip_eval(pipeline, model),
        "choice_cyclic": run_choice_cyclic_eval(pipeline, model),
        "noise_robustness": run_noise_robustness_eval(pipeline, model),
    }


if __name__ == "__main__":
    target_models = ["qwen3:8b", "phi4-mini:latest"]
    results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "models": {},
    }

    for model_name in target_models:
        print(f"\n{'=' * 70}")
        print(f"  Layer 2 Metamorphic Capability Evaluation: {model_name}")
        print(f"{'=' * 70}")
        results["models"][model_name] = run_model_metamorphic_suite(model_name)

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "logs"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "2026-11-14_model_metamorphic_raw.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Results saved to: {out_path}")
