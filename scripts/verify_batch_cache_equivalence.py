"""[Layer 1] バッチ連続推論（KVキャッシュあり）vs 単発推論（キャッシュリセット）の等価性実機検証 (Issue #64)

State-First 最適化（PR #55）により KV キャッシュ（Prefix Caching）が効く連続バッチ推論と、
キャッシュを完全リセット（keep_alive: 0）した単発推論で、同一モデルが同一の結果を再現するかを計測する。
"""

import json
import os
import sys
import time
from typing import Any, Dict, List
import urllib.request

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from fastapi.testclient import TestClient

from jev import JudgePipeline
from jev.server.app import app

BASE_DOC = """
第1条（総則および適用範囲）
当社グループに属するすべての役員、正社員、契約社員、および委託先ベンダーは、
当社が管理する情報資産（顧客データ、ソースコード、財務情報、営業秘密）の機密性・完全性・可用性を維持する責務を負う。

第2条（認証およびアクセス制御）
1. 社内ネットワークおよび本番クラウド環境へのアクセスには、多要素認証（MFA）が必須である。
2. パスワードは最低16文字以上とし、英大文字、英小文字、数字、特殊記号を含めなければならない。
3. アカウントの共有は禁止し、退職者のアカウントは退職当日に即時無効化する。

第3条（インシデント対応および監査ログ）
すべてのアクセスログは改ざん不能なストレージへ最低7年間暗号化保存する。
データベース接続プールの最大許容数はノードあたり256接続とし、アイドル接続は10分でクローズする。
障害発生時、DB接続プール枯渇またはTCPタイムアウトを検知した場合は30秒以内に自動フェイルオーバーを行う。
"""


def build_context(target_chars: int) -> str:
    text = BASE_DOC.strip()
    while len(text) < target_chars:
        text += "\n\n" + BASE_DOC.strip()
    return text[:target_chars]


def reset_ollama_cache(model: str) -> None:
    """Ollama のモデルおよび KV キャッシュをアンロードして完全に初期化する。"""
    url = "http://localhost:11434/api/generate"
    payload = {"model": model, "keep_alive": 0}
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        resp.read()
    time.sleep(0.5)


def evaluate_questions(client: TestClient, model: str, state: str) -> Dict[str, Any]:
    return {
        "policy_choice": {
            "type": "choice",
            "instructions": "本規程に基づき、最も適切な運用方針を選んでください。",
            "criteria": {
                "mfa_required": "本番アクセスには多要素認証(MFA)を必須とする",
                "shared_account": "業務効率化のため共有アカウントを許可する",
                "skip_logging": "アクセスログは保存せず破棄する",
            },
        },
        "compliance_score": {
            "type": "score",
            "instructions": "記載されているセキュリティ管理体制の厳格さを評価してください。",
            "criteria": ["極めて不十分", "不十分", "標準", "良好", "極めて厳格"],
        },
        "mfa_mandatory": {
            "type": "noul",
            "instructions": "本規程において多要素認証（MFA）は必須ですか？",
            "criteria": {"true": "必須である", "false": "必須ではない"},
        },
        "topics": {
            "type": "multilabel",
            "instructions": "本規程に含まれるトピックをすべて選択してください。",
            "criteria": ["Authentication", "Database", "Logging", "FrontendUI"],
            "threshold": 0.5,
        },
    }


def run_evaluate_equivalence(model: str, state_label: str, state_text: str) -> Dict[str, Any]:
    client = TestClient(app)
    questions = evaluate_questions(client, model, state_text)

    # 1. Batch 実行（最初にキャッシュをリセットしてから 4 問連続投入）
    reset_ollama_cache(model)
    batch_payload = {
        "state": state_text,
        "questions": questions,
        "model": model,
        "swap_verify": False,
    }
    t0 = time.perf_counter()
    batch_resp = client.post("/api/v1/evaluate", json=batch_payload)
    batch_ms = (time.perf_counter() - t0) * 1000.0
    assert batch_resp.status_code == 200, batch_resp.text
    batch_answers = batch_resp.json()["answers"]

    # 2. Single 実行（各問ごとにキャッシュリセットしてから 1 問ずつ投入）
    single_answers: Dict[str, Any] = {}
    single_latencies: Dict[str, float] = {}
    for q_name, q_def in questions.items():
        reset_ollama_cache(model)
        single_payload = {
            "state": state_text,
            "questions": {q_name: q_def},
            "model": model,
            "swap_verify": False,
        }
        ts = time.perf_counter()
        resp = client.post("/api/v1/evaluate", json=single_payload)
        single_latencies[q_name] = round((time.perf_counter() - ts) * 1000.0, 2)
        assert resp.status_code == 200, resp.text
        single_answers[q_name] = resp.json()["answers"][q_name]

    # 3. 比較分析
    b_ch = batch_answers["policy_choice"]
    s_ch = single_answers["policy_choice"]
    choice_prob_diff = max(
        abs(b_ch["probabilities"][k] - s_ch["probabilities"][k]) for k in b_ch["probabilities"]
    )

    b_sc = batch_answers["compliance_score"]
    s_sc = single_answers["compliance_score"]
    score_diff = abs((b_sc["score"] or 0.0) - (s_sc["score"] or 0.0))
    score_prob_diff = max(
        abs(b_sc["probabilities"][k] - s_sc["probabilities"][k]) for k in b_sc["probabilities"]
    )

    b_no = batch_answers["mfa_mandatory"]
    s_no = single_answers["mfa_mandatory"]
    noul_prob_diff = (
        abs(b_no["noul"] - s_no["noul"])
        if b_no["noul"] is not None and s_no["noul"] is not None
        else None
    )

    b_ml = batch_answers["topics"]
    s_ml = single_answers["topics"]
    ml_prob_diff = max(
        abs(b_ml["probabilities"][k] - s_ml["probabilities"][k]) for k in b_ml["probabilities"]
    )
    ml_margin_diff = max(
        abs(b_ml["margins"][k] - s_ml["margins"][k]) for k in b_ml["margins"]
    )

    comparisons = {
        "choice": {
            "batch_verdict": b_ch["choice"],
            "single_verdict": s_ch["choice"],
            "verdict_match": b_ch["choice"] == s_ch["choice"],
            "max_prob_diff": round(choice_prob_diff, 5),
        },
        "score": {
            "batch_score": b_sc["score"],
            "single_score": s_sc["score"],
            "score_diff": round(score_diff, 4),
            "max_prob_diff": round(score_prob_diff, 5),
        },
        "noul": {
            "batch_verdict": b_no["verdict"],
            "single_verdict": s_no["verdict"],
            "verdict_match": b_no["verdict"] == s_no["verdict"],
            "prob_diff": round(noul_prob_diff, 5) if noul_prob_diff is not None else None,
            "batch_margin": b_no["margin"],
            "single_margin": s_no["margin"],
        },
        "multilabel": {
            "batch_matched": b_ml["matched_labels"],
            "single_matched": s_ml["matched_labels"],
            "matched_match": set(b_ml["matched_labels"] or []) == set(s_ml["matched_labels"] or []),
            "max_prob_diff": round(ml_prob_diff, 5),
            "max_margin_diff": round(ml_margin_diff, 4),
        },
    }

    single_sum = sum(single_latencies.values())
    print(f"\n  [{state_label} ({len(state_text)} chars)] Batch={batch_ms:.1f}ms vs SingleSum={single_sum:.1f}ms")
    for t_name, cmp in comparisons.items():
        print(f"    - {t_name}: {cmp}")

    return {
        "state_label": state_label,
        "chars": len(state_text),
        "batch_latency_ms": round(batch_ms, 2),
        "single_latencies_ms": single_latencies,
        "comparisons": comparisons,
    }


def run_multilabel_pipeline_equivalence(model: str, state_label: str, state_text: str) -> Dict[str, Any]:
    pipeline = JudgePipeline(default_model=model)
    labels = ["Authentication", "Database", "Logging", "FrontendUI", "Hardware"]
    rule = "本規程に記載されている管理領域を判定してください。"

    # 1. Batch（連続実行）
    reset_ollama_cache(model)
    batch_res = pipeline.judge_multilabel(
        context_text=state_text,
        labels=labels,
        rule_definition=rule,
        model=model,
    )

    # 2. Single（1ラベルずつリセットして実行）
    single_probs: Dict[str, float] = {}
    single_margins: Dict[str, float] = {}
    single_matched: List[str] = []
    for lbl in labels:
        reset_ollama_cache(model)
        res = pipeline.judge_multilabel(
            context_text=state_text,
            labels=[lbl],
            rule_definition=rule,
            model=model,
        )
        assert res.status == "SUCCESS" and res.details is not None
        single_probs[lbl] = res.details["probabilities"][lbl]
        single_margins[lbl] = res.details["margins"][lbl]
        if res.verdict:
            single_matched.extend(res.verdict)

    assert batch_res.status == "SUCCESS" and batch_res.details is not None
    b_probs = batch_res.details["probabilities"]
    b_margins = batch_res.details["margins"]

    max_prob_diff = max(abs(b_probs[lbl] - single_probs[lbl]) for lbl in labels)
    max_margin_diff = max(abs(b_margins[lbl] - single_margins[lbl]) for lbl in labels)
    matched_equal = set(batch_res.verdict or []) == set(single_matched)

    out = {
        "batch_matched": batch_res.verdict,
        "single_matched": single_matched,
        "matched_match": matched_equal,
        "max_prob_diff": round(max_prob_diff, 5),
        "max_margin_diff": round(max_margin_diff, 4),
        "batch_probs": b_probs,
        "single_probs": single_probs,
    }
    print(
        f"    - pipeline_multilabel ({state_label}): match={matched_equal}, "
        f"max_prob_diff={out['max_prob_diff']}, max_margin_diff={out['max_margin_diff']}"
    )
    return out


if __name__ == "__main__":
    models = ["qwen3:8b", "phi4-mini:latest"]
    scales = [
        ("short_1000c", build_context(1000)),
        ("medium_3000c", build_context(3000)),
    ]

    all_results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "models": {},
    }

    for m in models:
        print(f"\n{'='*70}")
        print(f"  Batch vs Single Equivalence Verification: {m}")
        print(f"{'='*70}")
        m_res: Dict[str, Any] = {"evaluate": [], "pipeline_multilabel": []}
        for s_label, s_text in scales:
            m_res["evaluate"].append(run_evaluate_equivalence(m, s_label, s_text))
            m_res["pipeline_multilabel"].append(run_multilabel_pipeline_equivalence(m, s_label, s_text))
        all_results["models"][m] = m_res

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "logs"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "2026-11-14_batch_cache_equivalence_raw.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Results saved to: {out_path}")
