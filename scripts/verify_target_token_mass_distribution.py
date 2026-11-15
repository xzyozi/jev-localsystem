"""対象トークン確率質量(MIN_TARGET_MASS)の実機妥当性検証スクリプト (Issue #26)

qwen3:8b (Tier 1) と phi4-mini:latest (Tier 2) を対象に、
Noul, Choice, Score, Multi-Label 各タスクにおける対象トークン確率質量の分布を計測し、
下限 0.1 の妥当性、誤検知の有無、長文や思考抑制Prefillの影響を検証する。
"""

import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from jev import JudgePipeline, JudgeRequestDTO, ResultMapper, ZeroDecodeClient
from jev.prompt_builder import PromptBuilder


# ----------------------------------------------------------------------
# テストデータセット定義
# ----------------------------------------------------------------------

NOUL_CASES = [
    {"id": "Noul_Tokyo_Capital", "q": "日本の首都は東京ですか？", "c": "日本の首都に関する地理的事実確認", "truth": "Yes"},
    {"id": "Noul_Sky_Color", "q": "晴れた日の昼間の空は青色ですか？", "c": "自然科学に関する一般知識", "truth": "Yes"},
    {"id": "Noul_Math_Addition", "q": "1 + 1 は 3 ですか？", "c": "算術演算の真偽判定", "truth": "No"},
    {"id": "Noul_Security_Password", "q": "パスワードを平文でSlackに投稿することは推奨されますか？", "c": "セキュリティ基本規程", "truth": "No"},
    {"id": "Noul_Python_Dynamic", "q": "Pythonは動的型付け言語ですか？", "c": "プログラミング言語の仕様", "truth": "Yes"},
]

JP_BASE_DOC = """
第3条（情報システムの運用および監視基準）
当社が管理するすべての基幹システム、クラウドインフラ、およびマイクロサービス群は、24時間365日の死活監視を行うものとする。
監視エージェントは30秒間隔でヘルスチェックPingを発行し、3回連続で無応答となった場合は自動フェイルオーバーを発火させる。
すべてのアクセスログは、セキュリティ監査の目的のため最低7年間改ざん不能なストレージに暗号化保存されなければならない。
データベース接続プールの最大許容数はノードあたり256接続とし、アイドル接続は10分でクローズする方針を採る。
"""
TARGET_RULE = "【最重要規則】：本システムの管理者権限アクセスにおいては、多要素認証（MFA）が必須である。"

CHOICE_CASES = [
    {
        "id": "Choice_Sec_Policy",
        "task": "情報セキュリティ規程として適切な行動を選択してください。",
        "labels": [
            "承認済みVPNと多要素認証を経由した暗号化通信を必須とする",
            "個人の私物クラウドストレージにバックアップを保存して自宅作業を行う",
        ],
    },
    {
        "id": "Choice_Labor_Law",
        "task": "労働基準法の観点から適切な勤怠運用を選択してください。",
        "labels": [
            "勤怠打刻とPCログを突合し、客観的に労働時間を把握・管理する",
            "残業時間の上限は無視し、月末にまとめて自己申告で定時退社として修正打刻させる",
        ],
    },
    {
        "id": "Choice_Manner_Email",
        "task": "ビジネスメールにおけるお詫びと日程再調整の文面として適切なものを選択してください。",
        "labels": [
            "当方の都合により急な日程変更をお願いすることとなり深くお詫び申し上げます。来週以降で候補日をご教示いただけますでしょうか。",
            "大変申し訳ございませんが別件が入ったため明日の打ち合わせをリスケしてください。来週なら調整可能です。",
        ],
    },
    {
        "id": "Choice_4Way_Framework",
        "task": "Pythonで高速な非同期Web APIを構築するのに最も適したフレームワークを選択してください。",
        "labels": [
            "Django (フルスタック同期指向)",
            "FastAPI (ASGIネイティブ・型安全)",
            "Flask (WSGI同期マイクロフレームワーク)",
            "Tornado (レガシー非同期フレームワーク)",
        ],
    },
]

SCORE_CASES = [
    {
        "id": "Score_High_CleanCode",
        "text": "本システムはクリーンアーキテクチャに準拠し、カバレッジ95%、明示的型付けが完備されています。",
    },
    {
        "id": "Score_Mid_Functional",
        "text": "一通りの機能は実装され動作しますが、一部ビジネスロジックがコントローラーに漏洩し、単体テストは50%程度です。",
    },
    {
        "id": "Score_Low_DirtyCode",
        "text": "エラー処理が一切なく、変数名も乱雑でテストゼロです。本番投入すると即座にクラッシュします。",
    },
    {
        "id": "Score_MidHigh_GoodDocs",
        "text": "ドキュメントとAPI仕様書が明瞭に整備され、CIでLintが強制されていますが、結合テストの一部がFlakyです。",
    },
]

MULTILABEL_CASES = [
    {
        "id": "ML_DB_Connection",
        "text": "DB接続プールが枯渇し、データベースサーバーへのTCP接続がタイムアウトしました。",
        "labels": ["Database", "Network", "Security", "Frontend", "Hardware"],
    },
    {
        "id": "ML_SQL_Injection",
        "text": "外部からの不正なSQL文の注入が検知され、認証テーブルへの不正アクセスがブロックされました。",
        "labels": ["Database", "Network", "Security", "Frontend", "Hardware"],
    },
    {
        "id": "ML_CSS_Broken",
        "text": "モバイル端末で画面のCSSが崩れ、ログインボタンがタップできない不具合が発生しています。",
        "labels": ["Database", "Network", "Security", "Frontend", "Hardware"],
    },
]


def calculate_percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    sorted_d = sorted(data)
    idx = (len(sorted_d) - 1) * p
    lower = math.floor(idx)
    upper = math.ceil(idx)
    weight = idx - lower
    return sorted_d[lower] * (1.0 - weight) + sorted_d[upper] * weight


def run_evaluation(model_name: str) -> Dict[str, Any]:
    print(f"\n======================================================================")
    print(f"  Target Token Mass Evaluation: {model_name}")
    print(f"======================================================================")

    client = ZeroDecodeClient()
    pipeline = JudgePipeline(default_model=model_name, client=client)

    results: Dict[str, List[Dict[str, Any]]] = {
        "noul": [],
        "noul_long_context": [],
        "choice": [],
        "score": [],
        "multilabel": [],
    }

    # 1. Noul (通常)
    print("\n--- [1] Noul (真偽判定) ---")
    for case in NOUL_CASES:
        dto = JudgeRequestDTO(
            task_type="noul",
            model=model_name,
            context_text=case["q"],
            rule_definition=case["c"],
        )
        messages, tokens, _ = PromptBuilder.build_messages(dto, model_name)
        logprobs, latency_ms = client.forward(model_name, messages, target_tokens=tokens)
        mass = ResultMapper.get_target_mass(logprobs, ("Yes", "No"))
        resp = ResultMapper.map_noul(logprobs, latency_ms)

        rec = {
            "id": case["id"],
            "target_mass": mass,
            "status": resp.status,
            "verdict": resp.verdict,
            "confidence": resp.confidence,
            "latency_ms": latency_ms,
        }
        results["noul"].append(rec)
        print(f"  [{case['id']}] mass={mass:.4f} | status={resp.status} | verdict={resp.verdict} ({latency_ms:.1f}ms)")

    # 2. Noul (長文コンテキスト: 1000〜12000文字)
    print("\n--- [2] Noul (長文コンテキスト減衰) ---")
    scales = [
        (1000, "1000文字(~330T)"),
        (3000, "3000文字(~1000T)"),
        (6000, "6000文字(~2000T SoftLimit)"),
        (12000, "12000文字(~4000T HardLimit)"),
    ]
    for chars, label in scales:
        base = JP_BASE_DOC
        while len(base) < chars:
            base += JP_BASE_DOC
        half = len(base) // 2
        context = base[:half] + "\n\n" + TARGET_RULE + "\n\n" + base[half:chars]

        dto = JudgeRequestDTO(
            task_type="noul",
            model=model_name,
            context_text=context,
            rule_definition="多要素認証（MFA）が必須であるか判定してください。",
        )
        messages, tokens, _ = PromptBuilder.build_messages(dto, model_name)
        logprobs, latency_ms = client.forward(model_name, messages, target_tokens=tokens)
        mass = ResultMapper.get_target_mass(logprobs, ("Yes", "No"))
        resp = ResultMapper.map_noul(logprobs, latency_ms)

        rec = {
            "id": f"Noul_Context_{chars}c",
            "label": label,
            "chars": len(context),
            "target_mass": mass,
            "status": resp.status,
            "verdict": resp.verdict,
            "confidence": resp.confidence,
            "latency_ms": latency_ms,
        }
        results["noul_long_context"].append(rec)
        print(f"  [{label}] mass={mass:.4f} | status={resp.status} | verdict={resp.verdict} ({latency_ms:.1f}ms)")

    # 3. Choice (単一選択)
    print("\n--- [3] Choice (単一選択) ---")
    for case in CHOICE_CASES:
        dto = JudgeRequestDTO(
            task_type="choice",
            model=model_name,
            context_text=case["task"],
            labels=case["labels"],
            swap_verify=True,
        )
        resp = pipeline.judge(dto)

        # Forward と Swap の logprobs を個別に取得して質量計測
        msg_f, tok_f, sym_f = PromptBuilder.build_messages(dto, model_name, swap=False)
        lp_f, _ = client.forward(model_name, msg_f, target_tokens=tok_f)
        mass_f = ResultMapper.get_target_mass(lp_f, sym_f)

        mass_s = None
        if len(case["labels"]) >= 2:
            msg_s, tok_s, sym_s = PromptBuilder.build_messages(dto, model_name, swap=True)
            lp_s, _ = client.forward(model_name, msg_s, target_tokens=tok_s)
            mass_s = ResultMapper.get_target_mass(lp_s, sym_s)

        rec = {
            "id": case["id"],
            "fwd_mass": mass_f,
            "swap_mass": mass_s,
            "target_mass": min(mass_f, mass_s) if mass_s is not None else mass_f,
            "status": resp.status,
            "verdict": resp.verdict,
            "is_consistent": resp.details.get("is_consistent") if resp.details else None,
        }
        results["choice"].append(rec)
        print(f"  [{case['id']}] fwd_mass={mass_f:.4f} | swap_mass={mass_s or 0.0:.4f} | status={resp.status}")

    # 4. Score (段階評価 1〜5)
    print("\n--- [4] Score (段階評価 1〜5) ---")
    for case in SCORE_CASES:
        dto = JudgeRequestDTO(
            task_type="score",
            model=model_name,
            context_text=case["text"],
            rule_definition="コード設計の品質を1〜5で採点してください。",
            score_levels=5,
        )
        messages, tokens, _ = PromptBuilder.build_messages(dto, model_name)
        logprobs, latency_ms = client.forward(model_name, messages, target_tokens=tokens)
        symbols = [str(i) for i in range(1, 6)]
        mass = ResultMapper.get_target_mass(logprobs, symbols)
        resp = ResultMapper.map_score(logprobs, latency_ms, levels=5)

        rec = {
            "id": case["id"],
            "target_mass": mass,
            "status": resp.status,
            "verdict": resp.verdict,
            "confidence": resp.confidence,
        }
        results["score"].append(rec)
        print(f"  [{case['id']}] mass={mass:.4f} | status={resp.status} | score={resp.verdict}")

    # 5. Multi-Label (マルチラベル独立Sigmoid)
    print("\n--- [5] Multi-Label (独立Sigmoid判定) ---")
    for case in MULTILABEL_CASES:
        dto = JudgeRequestDTO(
            task_type="multilabel",
            model=model_name,
            context_text=case["text"],
            labels=case["labels"],
        )
        resp = pipeline.judge(dto)

        for lbl in case["labels"]:
            msg, tok, _ = PromptBuilder.build_messages(dto, model_name, single_label_target=lbl)
            lp, _ = client.forward(model_name, msg, target_tokens=tok)
            mass = ResultMapper.get_target_mass(lp, ("Yes", "No"))
            rec = {
                "id": f"{case['id']}_{lbl}",
                "target_mass": mass,
                "status": "SUCCESS" if mass >= ResultMapper.MIN_TARGET_MASS else "INCONCLUSIVE",
            }
            results["multilabel"].append(rec)

    all_masses = [r["target_mass"] for r in results["multilabel"]]
    print(f"  Multi-Label ({len(all_masses)}件): min_mass={min(all_masses):.4f} | avg_mass={sum(all_masses)/len(all_masses):.4f}")

    # 統計サマリーの計算
    summary = {}
    for task_name, items in results.items():
        masses = [it.get("target_mass", 0.0) for it in items]
        if masses:
            summary[task_name] = {
                "count": len(masses),
                "min": round(min(masses), 4),
                "p05": round(calculate_percentile(masses, 0.05), 4),
                "median": round(calculate_percentile(masses, 0.50), 4),
                "p95": round(calculate_percentile(masses, 0.95), 4),
                "max": round(max(masses), 4),
                "inconclusive_count": sum(1 for it in items if it.get("status") == "INCONCLUSIVE"),
            }

    return {"model": model_name, "summary": summary, "raw_results": results}


def test_without_prefill_suppression(model_name: str = "qwen3:8b") -> Dict[str, Any]:
    """思考抑制Prefillなしの場合の質量検証"""
    print(f"\n======================================================================")
    print(f"  Qwen3 without Thinking-Suppression Prefill Test")
    print(f"======================================================================")
    
    prompt = "Is Tokyo the capital of Japan?\nAnswer Yes or No:\nAnswer:"
    messages = [{"role": "user", "content": prompt}]  # assistant: <think></think> なし
    
    import urllib.request
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": 1,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": 10,
    }
    req = urllib.request.Request(
        "http://localhost:11434/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    
    choice = body.get("choices", [{}])[0]
    top_items = choice.get("logprobs", {}).get("content", [{}])[0].get("top_logprobs", [])
    
    logprobs_dict = {it["token"]: it["logprob"] for it in top_items}
    mass = ResultMapper.get_target_mass(logprobs_dict, ("Yes", "No"))
    first_tok = choice.get("message", {}).get("content")
    
    print(f"  First token without prefill suppression: '{first_tok}'")
    print(f"  Top tokens: {[it['token'] for it in top_items[:5]]}")
    print(f"  Yes/No Target Mass: {mass:.4f}")
    
    return {
        "first_token": first_tok,
        "top_tokens": [it["token"] for it in top_items],
        "target_mass": mass,
    }


if __name__ == "__main__":
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "logs"))
    os.makedirs(out_dir, exist_ok=True)

    # 1. qwen3:8b (Tier 1) の実測
    qwen_res = run_evaluation("qwen3:8b")

    # 2. phi4-mini:latest (Tier 2) の実測
    phi_res = run_evaluation("phi4-mini:latest")

    # 3. 思考抑制Prefillなしのケース
    no_prefill_res = test_without_prefill_suppression("qwen3:8b")

    full_output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "models": {
            "qwen3:8b": qwen_res,
            "phi4-mini:latest": phi_res,
        },
        "no_prefill_experiment": no_prefill_res,
    }

    out_file = os.path.join(out_dir, "2026-11-03_target_token_mass_raw.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_output, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Raw results saved to: {out_file}")
