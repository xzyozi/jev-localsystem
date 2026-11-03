"""複数質問・Multi-Label における同一 State の Prefill 重複排除ベンチマーク (Issue #22)

現行の「Rule/Query-First (State後置)」と「State-First (State先頭固定)」の2つのプロンプト順序で、
同一 State に対する複数質問バッチ推論時のレイテンシを比較・実測する。

- モデル: qwen3:8b (Tier 1), phi4-mini:latest (Tier 2)
- State長: 500トークン (~1,500文字), 2,000トークン (~6,000文字), 4,000トークン (~12,000文字)
- 質問数: 1問, 4問, 8問
"""

import json
import os
import sys
import time
import urllib.request
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from jev import ResultMapper, ZeroDecodeClient
from jev.model_profiles import resolve_prefill

BASE_JP_DOC = """
第1条（総則および本規程の適用範囲）
当社グループに属するすべての役員、正社員、契約社員、パートタイマー、および委託先ベンダーは、
当社が保有・管理するすべての情報資産（顧客データ、ソースコード、財務情報、営業秘密を含む）の
機密性・完全性・可用性を維持する責務を負う。本規程に違反した場合は就業規則に基づく懲戒処分の対象となる。

第2条（認証およびアクセス制御）
1. 社内ネットワークおよびクラウド環境へのアクセスには、多要素認証（MFA）が必須である。
2. パスワードは最低16文字以上とし、英大文字、英小文字、数字、特殊記号をすべて含めなければならない。
3. アカウントの共有は理由の如何を問わず禁止する。また、退職者のアカウントは即座に無効化する。

第3条（情報システムの運用および監視基準）
当社が管理するすべての基幹システム、クラウドインフラ、およびマイクロサービス群は、24時間365日の死活監視を行う。
監視エージェントは30秒間隔でヘルスチェックPingを発行し、3回連続で無応答となった場合は自動フェイルオーバーを発火させる。
すべてのアクセスログはセキュリティ監査の目的のため最低7年間改ざん不能なストレージに暗号化保存されなければならない。
データベース接続プールの最大許容数はノードあたり256接続とし、アイドル接続は10分でクローズする方針を採る。
"""

QUESTIONS = [
    {"q": "本規程において多要素認証（MFA）は必須ですか？", "truth": "Yes"},
    {"q": "パスワードの最低文字数は8文字以上で十分ですか？", "truth": "No"},
    {"q": "退職者のアカウントは月末まで有効に維持されますか？", "truth": "No"},
    {"q": "ヘルスチェックは30秒間隔で発行されますか？", "truth": "Yes"},
    {"q": "アクセスログは最低1年間保存すれば十分ですか？", "truth": "No"},
    {"q": "アカウントの共有は上長の承認があれば許可されますか？", "truth": "No"},
    {"q": "データベースのアイドル接続は10分でクローズされますか？", "truth": "Yes"},
    {"q": "外部からの不正アクセスはブロックされますか？", "truth": "Yes"},
]


def reset_model_cache(model: str):
    """Ollama の KV キャッシュをクリア（keep_alive: 0 でモデルアンロード）"""
    try:
        url = "http://localhost:11434/api/generate"
        payload = {"model": model, "keep_alive": 0}
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
    except Exception:
        pass
    time.sleep(1.0)


def generate_state_text(target_chars: int) -> str:
    """指定文字数以上の State テキストを生成"""
    text = BASE_JP_DOC
    while len(text) < target_chars:
        text += "\n" + BASE_JP_DOC
    return text[:target_chars]


def build_prompt_query_first(state: str, question: str, prefill: str | None) -> List[Dict[str, str]]:
    """現行方式: 判定基準/質問が先頭、State が後方"""
    system_prompt = (
        "あなたは超高速な二値判定器です。提示された基準に基づいて、対象テキストが条件に合致するか判定してください。\n"
        "説明や理由は一切不要です。必ず 'Yes' または 'No' のどちらか1単語のみを出力してください。"
    )
    user_prompt = f"【質問】\n{question}\n\n【対象テキスト】\n{state}\n\n判定結果 (Yes / No):"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if prefill:
        messages.append({"role": "assistant", "content": prefill})
    return messages


def build_prompt_state_first(state: str, question: str, prefill: str | None) -> List[Dict[str, str]]:
    """改善方式: State が先頭、質問が後方 (Prefix Caching 最適化)"""
    system_prompt = (
        "あなたは超高速な二値判定器です。提示された基準に基づいて、対象テキストが条件に合致するか判定してください。\n"
        "説明や理由は一切不要です。必ず 'Yes' または 'No' のどちらか1単語のみを出力してください。"
    )
    user_prompt = f"【対象テキスト】\n{state}\n\n【質問】\n{question}\n\n判定結果 (Yes / No):"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if prefill:
        messages.append({"role": "assistant", "content": prefill})
    return messages


def run_batch_test(
    client: ZeroDecodeClient,
    model: str,
    state: str,
    questions: List[Dict[str, str]],
    build_fn,
) -> Dict[str, Any]:
    prefill = resolve_prefill(model)
    target_tokens = ["Yes", "No", " Yes", " No"]
    latencies = []
    verdicts = []

    for item in questions:
        msgs = build_fn(state, item["q"], prefill)
        logprobs, elapsed = client.forward(model, msgs, target_tokens=target_tokens)
        resp = ResultMapper.map_noul(logprobs, elapsed)
        latencies.append(round(elapsed, 2))
        verdicts.append(resp.verdict)

    first_latency = latencies[0]
    subsequent_latencies = latencies[1:] if len(latencies) > 1 else [latencies[0]]
    avg_subsequent = round(sum(subsequent_latencies) / len(subsequent_latencies), 2)
    total_latency = round(sum(latencies), 2)

    return {
        "latencies": latencies,
        "first_latency_ms": first_latency,
        "avg_subsequent_ms": avg_subsequent,
        "total_latency_ms": total_latency,
        "verdicts": verdicts,
    }


def benchmark_model(model_name: str) -> Dict[str, Any]:
    print(f"\n{'='*70}")
    print(f"  Benchmark: {model_name}")
    print(f"{'='*70}")

    client = ZeroDecodeClient()
    scales = [
        (1500, "~500T (1,500文字)"),
        (6000, "~2,000T (6,000文字)"),
        (12000, "~4,000T (12,000文字)"),
    ]
    q_counts = [1, 4, 8]
    model_results = {}

    for chars, scale_label in scales:
        state = generate_state_text(chars)
        print(f"\n--- State Length: {scale_label} ---")
        model_results[scale_label] = {}

        for n_q in q_counts:
            q_slice = QUESTIONS[:n_q]

            # 1. 現行方式 (Query-First) の測定
            reset_model_cache(model_name)
            res_query_first = run_batch_test(client, model_name, state, q_slice, build_prompt_query_first)

            # 2. 改善方式 (State-First) の測定
            reset_model_cache(model_name)
            res_state_first = run_batch_test(client, model_name, state, q_slice, build_prompt_state_first)

            speedup_total = (
                round(res_query_first["total_latency_ms"] / res_state_first["total_latency_ms"], 2)
                if res_state_first["total_latency_ms"] > 0
                else 1.0
            )
            speedup_subsequent = (
                round(res_query_first["avg_subsequent_ms"] / res_state_first["avg_subsequent_ms"], 2)
                if res_state_first["avg_subsequent_ms"] > 0
                else 1.0
            )

            print(
                f"  [{n_q}問] Query-First: 1st={res_query_first['first_latency_ms']}ms, 2nd以降avg={res_query_first['avg_subsequent_ms']}ms, total={res_query_first['total_latency_ms']}ms\n"
                f"       State-First: 1st={res_state_first['first_latency_ms']}ms, 2nd以降avg={res_state_first['avg_subsequent_ms']}ms, total={res_state_first['total_latency_ms']}ms\n"
                f"       => 2問目以降速度向上: {speedup_subsequent}x | 全体速度比: {speedup_total}x"
            )

            model_results[scale_label][f"{n_q}_questions"] = {
                "query_first": res_query_first,
                "state_first": res_state_first,
                "speedup_total": speedup_total,
                "speedup_subsequent": speedup_subsequent,
            }

    return model_results


if __name__ == "__main__":
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "logs"))
    os.makedirs(out_dir, exist_ok=True)

    print("Starting Isolated Prefill Caching Benchmark...")
    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "models": {
            "qwen3:8b": benchmark_model("qwen3:8b"),
            "phi4-mini:latest": benchmark_model("phi4-mini:latest"),
        },
    }

    out_path = os.path.join(out_dir, "2026-11-03_prefill_caching_benchmark.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Benchmark results saved to: {out_path}")
