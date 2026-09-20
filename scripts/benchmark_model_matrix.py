import json
import urllib.request
import math
import time
import sys
import argparse
from typing import List, Dict, Any, Tuple

OLLAMA_OPENAI_URL = "http://localhost:11434/v1/chat/completions"

# Multi-Label Dataset (Issue #5 準拠)
MULTILABEL_DATASET = [
    {
        "id": "Inc_1",
        "text": "DB接続プールが枯渇し、データベースサーバーへのTCP接続がタイムアウトしました。",
        "ground_truth": {"Database", "Network"}
    },
    {
        "id": "Inc_2",
        "text": "外部からの不正なSQL文の注入が検知され、認証テーブルへの不正アクセスがブロックされました。",
        "ground_truth": {"Security", "Database"}
    },
    {
        "id": "Inc_3",
        "text": "モバイル端末で画面のCSSが崩れ、ログインボタンがタップできない不具合が発生しています。",
        "ground_truth": {"Frontend"}
    },
    {
        "id": "Inc_4",
        "text": "ラック内サーバーの電源ユニットが物理故障し、冷却ファンが停止してノードがダウンしました。",
        "ground_truth": {"Hardware"}
    },
    {
        "id": "Inc_5",
        "text": "大量のSYNパケットによるDDoS攻撃を受け、境界ルーターの帯域が100%飽和して通信障害が発生しました。",
        "ground_truth": {"Security", "Network"}
    }
]
LABELS = ["Security", "Network", "Database", "Frontend", "Hardware"]

# Score Dataset (Issue #7 準拠)
SCORE_CASES = [
    ("Tier_5_High", "本システムはクリーンアーキテクチャに準拠し、ドメインロジックが完全に外部から独立しています。カバレッジは95%以上、エラーハンドリングは全域でResult型により明示的に型付けされています。"),
    ("Tier_3_Mid", "一通りの機能は実装されており動作しますが、コントローラー層に一部ビジネスロジックが漏れ出ています。単体テストは主要パスのみで50%程度です。"),
    ("Tier_1_Low", "エラー処理が一切なく、変数名もa, b, cと乱雑でグローバル変数を多用しています。テストはゼロで、本番投入すると即座にクラッシュします。")
]

def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)

def extract_token_logprob(top_items: List[Dict[str, Any]], target_char: str) -> float:
    probs = []
    for item in top_items:
        tok = item.get("token", "")
        if tok == target_char or tok == " " + target_char:
            probs.append(math.exp(item.get("logprob", -20.0)))
    if not probs:
        return -20.0
    return math.log(sum(probs))

def query_chat_logprobs(model: str, messages: List[Dict[str, str]], top_logprobs: int = 10) -> Tuple[Dict[str, Any], float]:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 1,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": top_logprobs
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(OLLAMA_OPENAI_URL, data=data, headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        elapsed_ms = (time.perf_counter() - start) * 1000
        return body, elapsed_ms
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"  [ERROR] query_chat_logprobs failed for {model}: {e}", file=sys.stderr)
        return {"error": str(e)}, elapsed_ms

def get_prefill_tag(model: str) -> str:
    m = model.lower()
    if "qwen3" in m or "deepseek" in m:
        return "<think>\n\n</think>\n"
    if "gemma4" in m or "gemma-4" in m:
        return "<thought>\n\n</thought>\n"
    return ""

def build_messages(model: str, user_content: str, system_content: str = "") -> List[Dict[str, str]]:
    msgs = []
    if system_content:
        msgs.append({"role": "system", "content": system_content})
    msgs.append({"role": "user", "content": user_content})
    tag = get_prefill_tag(model)
    if tag:
        msgs.append({"role": "assistant", "content": tag})
    return msgs

def extract_top_items(resp: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not resp or "choices" not in resp or not resp["choices"]:
        return []
    choice = resp["choices"][0]
    logprobs = choice.get("logprobs")
    if not logprobs or "content" not in logprobs or not logprobs["content"]:
        return []
    return logprobs["content"][0].get("top_logprobs", [])

def run_benchmark(model: str) -> Dict[str, Any]:
    print(f"\n{'='*70}")
    print(f" JEV 横断モデルベンチマーク: {model}")
    tag = get_prefill_tag(model)
    if tag:
        print(f" [Prefill Active] 思考抑制タグ適用: {repr(tag.strip())}")
    print(f"{'='*70}\n")
    
    # 0. ウォームアップ
    print("[0/4] モデルウォームアップ中...")
    warmup_msgs = build_messages(model, "1+1=")
    warmup_res, w_ms = query_chat_logprobs(model, warmup_msgs)
    if "error" in warmup_res:
        print(f"  [CRITICAL] ウォームアップ失敗: {warmup_res['error']}", file=sys.stderr)
        return {"model": model, "status": "FAILED", "error": warmup_res["error"]}
    print(f"  ウォームアップ完了 (初回ロード: {w_ms:.1f}ms)")
    
    # 1. 単一Forwardレイテンシ計測 (5回測定)
    print("\n[1/4] 単一Forwardレイテンシ計測 (5回実測)...")
    latencies = []
    test_msg = build_messages(model, "Ping. Answer P")
    for i in range(5):
        _, lat = query_chat_logprobs(model, test_msg)
        latencies.append(lat)
        print(f"  Run {i+1}: {lat:.1f} ms")
    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    print(f"  => 平均レイテンシ: {avg_latency:.1f} ms (Min: {min_latency:.1f}ms, Max: {max_latency:.1f}ms)")

    # 2. 空白バリアント対数和・トークンIDバインディング確認
    print("\n[2/4] トークンIDバインド & 空白シフト検証...")
    nl_msg = build_messages(
        model,
        "Is Paris the capital of France?\nA) Yes\nB) No\nAnswer:",
        system_content="You are a classifier. Answer with only 'A' or 'B'."
    )
    resp_token, _ = query_chat_logprobs(model, nl_msg)
    top_items = extract_top_items(resp_token)
    
    a_logprob = extract_token_logprob(top_items, "A")
    b_logprob = extract_token_logprob(top_items, "B")
    top_tokens = [item.get("token") for item in top_items[:5]]
    print(f"  Top 5 Tokens: {top_tokens}")
    print(f"  Logprob 'A' (対数和): {a_logprob:.3f} | Logprob 'B' (対数和): {b_logprob:.3f}")
    token_binding_ok = a_logprob > b_logprob and a_logprob > -10.0

    # 3. Multi-Label Sigmoid 分離度検証
    print("\n[3/4] Multi-Label Sigmoid 分離度検証 (5ケース)...")
    positive_logprobs = []
    negative_logprobs = []
    
    for case in MULTILABEL_DATASET:
        for lbl in LABELS:
            prompt = (
                f"あなたはITインフラおよびセキュリティの専門アナリストです。\n"
                f"以下の障害インシデント報告を読み、カテゴリ「{lbl}」に該当するか判定してください。\n"
                f"該当する場合は 'A'、該当しない場合は 'B' とだけ1文字で答えてください。\n\n"
                f"【報告内容】\n{case['text']}\n\n"
                f"判定結果 (AまたはB):"
            )
            c_msgs = build_messages(model, prompt)
            res, _ = query_chat_logprobs(model, c_msgs)
            c_items = extract_top_items(res)
            log_a = extract_token_logprob(c_items, "A")
            log_b = extract_token_logprob(c_items, "B")
            raw_diff = log_a - log_b
            
            is_positive = lbl in case["ground_truth"]
            if is_positive:
                positive_logprobs.append(raw_diff)
            else:
                negative_logprobs.append(raw_diff)

    avg_pos = sum(positive_logprobs) / len(positive_logprobs) if positive_logprobs else 0
    avg_neg = sum(negative_logprobs) / len(negative_logprobs) if negative_logprobs else 0
    margin_delta = avg_pos - avg_neg
    print(f"  正例平均 Logit差: {avg_pos:.3f}")
    print(f"  負例平均 Logit差: {avg_neg:.3f}")
    print(f"  => 分離度ギャップ (Margin Delta): {margin_delta:.3f} pt")

    # 4. Score 期待値キャリブレーション検証
    print("\n[4/4] Score 期待値キャリブレーション検証 (5段階評価)...")
    score_results = []
    for tier_name, text in SCORE_CASES:
        score_prompt = (
            f"あなたはソフトウェア設計のシニアレビュアーです。\n"
            f"以下のコード設計を1〜5の整数値で採点してください。\n"
            f"1: 最悪, 2: 劣る, 3: 普通, 4: 良い, 5: 完璧\n\n"
            f"【対象内容】\n{text}\n\n"
            f"点数 (1, 2, 3, 4, 5 のいずれか1文字):"
        )
        s_msgs = build_messages(model, score_prompt)
        res, _ = query_chat_logprobs(model, s_msgs)
        s_items = extract_top_items(res)
        
        # 1〜5の確率抽出とSoftmax正規化
        raw_probs = {}
        for d in ["1", "2", "3", "4", "5"]:
            lp = extract_token_logprob(s_items, d)
            raw_probs[int(d)] = math.exp(lp) if lp > -15.0 else 1e-6
        total_p = sum(raw_probs.values())
        norm_probs = {k: v / total_p for k, v in raw_probs.items()}
        expected_score = sum(k * v for k, v in norm_probs.items())
        score_results.append((tier_name, expected_score, norm_probs))
        print(f"  [{tier_name}] 期待値: {expected_score:.3f} 点 (分布: {', '.join(f'{k}:{v:.2f}' for k,v in norm_probs.items())})")

    is_monotonic = (score_results[0][1] > score_results[1][1] > score_results[2][1])
    print(f"  => 単調減少性 (High > Mid > Low): {'OK' if is_monotonic else 'FAIL'}")

    report = {
        "model": model,
        "status": "SUCCESS",
        "latency_ms": {
            "avg": round(avg_latency, 2),
            "min": round(min_latency, 2),
            "max": round(max_latency, 2)
        },
        "token_binding": {
            "passed": token_binding_ok,
            "top_tokens": top_tokens,
            "logprob_A": round(a_logprob, 3),
            "logprob_B": round(b_logprob, 3)
        },
        "multilabel_separation": {
            "avg_pos_logit": round(avg_pos, 3),
            "avg_neg_logit": round(avg_neg, 3),
            "margin_delta_pt": round(margin_delta, 3)
        },
        "score_calibration": {
            "monotonic": is_monotonic,
            "tier_5_score": round(score_results[0][1], 3),
            "tier_3_score": round(score_results[1][1], 3),
            "tier_1_score": round(score_results[2][1], 3)
        }
    }
    return report

def main():
    parser = argparse.ArgumentParser(description="JEV 横断モデルベンチマークスクリプト")
    parser.add_argument("--model", type=str, required=True, help="Ollamaモデル名 (例: phi4-mini:latest)")
    parser.add_argument("--out", type=str, default=None, help="結果JSON出力先パス")
    args = parser.parse_args()

    report = run_benchmark(args.model)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n[REPORT] レポートを {args.out} に保存しました。")

if __name__ == "__main__":
    main()
