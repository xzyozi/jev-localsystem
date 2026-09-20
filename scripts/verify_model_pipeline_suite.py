import json
import urllib.request
import math
import time
import os
import sys
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
from typing import List, Dict, Any, Tuple

OLLAMA_OPENAI_URL = "http://localhost:11434/v1/chat/completions"
DEFAULT_MODEL = os.environ.get("JEV_TEST_MODEL", "qwen3:8b")

def get_vram_info() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,nounits,noheader"],
            encoding="utf-8"
        )
        used, total = out.strip().split(",")
        return f"{used.strip()}MB / {total.strip()}MB"
    except Exception:
        return "Unknown"

def get_prefill_tag(model: str) -> str:
    m = model.lower()
    if "qwen3" in m or "deepseek" in m:
        return "<think>\n\n</think>\n"
    if "gemma4" in m or "gemma-4" in m:
        return "<thought>\n\n</thought>\n"
    return ""

def extract_token_logprob(top_items: List[Dict[str, Any]], target_char: str) -> float:
    # Issue #1 実証済: 'A' と ' A' の確率を対数和合算
    probs = []
    for item in top_items:
        tok = item.get("token", "")
        if tok == target_char or tok == " " + target_char:
            probs.append(math.exp(item.get("logprob", -20.0)))
    if not probs:
        return -20.0
    return math.log(sum(probs))

def query_model(model: str, messages: List[Dict[str, str]], top_logprobs: int = 10) -> Tuple[Dict[str, Any], float]:
    prefilled_messages = list(messages)
    tag = get_prefill_tag(model)
    if tag:
        prefilled_messages.append({"role": "assistant", "content": tag})
    
    payload = {
        "model": model,
        "messages": prefilled_messages,
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
        return {"error": str(e)}, elapsed_ms

# ==============================================================================
# 1. Issue #1 & #3: トークンIDバインド・空白対数和 & 単一Forwardレイテンシ
# ==============================================================================
def run_test_token_and_latency(model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    print(f"\n[TEST 1] トークンIDバインド・空白対数和 & 単一Forwardレイテンシ ({model})")
    # ウォームアップ
    query_model(model, [{"role": "user", "content": "1+1="}])
    
    prompt = "Is Tokyo the capital of Japan?\nA) Yes\nB) No\nAnswer:"
    res, latency_ms = query_model(model, [{"role": "user", "content": prompt}])
    
    items = []
    if "choices" in res and res["choices"]:
        logprobs = res["choices"][0].get("logprobs")
        if logprobs and "content" in logprobs and logprobs["content"]:
            items = logprobs["content"][0].get("top_logprobs", [])
            
    log_a = extract_token_logprob(items, "A")
    log_b = extract_token_logprob(items, "B")
    passed = (log_a > log_b and latency_ms < 200.0)
    print(f"  Logprob(A): {log_a:.3f} | Logprob(B): {log_b:.3f} | レイテンシ: {latency_ms:.1f}ms | 合格: {passed}")
    return {
        "passed": passed,
        "logprob_a": log_a,
        "logprob_b": log_b,
        "latency_ms": latency_ms
    }

# ==============================================================================
# 2. Issue #4: 日本語ビジネス規程 ＆ 位置バイアス (Swap Consistency Rate)
# ==============================================================================
JP_SWAP_CASES = [
    {
        "id": "Case_1_Security_Compliance",
        "task": "情報セキュリティ基本規程に基づき、どちらの行動方針が適切ですか？",
        "c1": "顧客情報の持ち出しは一切禁止し、外部送信時は承認済みVPNと多要素認証を経由した暗号化通信を必須とする。",
        "c2": "作業効率向上のため、各自の私物クラウドストレージにバックアップを保存して自宅作業を行う。",
        "ground_truth": "C1"
    },
    {
        "id": "Case_2_Labor_Law",
        "task": "日本の労働基準法および36協定の観点から、どちらの運用管理が適切ですか？",
        "c1": "法定時間外労働が月45時間を超えないよう勤怠打刻とPCログを突合し、客観的に労働時間を把握・管理する。",
        "c2": "納期前のため残業時間の上限は無視し、月末にまとめて自己申告で定時退社として修正打刻させる。",
        "ground_truth": "C1"
    },
    {
        "id": "Case_3_Subtle_Business_Manner",
        "task": "ビジネスメールにおけるお詫びと日程再調整の文面として、どちらが適切ですか？（拮抗テスト）",
        "c1": "ご多忙の折、当方の都合により急な日程変更をお願いすることとなり、深くお詫び申し上げます。誠に恐縮ながら、来週以降で再度ご都合のよろしい候補日をご教示いただけますでしょうか。",
        "c2": "大変申し訳ございませんが、別件が入ってしまったため明日の打ち合わせをリスケジュールさせていただけますでしょうか。来週であれば調整可能ですのでよろしくお願いいたします。",
        "ground_truth": "TIE"
    },
    {
        "id": "Case_4_Contract_Legal",
        "task": "秘密保持契約書（NDA）の締結条項として、どちらの記述が法的リスクを低減できますか？",
        "c1": "本契約の有効期間中および終了後3年間、受領当事者は開示当事者の書面による事前承諾なしに秘密情報を第三者に漏洩してはならない。",
        "c2": "知り得た秘密情報は、一般的な慣習に従って適切に取り扱うものとする。",
        "ground_truth": "C1"
    },
    {
        "id": "Case_5_Subtle_Code_Doc",
        "task": "API仕様書の注記としてどちらの表現が適切ですか？（拮抗テスト）",
        "c1": "本エンドポイントは毎秒最大10リクエストのレートリミットが適用されます。超過時はHTTP 429が返却されます。",
        "c2": "アクセス集中を防ぐため、リクエスト頻度は秒間10回までに制限されており、超えた場合は429 Too Many Requestsとなります。",
        "ground_truth": "TIE"
    }
]

def run_test_position_swap(model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    print(f"\n[TEST 2] 日本語ビジネス規程 ＆ 位置バイアス (Swap Consistency) ({model})")
    consistent_count = 0
    results = []

    for case in JP_SWAP_CASES:
        p_f = (
            f"あなたは企業の法務・コンプライアンス審査官です。\n"
            f"以下の問いに対し、適切な選択肢を 'A' または 'B' の1文字で答えてください。\n\n"
            f"【審査課題】\n{case['task']}\n\n"
            f"A) {case['c1']}\n"
            f"B) {case['c2']}\n\n"
            f"判定 (AまたはB):"
        )
        res_f, _ = query_model(model, [{"role": "user", "content": p_f}])
        items_f = res_f.get("choices", [{}])[0].get("logprobs", {}).get("content", [{}])[0].get("top_logprobs", [])
        log_a_f = extract_token_logprob(items_f, "A")
        log_b_f = extract_token_logprob(items_f, "B")
        verdict_f = "C1" if log_a_f > log_b_f else "C2"

        p_r = (
            f"あなたは企業の法務・コンプライアンス審査官です。\n"
            f"以下の問いに対し、適切な選択肢を 'A' または 'B' の1文字で答えてください。\n\n"
            f"【審査課題】\n{case['task']}\n\n"
            f"A) {case['c2']}\n"
            f"B) {case['c1']}\n\n"
            f"判定 (AまたはB):"
        )
        res_r, _ = query_model(model, [{"role": "user", "content": p_r}])
        items_r = res_r.get("choices", [{}])[0].get("logprobs", {}).get("content", [{}])[0].get("top_logprobs", [])
        log_a_r = extract_token_logprob(items_r, "A")
        log_b_r = extract_token_logprob(items_r, "B")
        verdict_r = "C2" if log_a_r > log_b_r else "C1"

        is_consistent = (verdict_f == verdict_r)
        if is_consistent:
            consistent_count += 1
        print(f"  [{case['id']}] Fwd={verdict_f} | Rev={verdict_r} | 一貫性={is_consistent}")
        results.append({"id": case["id"], "consistent": is_consistent})

    consistency_rate = (consistent_count / len(JP_SWAP_CASES)) * 100.0
    print(f"  => Swap Consistency Rate: {consistency_rate:.1f}%")
    return {"consistency_rate": consistency_rate, "results": results}

# ==============================================================================
# 3. Issue #5: マルチラベル判定 (Sigmoid & 分離度 Margin Delta >= 6.0pt)
# ==============================================================================
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
    }
]
LABELS = ["Security", "Network", "Database", "Frontend", "Hardware"]

def run_test_multilabel_separation(model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    print(f"\n[TEST 3] マルチラベル判定 (Sigmoid 分離度) ({model})")
    pos_diffs = []
    neg_diffs = []

    for case in MULTILABEL_DATASET:
        for lbl in LABELS:
            prompt = (
                f"以下の障害報告を読み、カテゴリ「{lbl}」に該当するか判定してください。\n"
                f"該当する場合は 'A'、該当しない場合は 'B' と答えてください。\n\n"
                f"【報告】{case['text']}\n判定 (AまたはB):"
            )
            res, _ = query_model(model, [{"role": "user", "content": prompt}])
            items = res.get("choices", [{}])[0].get("logprobs", {}).get("content", [{}])[0].get("top_logprobs", [])
            log_a = extract_token_logprob(items, "A")
            log_b = extract_token_logprob(items, "B")
            diff = log_a - log_b

            if lbl in case["ground_truth"]:
                pos_diffs.append(diff)
            else:
                neg_diffs.append(diff)

    avg_pos = sum(pos_diffs) / len(pos_diffs) if pos_diffs else 0.0
    avg_neg = sum(neg_diffs) / len(neg_diffs) if neg_diffs else 0.0
    margin_delta = avg_pos - avg_neg
    passed = margin_delta >= 6.0
    print(f"  正例平均: {avg_pos:.2f} | 負例平均: {avg_neg:.2f} | 分離度ギャップ: {margin_delta:.2f}pt | 合格: {passed}")
    return {"passed": passed, "margin_delta_pt": margin_delta, "avg_pos": avg_pos, "avg_neg": avg_neg}

# ==============================================================================
# 4. Issue #6: 日本語長文コンテキスト減衰 (Soft Limit 2000T / Hard Limit 4000T)
# ==============================================================================
JP_DUMMY_DOC = """
第3条（情報システムの運用および監視基準）
当社が管理するすべての基幹システム、クラウドインフラ、およびマイクロサービス群は、24時間365日の死活監視を行うものとする。
監視エージェントは30秒間隔でヘルスチェックPingを発行し、3回連続で無応答となった場合は自動フェイルオーバーを発火させる。
すべてのアクセスログは、セキュリティ監査の目的のため最低7年間改ざん不能なストレージに暗号化保存されなければならない。
データベース接続プールの最大許容数はノードあたり256接続とし、アイドル接続は10分でクローズする方針を採る。
"""
TARGET_RULE = "【最重要規則】：本システムの管理者権限アクセスにおいては、多要素認証（MFA）が必須である。"

def run_test_context_decay(model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    print(f"\n[TEST 4] 日本語長文コンテキスト減衰 (Soft/Hard Limit) ({model})")
    scales = [
        (1000, "小中規模 (~330T)"),
        (3000, "推奨ソフト限界 (~1,000T)"),
        (6000, "Soft Limit (~2,000T)"),
        (12000, "Hard Limit (~4,000T)")
    ]
    scale_results = []
    for length_chars, label in scales:
        base = JP_DUMMY_DOC
        while len(base) < length_chars:
            base += JP_DUMMY_DOC
        half = len(base) // 2
        context = base[:half] + "\n\n" + TARGET_RULE + "\n\n" + base[half:length_chars]
        prompt = f"規程:\n{context}\n\n質問: 多要素認証（MFA）は必須ですか？\nA) 必須である\nB) 必須ではない\n判定:"
        res, el = query_model(model, [{"role": "user", "content": prompt}])
        items = res.get("choices", [{}])[0].get("logprobs", {}).get("content", [{}])[0].get("top_logprobs", [])
        log_a = extract_token_logprob(items, "A")
        log_b = extract_token_logprob(items, "B")
        margin = log_a - log_b
        print(f"  [{label}] 文字数: {len(context)} | マージン: {margin:+.2f}pt | レイテンシ: {el:.1f}ms")
        scale_results.append({
            "length_chars": len(context),
            "margin_pt": margin,
            "verdict_correct": margin > 0
        })

    # Soft Limit（6000文字 / ~2000T）までは正解マージン保持
    soft_ok = all(r["verdict_correct"] for r in scale_results[:3])
    # Hard Limit（12000文字 / ~4000T）ではマージン減衰
    hard_decay = scale_results[3]["margin_pt"] <= 2.0
    passed = soft_ok and hard_decay
    print(f"  => Soft Limit保持: {soft_ok} | Hard Limit減衰境界: {hard_decay} | 合格: {passed}")
    return {"passed": passed, "scales": scale_results}

# ==============================================================================
# 5. Issue #7: Scoreタスク期待値キャリブレーション (5段階評価の単調減少性)
# ==============================================================================
SCORE_CASES = [
    ("Tier_5_High", "本システムはクリーンアーキテクチャに準拠し、カバレッジ95%、明示的型付けが完備されています。"),
    ("Tier_3_Mid", "一通りの機能は実装され動作しますが、一部ビジネスロジックが漏洩し、単体テストは50%程度です。"),
    ("Tier_1_Low", "エラー処理が一切なく、変数名も乱雑でテストゼロです。本番投入すると即座にクラッシュします。")
]

def run_test_score_calibration(model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    print(f"\n[TEST 5] Scoreタスク期待値キャリブレーション (単調減少性) ({model})")
    expected_scores = []
    for name, text in SCORE_CASES:
        prompt = (
            f"あなたはソフトウェア設計のシニアレビュアーです。\n"
            f"以下のコード設計を1〜5の整数値で採点してください。\n"
            f"1: 最悪, 2: 劣る, 3: 普通, 4: 良い, 5: 完璧\n\n"
            f"【対象内容】\n{text}\n\n"
            f"点数 (1, 2, 3, 4, 5 のいずれか1文字):"
        )
        res, _ = query_model(model, [{"role": "user", "content": prompt}])
        items = res.get("choices", [{}])[0].get("logprobs", {}).get("content", [{}])[0].get("top_logprobs", [])
        raw_probs = {}
        for d in ["1", "2", "3", "4", "5"]:
            lp = extract_token_logprob(items, d)
            raw_probs[int(d)] = math.exp(lp) if lp > -15.0 else 1e-6
        total = sum(raw_probs.values())
        score = sum(k * (v / total) for k, v in raw_probs.items())
        expected_scores.append((name, score))
        print(f"  [{name}] 期待値: {score:.2f}点")

    is_monotonic = (expected_scores[0][1] > expected_scores[1][1] > expected_scores[2][1])
    print(f"  => 単調減少性 (High > Mid > Low): {is_monotonic}")
    return {"monotonic": is_monotonic, "scores": expected_scores}

# ==============================================================================
# 6. Issue #8: シーケンシャルキュー・並行ストレステスト (直列セマフォ)
# ==============================================================================
class SequentialVRAMQueue:
    def __init__(self, concurrency: int = 1):
        self.semaphore = threading.Semaphore(concurrency)

    def execute(self, model: str, req_id: int, prompt: str) -> Dict[str, Any]:
        q_start = time.perf_counter()
        with self.semaphore:
            wait_ms = (time.perf_counter() - q_start) * 1000
            res, infer_ms = query_model(model, [{"role": "user", "content": prompt}])
            return {
                "req_id": req_id,
                "wait_ms": wait_ms,
                "infer_ms": infer_ms,
                "success": "error" not in res
            }

def run_test_sequential_queue(model: str = DEFAULT_MODEL, count: int = 5) -> Dict[str, Any]:
    print(f"\n[TEST 6] シーケンシャルキュー並行ストレステスト ({count}件並行投入) ({model})")
    queue = SequentialVRAMQueue(concurrency=1)
    prompt = "Ping. Answer P"
    
    vram_before = get_vram_info()
    start_total = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=count) as executor:
        futures = [executor.submit(queue.execute, model, i+1, prompt) for i in range(count)]
        for fut in as_completed(futures):
            results.append(fut.result())

    total_time_ms = (time.perf_counter() - start_total) * 1000
    vram_after = get_vram_info()
    all_success = all(r["success"] for r in results)
    print(f"  全件成功: {all_success} | 合計時間: {total_time_ms:.1f}ms | VRAM: {vram_before} -> {vram_after}")
    return {"all_success": all_success, "total_time_ms": total_time_ms, "results": results}

def main():
    parser = argparse.ArgumentParser(description="JEV モデル汎用パイプライン検証テストスイート")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="検証対象モデル識別名 (デフォルト: qwen3:8b)")
    parser.add_argument("--out", type=str, default=None, help="結果JSON出力先パス")
    args = parser.parse_args()

    print(f"\n{'#'*70}")
    print(f" JEV モデル汎用パイプライン包括テストスイート")
    print(f" 検証モデル: {args.model} | 管理: uv")
    print(f"{'#'*70}")

    t1 = run_test_token_and_latency(args.model)
    t2 = run_test_position_swap(args.model)
    t3 = run_test_multilabel_separation(args.model)
    t4 = run_test_context_decay(args.model)
    t5 = run_test_score_calibration(args.model)
    t6 = run_test_sequential_queue(args.model, count=5)

    report = {
        "model": args.model,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_1_token_and_latency": t1,
        "test_2_position_swap": t2,
        "test_3_multilabel_separation": t3,
        "test_4_context_decay": t4,
        "test_5_score_calibration": t5,
        "test_6_sequential_queue": t6
    }

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n[REPORT] レポートを {args.out} に保存しました。")

if __name__ == "__main__":
    main()
