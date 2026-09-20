import json
import urllib.request
import math
import time

OLLAMA_OPENAI_URL = "http://localhost:11434/v1/chat/completions"
MODEL_NAME = "qwen2.5-coder:14b-instruct"

# ダミー技術文書テキスト（埋め草）
DUMMY_CHUNK = """
Section: Network Protocol Specifications
The system utilizes TCP/IP for all network communication between microservices.
HTTP/2 is enforced for all REST API endpoints to minimize connection overhead.
All payload data must be serialized in JSON format using UTF-8 encoding.
Heartbeat ping interval is set to 30 seconds with a timeout threshold of 90 seconds.
Database replication operates in asynchronous mode with a maximum acceptable lag of 500ms.
Cache layer implements an LRU eviction policy with a memory cap of 8GB per node.
Logging is handled via structured JSON emission directly to stdout/stderr streams.
Service discovery relies on a distributed key-value store with consistent hashing.
Backup snapshots are taken every 6 hours and archived to encrypted object storage.
"""

# 探索対象のターゲット情報 (中央に配置)
TARGET_FACT = "CRITICAL SECURITY RULE: The system private master key MUST be encrypted using AES-256-GCM."

def build_context(target_length_chars: int) -> str:
    # ターゲット情報を中央に埋め込む
    base_text = DUMMY_CHUNK
    while len(base_text) < target_length_chars:
        base_text += DUMMY_CHUNK
    
    half = len(base_text) // 2
    # 前半 + ターゲット事実 + 後半
    full_context = base_text[:half] + "\n\n" + TARGET_FACT + "\n\n" + base_text[half:target_length_chars]
    return full_context

def extract_token_logprob(top_items, target_char):
    probs = []
    for item in top_items:
        tok = item["token"]
        if tok == target_char or tok == " " + target_char:
            probs.append(math.exp(item["logprob"]))
    if not probs:
        return -20.0
    return math.log(sum(probs))

def query_margin_at_length(context_text: str):
    user_prompt = (
        f"Documentation:\n{context_text}\n\n"
        f"Question: According to the documentation, is the master key encrypted using AES-256-GCM?\n"
        f"A: Yes\n"
        f"B: No\n"
        f"Respond with only A or B."
    )
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 1,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": 10
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(OLLAMA_OPENAI_URL, data=data, headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=180) as resp:
        res = json.loads(resp.read().decode("utf-8"))
    latency_ms = (time.perf_counter() - start) * 1000

    choice = res["choices"][0]
    out_tok = choice["message"]["content"].strip()
    top_items = choice.get("logprobs", {}).get("content", [{}])[0].get("top_logprobs", [])

    logit_a = extract_token_logprob(top_items, "A")
    logit_b = extract_token_logprob(top_items, "B")
    margin = logit_a - logit_b

    return {
        "out_tok": out_tok,
        "logit_a": logit_a,
        "logit_b": logit_b,
        "margin": margin,
        "latency_ms": latency_ms
    }

def run_experiment():
    print("================================================================")
    print(" Issue #6: コンテキスト長限界 (Lost in the middle) 確信度減衰検証")
    print(f" Target Model: {MODEL_NAME}")
    print("================================================================\n")

    # 段階的な文字数 (約 150 〜 4,000 トークン規模)
    steps = [
        ("Step 1 (短文)", 500),
        ("Step 2 (中文)", 1500),
        ("Step 3 (長文)", 3000),
        ("Step 4 (超長文)", 6000),
        ("Step 5 (限界級)", 12000),
    ]

    results = []

    for label, length in steps:
        print(f"[{label}] コンテキスト長: {length} 文字...")
        ctx = build_context(length)
        res = query_margin_at_length(ctx)
        
        out = res["out_tok"]
        m = res["margin"]
        t = res["latency_ms"]
        is_correct = (out == "A")

        print(f"  出力: '{out}' ({'正解' if is_correct else '誤答'}) | マージン (A-B): {m:+6.2f} | レイテンシ: {t:6.1f}ms")
        results.append({
            "label": label,
            "length_chars": length,
            "out_tok": out,
            "margin": m,
            "latency_ms": t,
            "is_correct": is_correct
        })

    print("\n================================================================")
    print(" コンテキスト長と確信度マージン減衰の集計")
    print("================================================================")
    print(" 長さ (文字) | トークン換算 | 出力 | マージン (A-B) | 処理時間  | 状態")
    print("-------------+--------------+------+----------------+-----------+-------")
    for r in results:
        est_tokens = int(r["length_chars"] / 3.2)
        status = "健全 (Safe)" if r["margin"] >= 3.0 else ("低下 (Caution)" if r["margin"] >= 1.0 else "危険 (Degraded)")
        print(f" {r['length_chars']:11d} | {est_tokens:10d}T | '{r['out_tok']}'  | {r['margin']:+14.2f} | {r['latency_ms']:7.1f}ms | {status}")
    print("================================================================")

if __name__ == "__main__":
    run_experiment()
