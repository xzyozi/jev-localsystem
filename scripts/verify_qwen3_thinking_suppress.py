import json
import urllib.request
import time

OLLAMA_OPENAI_URL = "http://localhost:11434/v1/chat/completions"
OLLAMA_NATIVE_URL = "http://localhost:11434/api/chat"

print("================================================================")
print(" Qwen 3:8b 思考抑制（Reasoning / Think Suppress）とLogit抽出検証")
print("================================================================\n")

# パターン1: 通常（思考タグなし）
# パターン2: 空の think タグを assistant プレフィル (OpenAI互換)
# パターン3: 空の think タグを native chat (Ollama ネイティブ)
# パターン4: /no_think をユーザープロンプトに付加

prompt = "Is Paris the capital of France?\nA) Yes\nB) No\nAnswer:"

# 1. 通常
payload_normal = {
    "model": "qwen3:8b",
    "messages": [
        {"role": "user", "content": prompt}
    ],
    "max_tokens": 1,
    "temperature": 0.0,
    "logprobs": True,
    "top_logprobs": 5
}

# 2. Assistant Prefill: <think>\n\n</think>\n
payload_prefill = {
    "model": "qwen3:8b",
    "messages": [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "<think>\n\n</think>\n"}
    ],
    "max_tokens": 1,
    "temperature": 0.0,
    "logprobs": True,
    "top_logprobs": 5
}

# 3. Native API with think=False or options
payload_native_no_think = {
    "model": "qwen3:8b",
    "messages": [
        {"role": "user", "content": prompt}
    ],
    "think": False,
    "options": {
        "num_predict": 1,
        "temperature": 0.0
    },
    "stream": False
}

def test_request(title, url, payload):
    print(f"--- [{title}] ---")
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        elapsed = (time.perf_counter() - start) * 1000
        print(f"  Latency: {elapsed:.1f}ms")
        if "choices" in body:
            choice = body["choices"][0]
            print(f"  Content: {repr(choice.get('message', {}).get('content'))}")
            print(f"  Reasoning: {repr(choice.get('message', {}).get('reasoning_content'))}")
            logprobs = choice.get("logprobs")
            if logprobs and "content" in logprobs and logprobs["content"]:
                top = logprobs["content"][0].get("top_logprobs", [])
                tokens = [(item["token"], round(item["logprob"], 3)) for item in top[:5]]
                print(f"  Logprobs: {tokens}")
            else:
                print(f"  Logprobs: None or empty (Raw logprobs key: {logprobs is not None})")
        else:
            msg = body.get("message", {})
            print(f"  Message: {msg}")
            print(f"  Done: {body.get('done')}")
    except Exception as e:
        print(f"  Error: {e}")
    print()

test_request("Pattern 1: 通常 (思考抑制なし)", OLLAMA_OPENAI_URL, payload_normal)
test_request("Pattern 2: Assistant Prefill (<think>\\n\\n</think>\\n)", OLLAMA_OPENAI_URL, payload_prefill)
test_request("Pattern 3: Native API with think=False", OLLAMA_NATIVE_URL, payload_native_no_think)
