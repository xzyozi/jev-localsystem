import json
import urllib.request
import time

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma4-12b-it-Q4_K_M:latest"

test_prompts = [
    ("コロン直後 (空白なし)", "Is the sky blue? Answer with Yes or No.\nAnswer:"),
    ("コロン直後 (空白あり)", "Is the sky blue? Answer with Yes or No.\nAnswer: "),
    ("Gemma Turn形式 (空白なし)", "<start_of_turn>user\nIs the sky blue? Answer with Yes or No.<end_of_turn>\n<start_of_turn>model\n"),
    ("Gemma Turn形式 (Answer:付き)", "<start_of_turn>user\nIs the sky blue? Answer with Yes or No.<end_of_turn>\n<start_of_turn>model\nAnswer: "),
]

def query_ollama(prompt: str, num_predict: int = 1):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": num_predict,
            "temperature": 0.0
        }
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=data, headers={"Content-Type": "application/json"})
    
    start = time.perf_counter()
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode("utf-8"))
    elapsed_ms = (time.perf_counter() - start) * 1000
    return res, elapsed_ms

print(f"=== Testing Ollama Model: {MODEL_NAME} ===\n")

for label, prompt in test_prompts:
    print(f"--- Case: {label} ---")
    print(f"Prompt (repr): {repr(prompt)}")
    
    res1, t1 = query_ollama(prompt, num_predict=1)
    print(f"Full JSON keys: {list(res1.keys())}")
    print(f"Response repr: {repr(res1.get('response'))}")
    context = res1.get("context", [])
    print(f"Context length: {len(context)}, last 5 tokens: {context[-5:]}")
    print()
