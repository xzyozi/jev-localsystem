import json
import urllib.request
import time

OLLAMA_URL = "http://localhost:11434/api/generate"

def query_ollama(model: str, prompt: str, num_predict: int = 1):
    payload = {
        "model": model,
        "prompt": prompt,
        "raw": True,
        "stream": False,
        "options": {
            "num_predict": num_predict,
            "temperature": 0.0,
        }
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=data, headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            res = json.loads(resp.read().decode("utf-8"))
        elapsed_ms = (time.perf_counter() - start) * 1000
        return res, elapsed_ms
    except Exception as e:
        print(f"  [ERROR] query_ollama failed: {e}")
        return {"response": f"ERROR: {e}", "context": []}, 0

def run_experiment_1():
    models = ["gemma4-12b-it-Q4_K_M:latest", "qwen2.5-coder:14b-instruct"]
    
    print("================================================================")
    print(" Issue #1: トークンID静的バインドと空白正規化・語彙整合性検証")
    print("================================================================\n")
    
    # 1. プロンプト末尾の空白有無による影響テスト (Yes/No タスク)
    trailing_cases = [
        ("末尾改行のみ (\\n)", "<start_of_turn>user\nIs ice cold? Answer with Yes or No.<end_of_turn>\n<start_of_turn>model\n<thought>\n\n</thought>\n"),
        ("末尾改行＋スペース (\\n )", "<start_of_turn>user\nIs ice cold? Answer with Yes or No.<end_of_turn>\n<start_of_turn>model\n<thought>\n\n</thought>\n "),
        ("Answer: (末尾スペースあり)", "<start_of_turn>user\nIs ice cold? Answer with Yes or No.<end_of_turn>\n<start_of_turn>model\n<thought>\n\n</thought>\nAnswer: "),
        ("Answer: (末尾スペースなし)", "<start_of_turn>user\nIs ice cold? Answer with Yes or No.<end_of_turn>\n<start_of_turn>model\n<thought>\n\n</thought>\nAnswer:"),
    ]
    
    results = {}

    for model in models:
        print(f"\n################################################################")
        print(f" Target Model: {model}")
        print(f"################################################################")
        print("  Warming up model load...")
        query_ollama(model, "hi", num_predict=1)
        print("  Model loaded.")
        results[model] = []

        # (1) 末尾空白テスト
        print("\n--- Part 1: プロンプト末尾の空白有無と出力トークンの関係 ---")
        for label, prompt in trailing_cases:
            res, elapsed = query_ollama(model, prompt, num_predict=1)
            resp = res.get("response", "")
            context = res.get("context", [])
            last_token_id = context[-1] if context else None
            
            has_leading_space = resp.startswith(" ")
            print(f"[{label}]")
            print(f"  Response: {repr(resp)} (len={len(resp)}) | Leading Space: {has_leading_space}")
            print(f"  Last Token ID: {last_token_id} | Latency: {elapsed:.1f}ms")
            
            results[model].append({
                "test": "trailing_space",
                "label": label,
                "response": resp,
                "token_id": last_token_id,
                "has_leading_space": has_leading_space
            })

        # (2) 多クラス選択肢 (A, B, C) テスト
        print("\n--- Part 2: 記号選択肢 (A, B, C) の 1トークン出力 ---")
        choice_prompt = "<start_of_turn>user\nWhich is a fruit? A: Dog, B: Apple, C: Car. Answer with only the letter A, B, or C.<end_of_turn>\n<start_of_turn>model\n<thought>\n\n</thought>\n"
        if "qwen" in model.lower():
            choice_prompt = "<|im_start|>user\nWhich is a fruit? A: Dog, B: Apple, C: Car. Answer with only the letter A, B, or C.<|im_end|>\n<|im_start|>assistant\n"
            
        res, elapsed = query_ollama(model, choice_prompt, num_predict=1)
        resp = res.get("response", "")
        context = res.get("context", [])
        print(f"Choice Test Prompt Response: {repr(resp)} | Token ID: {context[-1] if context else None}")

        # (3) 段階評価 (1〜5) テスト
        print("\n--- Part 3: 数値スコア (1〜5) の 1トークン出力 ---")
        score_prompt = "<start_of_turn>user\nRate the quality from 1 to 5 (5 is best). Quality is perfect. Answer with only the number.<end_of_turn>\n<start_of_turn>model\n<thought>\n\n</thought>\n"
        if "qwen" in model.lower():
            score_prompt = "<|im_start|>user\nRate the quality from 1 to 5 (5 is best). Quality is perfect. Answer with only the number.<|im_end|>\n<|im_start|>assistant\n"
            
        res, elapsed = query_ollama(model, score_prompt, num_predict=1)
        resp = res.get("response", "")
        context = res.get("context", [])
        print(f"Score Test Prompt Response: {repr(resp)} | Token ID: {context[-1] if context else None}")

        # (4) 日本語生ラベル直接出力の検証 (サブワード分割リスク)
        print("\n--- Part 4: 日本語生ラベル直接出力検証 (サブワード分割チェック) ---")
        jp_prompts = [
            ("2文字: 「正解」", "<start_of_turn>user\nこの文は正しいですか？「正解」または「不正解」で答えてください。<end_of_turn>\n<start_of_turn>model\n<thought>\n\n</thought>\n"),
            ("1文字: 「良」", "<start_of_turn>user\nこの文を評価してください。「良」「可」「不可」の1文字で答えてください。<end_of_turn>\n<start_of_turn>model\n<thought>\n\n</thought>\n"),
        ]
        if "qwen" in model.lower():
            jp_prompts = [
                ("2文字: 「正解」", "<|im_start|>user\nこの文は正しいですか？「正解」または「不正解」で答えてください。<|im_end|>\n<|im_start|>assistant\n"),
                ("1文字: 「良」", "<|im_start|>user\nこの文を評価してください。「良」「可」「不可」の1文字で答えてください。<|im_end|>\n<|im_start|>assistant\n"),
            ]
            
        for label, p in jp_prompts:
            # num_predict=1 の場合
            res1, _ = query_ollama(model, p, num_predict=1)
            # num_predict=2 の場合（もし分割されていたら2トークン目で続きが出る）
            res2, _ = query_ollama(model, p, num_predict=2)
            print(f"[{label}]")
            print(f"  num_predict=1: {repr(res1.get('response'))}")
            print(f"  num_predict=2: {repr(res2.get('response'))}")

    return results

if __name__ == "__main__":
    run_experiment_1()
