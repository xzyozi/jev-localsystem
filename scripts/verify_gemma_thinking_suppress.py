import json
import urllib.request
import time

OLLAMA_URL = "http://localhost:11434/api/generate"

def test_query(model: str, prompt: str, system: str = "", raw: bool = False, num_predict: int = 1):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "raw": raw,
        "options": {
            "num_predict": num_predict,
            "temperature": 0.0,
        }
    }
    if system:
        payload["system"] = system
        
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=data, headers={"Content-Type": "application/json"})
    
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            res = json.loads(resp.read().decode("utf-8"))
        elapsed_ms = (time.perf_counter() - start) * 1000
        return res, elapsed_ms
    except Exception as e:
        return {"error": str(e)}, 0

def run_tests():
    models = ["gemma4-12b-it-Q4_K_M:latest", "gemma-4-py_coder:latest"]
    
    # 思考抑止のための各種アプローチ
    experiments = [
        {
            "name": "Exp-1: 思考禁止システムプロンプト (Standard Template)",
            "system": "You must NOT think. Do not use any thinking tags or explanations. Output ONLY the single word: Yes or No.",
            "prompt": "Is the sky blue? Answer with Yes or No directly.",
            "raw": False
        },
        {
            "name": "Exp-2: 思考タグ事前注入 Prefill (raw=True)",
            "system": "",
            "prompt": "<start_of_turn>user\nIs the sky blue? Answer with Yes or No.<end_of_turn>\n<start_of_turn>model\n<thought>\nNo thought needed.\n</thought>\n",
            "raw": True
        },
        {
            "name": "Exp-3: 回答プレフィックス事前注入 'Answer: ' (raw=True)",
            "system": "",
            "prompt": "<start_of_turn>user\nIs the sky blue? Answer with Yes or No.<end_of_turn>\n<start_of_turn>model\nAnswer: ",
            "raw": True
        },
        {
            "name": "Exp-4: 'The answer is ' 誘導プレフィックス (raw=True)",
            "system": "",
            "prompt": "<start_of_turn>user\nIs the sky blue? Answer with Yes or No.<end_of_turn>\n<start_of_turn>model\nThe answer is ",
            "raw": True
        },
        {
            "name": "Exp-5: プレーンテキスト誘導 (raw=True, ターンタグなし)",
            "system": "",
            "prompt": "Question: Is the sky blue?\nAnswer (Yes or No): ",
            "raw": True
        },
    ]

    print("=========================================================")
    print(" Gemma Thinking 抑止・1トークン判定検証 (方針B)")
    print("=========================================================\n")

    for model in models:
        print(f"#########################################################")
        print(f" Target Model: {model}")
        print(f"#########################################################")
        for exp in experiments:
            print(f"\n--- {exp['name']} ---")
            res, elapsed = test_query(
                model=model,
                prompt=exp["prompt"],
                system=exp["system"],
                raw=exp["raw"],
                num_predict=1
            )
            resp_str = res.get("response", "")
            done_reason = res.get("done_reason", "")
            eval_dur = res.get("eval_duration", 0) / 1e6
            prompt_eval_dur = res.get("prompt_eval_duration", 0) / 1e6
            context = res.get("context", [])
            last_tok = context[-1] if context else None
            
            print(f"  Response repr: {repr(resp_str)}")
            print(f"  Last Token ID: {last_tok}")
            print(f"  Latency: {elapsed:.1f}ms (prompt_eval: {prompt_eval_dur:.1f}ms, eval: {eval_dur:.1f}ms)")
            print(f"  Done Reason: {done_reason}")
            
            # 成功判定
            is_success = resp_str.strip().lower() in ["yes", "no"]
            print(f"  Result: {'[SUCCESS! DIRECT JUDGMENT]' if is_success else '[FAIL / NON-JUDGMENT TOKEN]'}")

if __name__ == "__main__":
    run_tests()
