import json
import urllib.request
import time

OLLAMA_URL = "http://localhost:11434/api/generate"

test_tasks = [
    {
        "id": "T1_BugCheck",
        "question": "Does this Python function have a ZeroDivisionError?\n```python\ndef divide(a, b):\n    return a / b\n```",
        "zero_prompt": "<start_of_turn>user\nDoes this Python function have a ZeroDivisionError? A: Yes, B: No.\n```python\ndef divide(a, b):\n    return a / b\n```\nAnswer with only A or B.<end_of_turn>\n<start_of_turn>model\n<thought>\n\n</thought>\n",
        "gen_prompt": "<start_of_turn>user\nDoes this Python function have a ZeroDivisionError? Explain briefly and conclude with Yes or No.\n```python\ndef divide(a, b):\n    return a / b\n```<end_of_turn>\n<start_of_turn>model\n<thought>\n\n</thought>\n"
    },
    {
        "id": "T2_SQLInjection",
        "question": "Is this SQL query vulnerable to SQL Injection?\n`query = 'SELECT * FROM users WHERE name = ' + user_input`",
        "zero_prompt": "<start_of_turn>user\nIs this query vulnerable to SQL Injection? A: Yes, B: No.\nquery = 'SELECT * FROM users WHERE name = ' + user_input\nAnswer with only A or B.<end_of_turn>\n<start_of_turn>model\n<thought>\n\n</thought>\n",
        "gen_prompt": "<start_of_turn>user\nIs this query vulnerable to SQL Injection? Explain briefly.\nquery = 'SELECT * FROM users WHERE name = ' + user_input<end_of_turn>\n<start_of_turn>model\n<thought>\n\n</thought>\n"
    },
    {
        "id": "T3_SyntaxValid",
        "question": "Is this JSON syntax valid?\n`{'key': 'value',}`",
        "zero_prompt": "<start_of_turn>user\nIs this JSON syntax valid? A: Yes, B: No.\n{'key': 'value',}\nAnswer with only A or B.<end_of_turn>\n<start_of_turn>model\n<thought>\n\n</thought>\n",
        "gen_prompt": "<start_of_turn>user\nIs this JSON syntax valid? Explain briefly.\n{'key': 'value',}<end_of_turn>\n<start_of_turn>model\n<thought>\n\n</thought>\n"
    },
    {
        "id": "T4_LogAnomaly",
        "question": "Is this log an anomaly? `FATAL: Out of memory while allocating 4GB buffer`",
        "zero_prompt": "<start_of_turn>user\nIs this log an anomaly? A: Yes, B: No.\nFATAL: Out of memory while allocating 4GB buffer\nAnswer with only A or B.<end_of_turn>\n<start_of_turn>model\n<thought>\n\n</thought>\n",
        "gen_prompt": "<start_of_turn>user\nIs this log an anomaly? Explain briefly.\nFATAL: Out of memory while allocating 4GB buffer<end_of_turn>\n<start_of_turn>model\n<thought>\n\n</thought>\n"
    }
]

def query(model: str, prompt: str, num_predict: int):
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
    with urllib.request.urlopen(req, timeout=120) as resp:
        res = json.loads(resp.read().decode("utf-8"))
    elapsed_ms = (time.perf_counter() - start) * 1000
    return res, elapsed_ms

def run_benchmark():
    model = "gemma4-12b-it-Q4_K_M:latest"
    print("================================================================")
    print(" Issue #3: 単一Forward推論 (Zero-Decode) vs 文章生成 ベンチマーク")
    print(f" Target Model: {model}")
    print("================================================================\n")
    
    # ウォームアップ
    print("Warming up model...")
    query(model, "warmup", num_predict=1)
    print("Warm-up complete.\n")
    
    zero_latencies = []
    gen_latencies = []
    
    print("--- 個別タスク実測 ---")
    for task in test_tasks:
        tid = task["id"]
        
        # 1. Zero-Decode (1-Forward)
        res_zero, t_zero = query(model, task["zero_prompt"], num_predict=1)
        resp_zero = res_zero.get("response", "").strip()
        eval_dur_zero = res_zero.get("eval_duration", 0) / 1e6
        zero_latencies.append(t_zero)
        
        # 2. 通常文章生成 (Generate: 32 tokens)
        res_gen, t_gen = query(model, task["gen_prompt"], num_predict=32)
        resp_gen = res_gen.get("response", "").strip().replace("\n", " ")
        eval_count_gen = res_gen.get("eval_count", 0)
        eval_dur_gen = res_gen.get("eval_duration", 0) / 1e6
        gen_latencies.append(t_gen)
        
        speedup = t_gen / t_zero if t_zero > 0 else 0
        reduction = (1 - (t_zero / t_gen)) * 100 if t_gen > 0 else 0
        
        print(f"[{tid}]")
        print(f"  Zero-Decode: {t_zero:.1f}ms (eval: {eval_dur_zero:.1f}ms, output: {repr(resp_zero)})")
        print(f"  通常生成:     {t_gen:.1f}ms (tokens: {eval_count_gen}, eval: {eval_dur_gen:.1f}ms)")
        print(f"  生成内容抜粋: {repr(resp_gen[:50])}...")
        print(f"  => 高速化比:  {speedup:.2f}倍 (レイテンシ {reduction:.1f}% 削減)\n")
        
    avg_zero = sum(zero_latencies) / len(zero_latencies)
    avg_gen = sum(gen_latencies) / len(gen_latencies)
    overall_speedup = avg_gen / avg_zero
    overall_reduction = (1 - (avg_zero / avg_gen)) * 100
    
    print("================================================================")
    print(" ベンチマーク集計結果")
    print("================================================================")
    print(f" 平均レイテンシ (Zero-Decode): {avg_zero:.1f} ms")
    print(f" 平均レイテンシ (通常生成):    {avg_gen:.1f} ms")
    print(f" 全体平均高速化比:             {overall_speedup:.2f} 倍")
    print(f" レイテンシ削減率:             {overall_reduction:.1f} %")
    print("================================================================")

if __name__ == "__main__":
    run_benchmark()
