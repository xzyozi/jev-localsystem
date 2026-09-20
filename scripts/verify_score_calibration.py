import json
import urllib.request
import math
import time

OLLAMA_OPENAI_URL = "http://localhost:11434/v1/chat/completions"
MODEL_NAME = "qwen2.5-coder:14b-instruct"

# 4つの品質レベルの評価テストデータ
test_cases = [
    {
        "id": "Case_1_Excellent",
        "description": "最高品質コード (型ヒント・docstring・バリデーション完備)",
        "code": """def calculate_discounted_price(price: float, discount_rate: float) -> float:
    \"\"\"Calculate the final price after applying discount rate (0.0 to 1.0).\"\"\"
    if not (0.0 <= discount_rate <= 1.0):
        raise ValueError('Discount rate must be between 0.0 and 1.0')
    if price < 0:
        raise ValueError('Price cannot be negative')
    return round(price * (1.0 - discount_rate), 2)""",
        "expected_range": (4.5, 5.0)
    },
    {
        "id": "Case_2_Good",
        "description": "良質コード (正常動作するがdocstringやバリデーションが簡潔)",
        "code": """def calc_discount(price, rate):
    return price - (price * rate)""",
        "expected_range": (3.0, 4.0)
    },
    {
        "id": "Case_3_Poor",
        "description": "低品質コード (非効率・マジックナンバー・変数名不良)",
        "code": """def f(x, y):
    # discount calculation
    a = x
    for i in range(1):
        a = a * (1 - y)
    return a""",
        "expected_range": (1.8, 2.8)
    },
    {
        "id": "Case_4_Critical",
        "description": "壊れたコード (構文エラー・未定義変数・クラッシュ必至)",
        "code": """def broken_func(price, rate)
    retrun price / 0 + undefined_var""",
        "expected_range": (1.0, 1.5)
    }
]

def softmax(logits, temperature=1.0):
    scaled = [l / temperature for l in logits]
    max_l = max(scaled)
    exp_l = [math.exp(l - max_l) for l in scaled]
    sum_exp = sum(exp_l)
    return [e / sum_exp for e in exp_l]

def query_score_distribution(code: str, temperature: float = 1.0):
    system_prompt = (
        "You are an expert code reviewer. Rate the code quality strictly on a scale from 1 to 5.\n"
        "1: Terrible / Broken\n"
        "2: Poor / Bad practice\n"
        "3: Average / Acceptable\n"
        "4: Good / Clean\n"
        "5: Excellent / Production ready\n"
        "Respond with ONLY a single digit number (1, 2, 3, 4, or 5)."
    )
    user_content = f"Rate this Python code:\n```python\n{code}\n```"

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "max_tokens": 1,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": 20
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(OLLAMA_OPENAI_URL, data=data, headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        res = json.loads(resp.read().decode("utf-8"))
    latency_ms = (time.perf_counter() - start) * 1000

    choice = res["choices"][0]
    top_items = choice.get("logprobs", {}).get("content", [{}])[0].get("top_logprobs", [])

    # 1〜5のトークンのlogprobを抽出 (デフォルトは非常に小さい負の値)
    score_logits = {}
    for item in top_items:
        tok = item["token"].strip()
        if tok in ["1", "2", "3", "4", "5"]:
            score_logits[tok] = item["logprob"]

    # 欠落しているトークンにはペナルティ最小値 (-20.0) を補完
    full_logits = []
    for d in ["1", "2", "3", "4", "5"]:
        full_logits.append(score_logits.get(d, -20.0))

    probs = softmax(full_logits, temperature=temperature)

    # 期待値計算: sum(i * P(i))
    expected_score = sum((i + 1) * probs[i] for i in range(5))

    return {
        "raw_logits": full_logits,
        "probs": probs,
        "expected_score": expected_score,
        "latency_ms": latency_ms,
        "top_token": choice["message"]["content"].strip()
    }

def run_experiment():
    print("================================================================")
    print(" Issue #7: Scoreタスク (期待値計算) 分布とキャリブレーション検証")
    print(f" Target Model: {MODEL_NAME}")
    print("================================================================\n")

    results = []

    for case in test_cases:
        cid = case["id"]
        desc = case["description"]
        exp_range = case["expected_range"]

        res = query_score_distribution(case["code"], temperature=1.0)
        score = res["expected_score"]
        probs = res["probs"]
        t_ms = res["latency_ms"]
        top = res["top_token"]

        is_in_range = exp_range[0] <= score <= exp_range[1]

        print(f"[{cid}] - {desc}")
        print(f"  最尤トークン: '{top}' | 期待値スコア: {score:.3f} (想定: {exp_range[0]}〜{exp_range[1]})")
        print(f"  確率分布 P(1)〜P(5):")
        for i in range(5):
            bar = "#" * int(probs[i] * 30)
            print(f"    [{i+1}]: {probs[i]*100:5.1f}% | {bar}")
        print(f"  判定精度: {'OK (期待範囲内)' if is_in_range else 'OutOfRange'} | 所要時間: {t_ms:.1f}ms\n")

        results.append({
            "id": cid,
            "expected_score": score,
            "probs": probs,
            "latency_ms": t_ms
        })

    # スコアの滑らかさ・順序性検証
    scores = [r["expected_score"] for r in results]
    is_monotonic = scores[0] > scores[1] > scores[2] > scores[3]

    print("================================================================")
    print(" 総合評価サマリー")
    print("================================================================")
    print(f" Case 1 (最高):   {scores[0]:.3f} 点")
    print(f" Case 2 (良質):   {scores[1]:.3f} 点")
    print(f" Case 3 (低品質): {scores[2]:.3f} 点")
    print(f" Case 4 (壊れた): {scores[3]:.3f} 点")
    print(f" 単調減少性 (品質に応じた連続順序性): {'完全成立 (Pass!)' if is_monotonic else '逆転あり (Fail)'}")
    print("================================================================")

if __name__ == "__main__":
    run_experiment()
