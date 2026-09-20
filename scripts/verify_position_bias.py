import json
import urllib.request
import time

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma4-12b-it-Q4_K_M:latest"

# 5つの比較評価シナリオ (Candidate 1 vs Candidate 2)
# ground_truth: 'C1' (1が明確に優れている), 'C2' (2が明確に優れている), 'TIE' (互角・甲乙つけがたい)
test_pairs = [
    {
        "id": "Pair_1_Clear_CodeQuality",
        "task": "どちらのPythonコードが安全で優れていますか？",
        "c1": "def get_user(db, uid): return db.execute('SELECT * FROM users WHERE id = ?', (uid,)).fetchone()",
        "c2": "def get_user(db, uid): return db.execute('SELECT * FROM users WHERE id = ' + uid).fetchone()",
        "ground_truth": "C1"  # C1がSQL Injection対策されていて安全
    },
    {
        "id": "Pair_2_Clear_Politeness",
        "task": "どちらのカスタマーサポート回答が適切で親切ですか？",
        "c1": "ご不便をおかけし大変申し訳ございません。以下の手順で設定をご確認いただけますでしょうか。",
        "c2": "マニュアル読んでください。書いてあります。",
        "ground_truth": "C1"  # C1が圧倒的に親切
    },
    {
        "id": "Pair_3_Subtle_Documentation",
        "task": "どちらの関数の説明コメントが適切ですか？（ほぼ同等の品質）",
        "c1": "# ユーザーIDからアカウント情報を取得する関数",
        "c2": "# 指定されたユーザーIDに紐づくアカウント情報を返却します",
        "ground_truth": "TIE" # 拮抗（バイアスが出やすい）
    },
    {
        "id": "Pair_4_Clear_AlgorithmEfficiency",
        "task": "どちらの探索アルゴリズムがソート済み配列に対して優れていますか？",
        "c1": "線形探索 (O(N)): 先頭から順に要素を比較する",
        "c2": "二分探索 (O(log N)): 中央値と比較して探索範囲を半分にする",
        "ground_truth": "C2"  # C2が圧倒的に効率的
    },
    {
        "id": "Pair_5_Subtle_VariableNaming",
        "task": "どちらの変数名が適切ですか？（ほぼ同等の品質）",
        "c1": "user_account_list",
        "c2": "user_accounts",
        "ground_truth": "TIE" # 拮抗（バイアスが出やすい）
    }
]

def query_ollama(prompt: str):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "raw": True,
        "stream": False,
        "options": {
            "num_predict": 1,
            "temperature": 0.0,
        }
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=data, headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        res = json.loads(resp.read().decode("utf-8"))
    elapsed_ms = (time.perf_counter() - start) * 1000
    return res.get("response", "").strip(), elapsed_ms

def build_prompt(task_desc: str, opt_a: str, opt_b: str):
    return (
        f"<start_of_turn>user\n"
        f"質問: {task_desc}\n\n"
        f"A: {opt_a}\n"
        f"B: {opt_b}\n\n"
        f"優れた方を A または B の1文字のみで回答してください。<end_of_turn>\n"
        f"<start_of_turn>model\n"
        f"<thought>\n\n</thought>\n"
    )

def run_experiment():
    print("================================================================")
    print(" Issue #4: LLM-as-a-Judge 位置バイアス測定とスワップ検証")
    print(f" Target Model: {MODEL_NAME}")
    print("================================================================\n")

    # ウォームアップ
    query_ollama("warmup")

    results = []
    consistent_count = 0
    bias_towards_first_count = 0

    for pair in test_pairs:
        pid = pair["id"]
        c1 = pair["c1"]
        c2 = pair["c2"]
        gt = pair["ground_truth"]

        # 1. 順方向 (Forward): A=C1, B=C2
        prompt_fwd = build_prompt(pair["task"], c1, c2)
        ans_fwd, t_fwd = query_ollama(prompt_fwd)

        # 2. 逆方向 (Swap): A=C2, B=C1
        prompt_swap = build_prompt(pair["task"], c2, c1)
        ans_swap, t_swap = query_ollama(prompt_swap)

        # 判定の対応付け
        # 順方向: 'A'ならC1選択, 'B'ならC2選択
        chosen_fwd = "C1" if ans_fwd == "A" else ("C2" if ans_fwd == "B" else ans_fwd)
        # 逆方向: 'A'ならC2選択, 'B'ならC1選択
        chosen_swap = "C2" if ans_swap == "A" else ("C1" if ans_swap == "B" else ans_swap)

        # 一貫性の判定
        is_consistent = (chosen_fwd == chosen_swap)
        if is_consistent:
            consistent_count += 1

        # 位置バイアス（常に先頭のAを選んでしまったか？）
        is_position_bias = (ans_fwd == "A" and ans_swap == "A")
        if is_position_bias:
            bias_towards_first_count += 1

        # スワップ統合判定 (Swap Resolve)
        if is_consistent:
            final_verdict = chosen_fwd
            resolve_status = "CONSISTENT_WINNER"
        elif is_position_bias:
            final_verdict = "TIE (POSITION_BIAS_DETECTED)"
            resolve_status = "BIASED_FALLBACK_TO_TIE"
        else:
            final_verdict = "INCONCLUSIVE"
            resolve_status = "INCONSISTENT"

        print(f"[{pid}] (Ground Truth: {gt})")
        print(f"  Forward (A=C1, B=C2) => 出力: {ans_fwd} (選択: {chosen_fwd}) [{t_fwd:.1f}ms]")
        print(f"  Swap    (A=C2, B=C1) => 出力: {ans_swap} (選択: {chosen_swap}) [{t_swap:.1f}ms]")
        print(f"  一貫性 (Consistency): {'一致 (Pass)' if is_consistent else '不一致 (Fail)'}")
        print(f"  位置バイアス検出:     {'あり (両方Aを選択)' if is_position_bias else 'なし'}")
        print(f"  => スワップ統合結果:  {final_verdict} ({resolve_status})\n")

        results.append({
            "id": pid,
            "ground_truth": gt,
            "ans_fwd": ans_fwd,
            "chosen_fwd": chosen_fwd,
            "ans_swap": ans_swap,
            "chosen_swap": chosen_swap,
            "is_consistent": is_consistent,
            "is_position_bias": is_position_bias,
            "final_verdict": final_verdict
        })

    consistency_rate = (consistent_count / len(test_pairs)) * 100
    bias_rate = (bias_towards_first_count / len(test_pairs)) * 100

    print("================================================================")
    print(" 位置バイアス・スワップ検証集計")
    print("================================================================")
    print(f" テストケース数:        {len(test_pairs)} 件")
    print(f" 一貫性率 (Consistency): {consistency_rate:.1f} % ({consistent_count}/{len(test_pairs)})")
    print(f" 位置バイアス発生率:    {bias_rate:.1f} % ({bias_towards_first_count}/{len(test_pairs)})")
    print("================================================================")

if __name__ == "__main__":
    run_experiment()
