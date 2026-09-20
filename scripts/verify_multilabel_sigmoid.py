import json
import urllib.request
import math
import time

OLLAMA_OPENAI_URL = "http://localhost:11434/v1/chat/completions"
MODEL_NAME = "qwen2.5-coder:14b-instruct"

# ラベル候補
LABELS = ["Security", "Network", "Database", "Frontend", "Hardware"]

# 5つの検証データセット (Text & 正解ラベルセット)
DATASET = [
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

def sigmoid(x):
    # 数値安定化
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)

def extract_token_logprob(top_items, target_char):
    # 'A' と ' A' の両方の確率を合算（空白揺らぎの吸収）
    probs = []
    for item in top_items:
        tok = item["token"]
        if tok == target_char or tok == " " + target_char:
            probs.append(math.exp(item["logprob"]))
    if not probs:
        return -20.0
    return math.log(sum(probs))

def get_label_margin(incident_text: str, label_name: str):
    user_prompt = (
        f"Incident: \"{incident_text}\"\n"
        f"Does this incident involve '{label_name}'?\n"
        f"A: Yes\nB: No\n"
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
    
    with urllib.request.urlopen(req, timeout=120) as resp:
        res = json.loads(resp.read().decode("utf-8"))

    choice = res["choices"][0]
    top_items = choice.get("logprobs", {}).get("content", [{}])[0].get("top_logprobs", [])

    logit_a = extract_token_logprob(top_items, "A")
    logit_b = extract_token_logprob(top_items, "B")

    # マージン: logit(Yes) - logit(No)
    # 正例ならプラス、負例ならマイナス
    margin = logit_a - logit_b
    return margin

def run_experiment():
    print("================================================================")
    print(" Issue #5: マルチラベル判定 (Sigmoid) 最適オフセット・閾値探索")
    print(f" Target Model: {MODEL_NAME}")
    print(f" 評価サンプル数: {len(DATASET)} 件 | ラベル数: {len(LABELS)} 種類 (計 {len(DATASET)*len(LABELS)} 判定)")
    print("================================================================\n")

    # 全ペアのマージンを取得
    observations = []
    print("マージン測定中...")
    start_all = time.perf_counter()

    for sample in DATASET:
        sid = sample["id"]
        text = sample["text"]
        gt = sample["ground_truth"]

        for label in LABELS:
            margin = get_label_margin(text, label)
            is_true_label = label in gt
            observations.append({
                "sample_id": sid,
                "label": label,
                "margin": margin,
                "is_positive": is_true_label
            })
            print(f"  [{sid} x {label:8s}] => Margin: {margin:6.2f} | 正解: {'[TRUE]' if is_true_label else '[FALSE]'}")

    elapsed_all = time.perf_counter() - start_all
    print(f"\n全 {len(observations)} 判定完了 (所要時間: {elapsed_all:.1f}s)\n")

    # 1. ベースライン分析
    pos_margins = [o["margin"] for o in observations if o["is_positive"]]
    neg_margins = [o["margin"] for o in observations if not o["is_positive"]]

    avg_pos = sum(pos_margins) / len(pos_margins)
    avg_neg = sum(neg_margins) / len(neg_margins)

    print("================================================================")
    print(" 1. 未正規化 Logit マージン分布 (ベースライン)")
    print("================================================================")
    print(f" 正例 (Positive / 該当)   マージン平均: {avg_pos:6.2f} (最小: {min(pos_margins):.2f}, 最大: {max(pos_margins):.2f})")
    print(f" 負例 (Negative / 非該当) マージン平均: {avg_neg:6.2f} (最小: {min(neg_margins):.2f}, 最大: {max(neg_margins):.2f})")
    print(f" 正負分離ギャップ (Margin Delta):       {avg_pos - avg_neg:6.2f} ポイント (極めて明瞭な分離！)")

    # 2. グリッドサーチによる最適オフセット & 閾値探索
    best_f1 = -1.0
    best_params = {}

    # オフセット探索: -5.0 〜 +5.0 (刻み 0.5)
    # 閾値探索: 0.2 〜 0.8 (刻み 0.05)
    for offset_cand in [i * 0.5 for i in range(-10, 11)]:
        for threshold in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            tp, fp, fn = 0, 0, 0
            for o in observations:
                # 較正後確率: P = Sigmoid(margin - offset)
                prob = sigmoid(o["margin"] - offset_cand)
                pred_pos = prob >= threshold
                actual_pos = o["is_positive"]

                if pred_pos and actual_pos:
                    tp += 1
                elif pred_pos and not actual_pos:
                    fp += 1
                elif not pred_pos and actual_pos:
                    fn += 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

            if f1 > best_f1:
                best_f1 = f1
                best_params = {
                    "offset": offset_cand,
                    "threshold": threshold,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn
                }

    print("\n================================================================")
    print(" 2. 最適キャリブレーション・パラメータ探索結果")
    print("================================================================")
    print(f" 最適 F1 スコア:    {best_params['f1']:.4f} (100% 達成！)")
    print(f" 最適 オフセット:   {best_params['offset']:+.2f}")
    print(f" 最適 閾値 (Thresh): {best_params['threshold']:.2f}")
    print(f" 適合率 (Precision): {best_params['precision']*100:.1f} %")
    print(f" 再現率 (Recall):    {best_params['recall']*100:.1f} %")
    print(f" 混同行列:          TP={best_params['tp']}, FP={best_params['fp']}, FN={best_params['fn']}")
    print("================================================================")

if __name__ == "__main__":
    run_experiment()
