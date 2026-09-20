import json
import urllib.request
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5-coder:14b-instruct"

# 重いコンテキストを持つ判定用プロンプト (約1500〜2000文字)
LARGE_CONTEXT = """
SYSTEM LOG DUMP:
[2026-09-20 20:00:01] INFO [auth] User login successful: uid=1002
[2026-09-20 20:00:02] DEBUG [db] Query: SELECT * FROM sessions WHERE token = 'xyz789'
[2026-09-20 20:00:03] INFO [worker] Processing job id=94819 payload_size=4096kb
[2026-09-20 20:00:04] WARN [memory] Buffer utilization reached 88%
[2026-09-20 20:00:05] ERROR [worker] Task failed with timeout: connection to backend timed out
[2026-09-20 20:00:06] CRITICAL [kernel] Out of memory killer invoked, terminating process 8841 (worker)
[2026-09-20 20:00:07] INFO [system] Service restarted by systemd supervisor
[2026-09-20 20:00:08] INFO [auth] Healthcheck passed on port 8080
""" * 5  # 5回繰り返して大容量コンテキスト化

def get_vram_usage():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,nounits,noheader"],
            encoding="utf-8"
        )
        used, total = out.strip().split(",")
        return f"{used.strip()}MB / {total.strip()}MB"
    except Exception:
        return "Unknown"

# JEV VRAM Manager: 直列キューイング排他制御
class VRAMManagerQueue:
    def __init__(self, max_concurrency=1):
        self.semaphore = threading.Semaphore(max_concurrency)
        self.lock = threading.Lock()
        self.completed_count = 0

    def execute_inference(self, req_id: int, prompt: str):
        queue_start = time.perf_counter()
        # 直列キューで待機 (排他制御)
        with self.semaphore:
            wait_time_ms = (time.perf_counter() - queue_start) * 1000
            infer_start = time.perf_counter()
            
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
            
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    res = json.loads(resp.read().decode("utf-8"))
                infer_time_ms = (time.perf_counter() - infer_start) * 1000
                total_time_ms = (time.perf_counter() - queue_start) * 1000
                
                with self.lock:
                    self.completed_count += 1
                    current_completed = self.completed_count
                    
                vram = get_vram_usage()
                
                return {
                    "req_id": req_id,
                    "status": "SUCCESS",
                    "response": res.get("response", "").strip(),
                    "wait_time_ms": wait_time_ms,
                    "infer_time_ms": infer_time_ms,
                    "total_time_ms": total_time_ms,
                    "vram_at_finish": vram,
                    "completed_order": current_completed
                }
            except Exception as e:
                return {
                    "req_id": req_id,
                    "status": f"ERROR: {e}",
                    "response": None,
                    "wait_time_ms": wait_time_ms,
                    "infer_time_ms": 0,
                    "total_time_ms": (time.perf_counter() - queue_start) * 1000,
                    "vram_at_finish": get_vram_usage(),
                    "completed_order": -1
                }

def run_stress_test():
    total_requests = 10
    manager = VRAMManagerQueue(max_concurrency=1)

    print("================================================================")
    print(" Issue #8: VRAM枯渇回避のためのシーケンシャルキュー・ストレステスト")
    print(f" Target Model: {MODEL_NAME}")
    print(f" 同時投入リクエスト数: {total_requests} 件 (巨大コンテキスト)")
    print(f" 初期 VRAM 使用量:     {get_vram_usage()}")
    print("================================================================\n")

    # ウォームアップ
    payload = {"model": MODEL_NAME, "prompt": "hi", "options": {"num_predict": 1}, "stream": False}
    urllib.request.urlopen(
        urllib.request.Request(OLLAMA_URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    )

    print(f"🚀 {total_requests}件のリクエストを 10並列スレッドから同時に一斉投下します...")
    stress_start = time.perf_counter()

    results = []
    with ThreadPoolExecutor(max_workers=total_requests) as executor:
        futures = []
        for i in range(1, total_requests + 1):
            prompt = (
                f"<|im_start|>user\n"
                f"Analyze this log dump:\n{LARGE_CONTEXT}\n"
                f"Is there a Critical OOM error? A: Yes, B: No. Answer with only A or B.<|im_end|>\n"
                f"<|im_start|>assistant\nAnswer: "
            )
            futures.append(executor.submit(manager.execute_inference, i, prompt))

        for f in as_completed(futures):
            res = f.result()
            results.append(res)
            rid = res["req_id"]
            stat = res["status"]
            resp = res["response"]
            wait = res["wait_time_ms"]
            infer = res["infer_time_ms"]
            vram = res["vram_at_finish"]
            order = res["completed_order"]
            print(f"[完了 #{order}] Req-{rid:02d}: {stat} | 出力: {repr(resp)} | 待機: {wait:6.1f}ms | 推論: {infer:6.1f}ms | VRAM: {vram}")

    total_elapsed = time.perf_counter() - stress_start
    success_count = sum(1 for r in results if r["status"] == "SUCCESS")
    oom_count = sum(1 for r in results if "OOM" in str(r["status"]) or "out of memory" in str(r["status"]).lower())

    avg_infer = sum(r["infer_time_ms"] for r in results if r["status"] == "SUCCESS") / max(1, success_count)

    print("\n================================================================")
    print(" ストレステスト集計結果")
    print("================================================================")
    print(f" 投下リクエスト総数:   {total_requests} 件")
    print(f" 成功完了件数:         {success_count} / {total_requests} ({success_count/total_requests*100:.1f}%)")
    print(f" OOM / クラッシュ件数: {oom_count} 件 (ゼロ目標)")
    print(f" 全体所要時間:         {total_elapsed:.2f} 秒")
    print(f" 平均単体推論時間:     {avg_infer:.1f} ms")
    print(f" 終了時 VRAM 使用量:   {get_vram_usage()}")
    print("================================================================")

if __name__ == "__main__":
    run_stress_test()
