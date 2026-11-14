# Layer 1 実機検証レポート: バッチ連続推論（KVキャッシュあり）vs 単発推論（キャッシュリセット）等価性

- **関連 Issue**: 親 [#62](https://github.com/xzyozi/jev-localsystem/issues/62) / 子 [#64](https://github.com/xzyozi/jev-localsystem/issues/64)
- **検証スクリプト**: [verify_batch_cache_equivalence.py](file:///Users/saigoutsubasa/Desktop/python/jev-localsystem/scripts/verify_batch_cache_equivalence.py)
- **生データ**: [2026-11-14_batch_cache_equivalence_raw.json](file:///Users/saigoutsubasa/Desktop/python/jev-localsystem/docs/logs/2026-11-14_batch_cache_equivalence_raw.json)
- **検証日**: 2026-11-14

## 1. 検証目的

モデル能力に依存しないシステム整合性（Layer 1）の検証として、Ollama の KV キャッシュを保持したまま連続実行するバッチ推論と、`keep_alive: 0` によりキャッシュをリセットして個別に実行する単発推論で、判定結果および出力確率分布が一致するかを確認した。

## 2. 検証対象モデルと条件

- **対象モデル**: `qwen3:8b`, `phi4-mini:latest`
- **コンテキスト長**: `short_1000c` (1,000 文字), `medium_3000c` (3,000 文字)
- **検証パス**:
  1. `/api/v1/evaluate`（`choice`, `score`, `noul`, `multilabel` の 4 問連続バッチ vs 各問前に `keep_alive: 0` でリセットした単発推論）
  2. `JudgePipeline.judge_multilabel`（連続ラベル評価 vs 各ラベル評価ごとにキャッシュをリセットする単発推論）

## 3. 実測結果

### 3.1 `qwen3:8b`

| 条件 | API / メソッド | レイテンシ (Batch / Single 合計) | 判定一致 | 最大確率差 | スコア差 | 最大マージン差 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `short_1000c` | `/api/v1/evaluate` | 25459.71 ms / 28379.35 ms | True | 0.0 | 0.0 | 0.0 |
| `short_1000c` | `JudgePipeline.judge_multilabel` | - | True | 0.0 | - | 0.0 |
| `medium_3000c` | `/api/v1/evaluate` | 60874.78 ms / 62893.61 ms | True | 0.0 | 0.0 (`1.233` vs `1.233`) | 0.0 |
| `medium_3000c` | `JudgePipeline.judge_multilabel` | - | True | 0.0 | - | 0.0 |

### 3.2 `phi4-mini:latest`

| 条件 | API / メソッド | レイテンシ (Batch / Single 合計) | 判定一致 | 最大確率差 | スコア差 | 最大マージン差 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `short_1000c` | `/api/v1/evaluate` | 14604.13 ms / 14489.76 ms | True | 0.0 | 0.0 (`3.434` vs `3.434`) | 0.0 |
| `short_1000c` | `JudgePipeline.judge_multilabel` | - | True | 0.0023 | - | 0.011 |
| `medium_3000c` | `/api/v1/evaluate` | 35946.36 ms / 35764.62 ms | True | 0.0 | 0.0 (`3.468` vs `3.468`) | 0.0 |
| `medium_3000c` | `JudgePipeline.judge_multilabel` | - | True | 0.0005 | - | 0.003 |

## 4. 結論・副次的知見

1. **Layer 1 等価性の成立**:
   - `qwen3:8b` では全条件・全タスクで Batch と Single が完全一致（差分 `0.0`）した。
   - `phi4-mini:latest` でも `/api/v1/evaluate` は全条件で差分 `0.0`、`JudgePipeline.judge_multilabel` でも最大確率差 `0.0023`・最大マージン差 `0.011` と微小であり、判定ラベルの一致 (`matched_match=True`) を確認した。
2. **`/api/v1/evaluate` の混合タスクにおけるキャッシュ挙動**:
   - `/api/v1/evaluate` で `choice` / `score` / `noul` / `multilabel` のように異なるタスク種別が連続する場合、`PromptBuilder` の `system_prompt` 自体がタスク種別ごとに異なるため、異なるタスク間では Prefix Cache が効きにくく、同じタスク種別の連続実行や `judge_multilabel` 内の連続ラベル評価でキャッシュが効く構造になっている。
3. **文字数変化に対する挙動の再現性**:
   - `qwen3:8b` の `score` 値が `short_1000c` (`4.89`) から `medium_3000c` (`1.233`) へ変化したケースでも、Batch と Single は完全に同一の値 (`1.233`) を返しており、バッチ／キャッシュ機構由来のズレではないことを確認した。
