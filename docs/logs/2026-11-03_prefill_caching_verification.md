# 複数質問・Multi-Label Prefill 重複排除（State-First最適化）検証レポート

- **検証日時**: 2026-11-03
- **対象 Issue**: [#22 perf(core): 複数質問・Multi-Labelで同一stateを重複してprefillしている](https://github.com/xzyozi/jev-localsystem/issues/22)
- **検証環境**:
  - ホスト: macOS (Apple Silicon M3, 16GB Memory)
  - 推論バックエンド: Ollama (v0.40.13, Metal iGPU 加速)
  - 対象モデル:
    - **Tier 1 (主軸モデル)**: `qwen3:8b` (5.2GB)
    - **Tier 2 (軽量モデル)**: `phi4-mini:latest` (2.5GB)
  - 仮想環境: Python 3.12 (`uv`)

---

## 1. 検証目的と背景

`/api/v1/evaluate`（Jev互換バッチ評価API）や Multi-Label タスクでは、単一の共通長文テキスト（`state` / `context_text`）に対して複数の質問・ラベル（N問）を順次推論する。

### 従来の課題（Query-First）
従来の `PromptBuilder` は、プロンプトの先頭に可変の `【判定基準】` や `【評価カテゴリ】` を配置し、末尾に不変の `【対象テキスト】` を配置していた。
Decoder-only Transformer の因果的アテンション（Causal Attention）の性質上、プロンプトの先頭トークンが変化すると以降の全トークンの Key/Value テンソルを再計算（Prefill）する必要があり、Ollama / llama.cpp の Prefix Caching（プロンプトキャッシュ）が完全に不発となっていた。そのため、質問数 N 回分、長文テキスト全体の重い Prefill 計算が毎回重複して走っていた。

### 改善案（State-First）
不変かつ長大な `【対象テキスト】` をプロンプトの先頭に固定配置し、質問・ラベル固有の基準やカテゴリを末尾に配置する。
これにより、1問目で計算された先頭トークンの KV キャッシュが 2問目以降で 100% ヒットし、Prefill 計算をスキップして末尾の差分のみを即座に計算することが可能となる。

---

## 2. 実機ベンチマーク実測結果

モデルキャッシュを独立させた状態で、現行（Query-First）と改善案（State-First）のレイテンシを実機計測（`scripts/benchmark_prefill_caching.py`）。

### 2.1 実用バッチ領域 (~500 トークン / 1,500文字)

| モデル | 質問数 | Query-First (現行)<br>1問目 / 2問目以降avg / 合計 | State-First (改善案)<br>1問目 / 2問目以降avg / 合計 | 2問目以降<br>速度向上比 | 全体バッチ<br>速度比 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **qwen3:8b** (Tier 1) | 4問 | 7,204ms / 6,271ms / 26,018ms | 7,285ms / **373ms** / **8,405ms** | **16.80x 高速** | **3.10x 高速** |
| | 8問 | 7,183ms / 6,977ms / 56,019ms | 9,210ms / **357ms** / **11,712ms** | **19.52x 高速** | **4.78x 高速** |
| **phi4-mini:latest** (Tier 2) | 4問 | 3,890ms / 3,438ms / 14,205ms | 4,090ms / **117ms** / **4,440ms** | **29.49x 高速** | **3.20x 高速** |
| | 8問 | 4,065ms / 3,661ms / 29,690ms | 4,254ms / **117ms** / **5,072ms** | **31.34x 高速** | **5.85x 高速** |

> ※ 生ログ詳細データ: `docs/logs/2026-11-03_prefill_caching_benchmark.json`

### 2.2 単体 2 問連続推論検証
同一 State（約2,000文字）に対する 2 問連続推論のレイテンシ実測値：
- **Query-First**: 1問目: 5,294.6ms ──> 2問目: **5,292.2ms** (キャッシュ不発、全量再計算)
- **State-First**: 1問目: 11,845.7ms (初回ロード/Prefill) ──> 2問目: **215.3ms** (**25倍高速化・Prefill完全スキップ**)

---

## 3. 判定精度の非劣化検証

プロンプトの順序変更（State 先頭固定）による判定精度・一貫性への影響を、実機パイプラインテスト（[`tests/test_model_pipeline.py`](file:///Users/saigoutsubasa/Desktop/python/jev-localsystem/tests/test_model_pipeline.py)）および全テストスイートで検証：

1. **Choice A/B スワップ一貫性**: PASSED（整合率 75% 以上クリア）
2. **Multi-Label Sigmoid 分離度**: PASSED（マージンギャップ 6.0pt 以上クリア）
3. **Score 期待値単調減少性**: PASSED（High > Mid > Low を完全維持）
4. **コンテキスト減衰境界**: PASSED（Soft Limit 2,000T / Hard Limit 4,000T を維持）
5. **リポジトリ全テストスイート (121件)**: **100% PASS** (`121 passed in 51.66s`)

---

## 4. 結論

- `PromptBuilder` の全タスクプロンプト（Noul, Choice, Score, Multi-Label）において、`【対象テキスト】` を先頭に固定する State-First テンプレートへ刷新。
- バッチ評価および Multi-Label の直列推論において、2問目・2ラベル目以降のレイテンシが **約 16倍〜31倍高速化**（数秒 → 100〜300ms）され、かつ判定精度への悪影響はゼロ（全テスト合格）であることを確認した。
