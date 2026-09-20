# jev-localsystem (JEV: Judge & Evaluation via Vectorized-logits)

文章生成を伴わず、特定トークンのLogit（未正規化対数確率）のみを抽出・計算する超低遅延・省VRAMなローカル判定システム。

---

## 概要

従来の生成AIによる評価（LLM-as-a-Judge）では、文章生成ループ（Autoregressive Decode）に伴う推論遅延、KVキャッシュ肥大化によるVRAM枯渇、フォーマット崩れや出力ゆらぎが課題でした。
本システムは、**文章デコードを行わず、単一Forwardパスから目的トークンIDのLogitを直接抽出・マスキング・正規化** することで、決定的かつ高精度なスコアリング・分類をローカル環境で高速実行します。

### 理論的裏付け（主要諸元）

1. **出力強制とLogitマスキング（Guided Generation）**  
   - Willard & Louf (2023), *Efficient Guided Generation for Large Language Models* ([arXiv:2307.09702](https://arxiv.org/abs/2307.09702))  
   - 有限オートマトン（FSM）に基づく語彙マスキングにより目的トークンのみを計算。
2. **生成を伴わない判定特化モデルの優位性（Encoder-Only）**  
   - He et al. (2021/2023), *DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training* ([arXiv:2111.09543](https://arxiv.org/abs/2111.09543))  
   - デコードループとKVキャッシュを排除し、省VRAMかつ双方向文脈理解による高精度判定。
3. **LLM判定器のバイアス排除（LLM-as-a-Judge）**  
   - Zheng et al. (2023), *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* ([arXiv:2306.05685](https://arxiv.org/abs/2306.05685))  
   - 位置バイアス・冗長性バイアスを構造的に緩和するプロンプト設計。

---

## ドキュメント一覧

- **基本設計書**: [JEV-BD-001_基本設計書.md](docs/design/JEV-BD-001_基本設計書.md) (Rev.1.1)
- **推論判定パイプライン詳細設計書**: [JEV-DD-001_推論判定パイプライン詳細設計書.md](docs/design/JEV-DD-001_推論判定パイプライン詳細設計書.md) (Rev.1.1)
- **検証方法設計書**: [JEV-TEST-001_検証方法設計書.md](docs/test/JEV-TEST-001_検証方法設計書.md) (Rev.1.1)
- **Git Flow & ブランチ運用方針**: [docs/setup/git_flow_and_branch_policy.md](docs/setup/git_flow_and_branch_policy.md)
- **実機検証レポート群**: `docs/logs/` (Issue #1 〜 Issue #9)

---

## 本番推奨モデル階層アーキテクチャ

RTX 3060 12GB 環境下における実機横断ベンチマーク（Issue #9）および包括テストに基づき確定。

| Tier | モデル識別名 | パラメータ / 量子化 | VRAM専有 | 推論速度 | 特徴・実機検証成果 |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **Tier 1<br>(Primary / 主軸)** | **`qwen3:8b`** | 8.0B (Q4_K_M) | 約 5.2 GB (100% GPU) | **約 39 ms** | **本番標準主軸エンジン**。<br>・Multi-Label分離度 **36.612pt**（過去最高記録）<br>・空`<think>`タグ注入による即時Logit抽出<br>・日本語ビジネス規程、長文2,000T維持、直列キュー検証済 |
| **Tier 2<br>(Lightweight / 常駐)** | **`phi4-mini:latest`** | 3.8B (Q4/Q8) | 約 2.5 GB (100% GPU) | **約 39 ms** | **高速常駐ゲートキーパー**。<br>・省VRAM（常時起動で残り9.5GB空き確保）<br>・Multi-Label分離度 **14.656pt**<br>・二値判定（Noul: Yes/No）、高速前処理・仕分け専用 |

---

## 環境構築とテスト実行 (uv 管理)

本プロジェクトは [uv](https://github.com/astral-sh/uv) により依存関係およびテスト環境が厳密に管理されています。

```bash
# 依存関係の同期（開発依存を含む）
uv sync --extra dev

# モデル汎用パイプライン包括テストの実行 (全6項目)
uv run pytest tests/test_model_pipeline.py

# 任意モデルへの動的切り替えテスト実行
uv run pytest tests/test_model_pipeline.py --model phi4-mini:latest
```

---

## ディレクトリ構成

```text
jev-localsystem/
├── docs/
│   ├── design/         # 仕様書・設計書正本 (JEV-BD-001, JEV-DD-001)
│   ├── test/           # 検証方法設計書 (JEV-TEST-001)
│   ├── logs/           # 実機検証レポート & 実測JSONデータ (Issue #1〜#9)
│   └── setup/          # 環境構築・ブランチ運用規約
├── scripts/            # 包括検証スクリプトハーネス
│   ├── verify_model_pipeline_suite.py  # モデル汎用包括検証ハーネス
│   └── benchmark_model_matrix.py       # 横断モデルベンチマーク
├── tests/              # Pytest自動テストスイート
│   ├── conftest.py                     # 動的モデルオプション設定
│   └── test_model_pipeline.py          # 包括パイプラインテスト (6 passed)
├── pyproject.toml      # uv プロジェクト定義
├── uv.lock             # 依存関係ロックファイル
└── README.md
```
