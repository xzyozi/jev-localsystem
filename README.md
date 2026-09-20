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
- **推論判定パイプライン詳細設計書**: [JEV-DD-001_推論判定パイプライン詳細設計書.md](docs/design/JEV-DD-001_推論判定パイプライン詳細設計書.md) (Rev.1.3)
- **検証方法設計書**: [JEV-TEST-001_検証方法設計書.md](docs/test/JEV-TEST-001_検証方法設計書.md) (Rev.1.1)
- **Git Submodule 連携・他プロジェクト組み込みガイド**: [docs/how-to/git_submodule_integration_guide.md](docs/how-to/git_submodule_integration_guide.md)
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

## 利用方法

### 1. Python SDK として直接利用（最速・通信オーバーヘッドゼロ）
```python
from jev import JudgePipeline

# パイプラインの初期化 (Tier 1: qwen3:8b デフォルト)
pipeline = JudgePipeline()

# 真偽判定 (Noul)
res = pipeline.judge_noul(
    context_text="ユーザーが正しい二要素認証コードを入力してログインしました。",
    rule_definition="正常な認証アクティビティであるかを判定してください。"
)
print(res.verdict)  # "Yes"
print(res.confidence)  # 1.0 (所要時間: 39ms)

# 単一選択 (Choice / スワップ検証付き)
res_choice = pipeline.judge_choice(
    context_text="SELECT * FROM users WHERE id = 1;",
    labels=["Database", "Frontend", "CSS"],
    swap_verify=True
)
print(res_choice.verdict)  # "Database"
```

### 2. REST API サーバーとして起動（HTTP / マイクロサービス）
```bash
# REST API サーバー起動 (デフォルト: http://0.0.0.0:8000)
uv run jev-server

# 開発用自動リロード起動
uv run jev-server --reload --port 8000
```
起動後、ブラウザで Swagger UI（**http://localhost:8000/docs**）を開き、対話的に全APIをテストできます。

#### 主要エンドポイント一覧
| メソッド | パス | 概要 |
| :--- | :--- | :--- |
| `POST` | `/api/v1/noul` | 真偽判定（Yes/No、規程適合判定） |
| `POST` | `/api/v1/choice` | 単一選択（A/Bスワップ位置バイアス相殺付） |
| `POST` | `/api/v1/score` | 段階評価（1〜5の加重連続値期待値） |
| `POST` | `/api/v1/multilabel` | 複数ラベル分類（独立Sigmoid判定） |
| `POST` | `/api/v1/judge` | 全タスク統合エントリーポイント |
| `GET` | `/health` | サーバー死活およびOllama疎通確認 |
| `GET` | `/api/v1/vram/metrics` | 直列セマフォ稼働状況・累計処理件数 |
| `GET` | `/api/v1/models` | 本番推奨モデル階層カタログ一覧 |

### 3. 他プロジェクトへの組み込み（Git Submodule 方式）
他リポジトリ（自律エージェントや母艦アプリ等）から JEV を内部モジュールとして組み込んで利用する場合、Git Submodule として配置することで、外部パス依存ゼロ・通信オーバーヘッドゼロで利用できます。

```bash
# 親プロジェクト側でサブモジュールとして追加
git submodule add https://github.com/xzyozi/jev-localsystem.git submodules/jev-localsystem
```

親プロジェクトの `pyproject.toml` に `"jev-localsystem @ file://./submodules/jev-localsystem"` を指定することで、プロセス内関数呼出し（最速 39ms）として直接インポートできます。  
詳細は [Git Submodule 連携・他プロジェクト組み込みガイド](docs/how-to/git_submodule_integration_guide.md) を参照してください。

---

## テスト実行 (全25テスト完走)

```bash
# 依存関係の同期
uv sync --extra dev

# 全テストの実行 (Core + Model + REST API)
uv run pytest -v

# コード品質チェック
uv run ruff check src/
```

---

## ディレクトリ構成

```text
jev-localsystem/
├── docs/
│   ├── design/         # 仕様書・設計書正本 (JEV-BD-001, JEV-DD-001 Rev.1.3)
│   ├── test/           # 検証方法設計書 (JEV-TEST-001)
│   ├── logs/           # 実機検証レポート & 実測JSONデータ (Issue #1〜#9)
│   └── setup/          # 環境構築・ブランチ運用規約
├── src/
│   └── jev/            # JEV 本番実装パッケージ
│       ├── dto.py              # Pydantic型安全DTOスキーマ
│       ├── exceptions.py       # 安全回路例外群
│       ├── vram_manager.py     # 直列セマフォ・リミッター契約
│       ├── prompt_builder.py   # 思考抑制Prefill・記号化
│       ├── zero_decode.py      # 1サイクル推論クライアント
│       ├── result_mapper.py    # 空白対数和・確率計算
│       ├── pipeline.py         # JudgePipeline 統合ファサード
│       └── server/             # FastAPI REST API サーバー
│           ├── app.py          # FastAPI アプリファクトリ・例外ハンドラー
│           ├── config.py       # サーバー設定
│           ├── main.py         # サーバー起動CLI (jev-server)
│           ├── schemas.py      # API用リクエスト/レスポンススキーマ
│           └── api/v1/         # v1 ルーター (judge, system)
├── tests/              # Pytest自動テストスイート (25 passed)
│   ├── conftest.py             # 動的モデルオプション設定
│   ├── test_core_pipeline.py   # コアパイプライン単体・統合テスト
│   ├── test_api_server.py      # REST API サーバー包括テスト
│   └── test_model_pipeline.py  # モデル包括差異検証テスト
├── pyproject.toml      # uv プロジェクト定義
├── uv.lock             # 依存関係ロックファイル
└── README.md
```
