# Git Submodule 連携・他プロジェクト組み込みガイド

本ドキュメントは、Zero-Decodeローカル判定基盤 **JEV (Judge & Evaluation via Vectorized-logits)** を、外部の親プロジェクト（自律エージェント、Trading Bot、業務スクリプト等の母艦）に **Git Submodule（内部モジュール）として組み込んで利用するための標準運用手順** を定義します。

---

## 1. 組み込みアーキテクチャの概要

JEV は独立リポジトリ（サテライト）として設計書・包括テストスイート・実機ログを自己完結で保持しつつ、親プロジェクト（母艦）側では **Git Submodule として特定コミットを固定（ピン留め）** して組み込みます。

- **通信オーバーヘッドゼロ**: HTTPサーバーを介さず、親プロジェクトのプロセス内から直接 `from jev import JudgePipeline` で最速呼出し（39ms）。
- **外部絶対パス依存ゼロ**: プロジェクト内にソースコード実体が同居するため、リポジトリ単体で完結し完全な環境再現性を保証。
- **自律的バージョン管理**: JEV側の破壊的変更に巻き込まれず、親プロジェクト側で検証済みの特定コミットで固定可能。

---

## 2. ディレクトリ配置構造の標準モデル

親プロジェクトのリポジトリルート配下に `submodules/` または `src/vendor/` を設け、その配下に JEV をサブモジュールとして配置します。

```text
my-parent-project/               # 親プロジェクト (母艦)
├── pyproject.toml               # 親プロジェクトの依存定義
├── src/
│   └── my_app/
│       └── agent.py             # JEV を直接呼ぶコード
└── submodules/
    └── jev-localsystem/         # ★ Git Submodule として同居
        ├── pyproject.toml       # JEV 本体の依存定義
        ├── src/
        │   └── jev/             # JEV Core 実体 (dto, pipeline, etc.)
        └── tests/               # JEV のテスト群
```

---

## 3. 親プロジェクトへの導入手順（3ステップ）

### ステップ 1: サブモジュールの追加
親プロジェクトのリポジトリルートで以下のコマンドを実行します。

```bash
# サブモジュールとして JEV を追加
git submodule add https://github.com/xzyozi/jev-localsystem.git submodules/jev-localsystem

# 親プロジェクト側でコミット
git commit -m "feat: JEVローカル判定基盤をサブモジュールとして追加"
```

### ステップ 2: 親プロジェクトの `pyproject.toml` 設定 (uv 連携)
親プロジェクトの仮想環境から JEV をシームレスにインポートできるよう、親の `pyproject.toml` に相対パス参照を追加します。

```toml
# my-parent-project/pyproject.toml

[project]
name = "my-parent-project"
dependencies = [
    # 相対パスによる JEV のローカルパッケージ参照
    "jev-localsystem @ file://./submodules/jev-localsystem"
]
```

または `tool.uv.sources` 構文を利用する場合:
```toml
[tool.uv.sources]
jev-localsystem = { path = "submodules/jev-localsystem", editable = true }
```

設定後、親プロジェクトで依存関係を同期します：
```bash
uv sync
```

### ステップ 3: Python コードからの直接呼出し
親プロジェクトのコード内で、標準ライブラリのように直接インポートして使用できます。

```python
# my_app/agent.py
from jev import JudgePipeline

# パイプライン初期化 (Tier 1: qwen3:8b がデフォルト適用)
pipeline = JudgePipeline()

# 真偽判定 (Noul)
result = pipeline.judge_noul(
    context_text="2026年9月21日 バックアップ処理が正常に完了しました。",
    rule_definition="バックアップ成功であるかを判定してください。"
)

print(result.verdict)     # "Yes"
print(result.confidence)  # 1.0 (所要時間: 約39ms)

# 単一選択 (Choice / A/Bスワップ位置バイアス相殺付き)
choice_result = pipeline.judge_choice(
    context_text="SELECT * FROM users WHERE id = 1;",
    labels=["Database", "Frontend", "CSS"],
    swap_verify=True
)
print(choice_result.verdict)  # "Database"
```

---

## 4. 日常の運用・更新フロー

### A. 別環境での親プロジェクトのクローン
サブモジュールを含む親プロジェクトを別のPCやCI環境でクローンする際は、`--recursive` オプションを付与します。

```bash
git clone --recursive <親リポジトリURL>
```

すでにクローン済みの場合は、以下でサブモジュールを初期化・同期します：
```bash
git submodule update --init --recursive
```

### B. JEV のバージョン（コミット）更新
JEV 側で新機能やモデル検証のアップデートが行われ、その最新成果を親プロジェクトに取り込みたい場合：

```bash
# サブモジュールを最新コミットに追従
git submodule update --remote submodules/jev-localsystem

# 親プロジェクト側で更新コミット
git add submodules/jev-localsystem
git commit -m "chore: JEVサブモジュールを最新コミットに更新"
```

---

## 5. 設計上の留意点・トラブルシューティング

- **サブモジュール内が空（空フォルダ）になっている場合**:
  `git submodule update --init --recursive` を実行して実体を取得してください。
- **VRAM排他制御の挙動**:
  JEV の `VRAMManager`（直列セマフォ）は Python プロセス内（またはシングルトン）で直列キューイングを保証します。単一GPU（RTX 3060等）環境下で親プロジェクトから多重スレッドで呼び出した場合でも、OOMを防止し100%完走します。
