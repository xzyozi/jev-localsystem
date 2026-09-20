# Git Flow 運用規約 ＆ ブランチ自動管理ガイド

本ドキュメントは、プロジェクトにおける **Git Flow 運用ルール**、および **GitHub / CI/CD / ローカル Git** を活用したブランチの自動クリーンアップ・管理仕様を定義します。

---

## 1. ブランチ戦略 (Git Flow 概要)

### 主要ブランチ (Long-lived Branches)
- **`main`**: 本番環境用ブランチ。常時デプロイ可能・安定したコードを保持します。
- **`develop`**: 開発用メインブランチ。最新の開発成果が集約されます。

### サポートブランチ (Short-lived Branches)
- **`feat/<機能名>`**: 新機能開発用ブランチ。`develop` から分岐し、`develop` へ PR を作成してマージします。
- **`fix/<修正内容>`**: バグ修正用ブランチ。`develop` から分岐し、`develop` へマージします。
- **`docs/<ドキュメント名>`**: ドキュメント更新専用ブランチ。
- **`hotfix/<緊急修正>`**: 本番の急激な不具合修正用。`main` から分岐し、`main` および `develop` へマージします。

---

## 2. CI/CD ＆ リモート自動化 (GitHub / GitHub Actions)

不要なリモートブランチが残留しないよう、以下の2層の自動クリーンアップを組み込んでいます。

### ① GitHub 設定: PRマージ後のリモートブランチ自動削除
- GitHubの管理画面 (`Settings` > `General` > `Pull Requests`) にて **`Automatically delete head branches`** を有効化。
- PRが `develop` または `main` にマージされた時点で、リモートの作業ブランチが自動削除されます。

### ② GitHub Actions: 定期クリーンアップ ワークフロー
- 設定ファイル: [branch-cleanup.yml](../../.github/workflows/branch-cleanup.yml)
- **トリガー**: 毎週月曜午前 0:00 (UTC) または 手動実行 (`workflow_dispatch`)
- **機能**: `develop` にマージ済みのリモートブランチを自動検出して一括削除（`main`, `develop` は自動保護）。

---

## 3. ローカル Git 開発環境の自動化設定

リモートで削除されたブランチがローカル環境に古い追跡情報 (`[gone]`) として残るのを防ぐため、以下の設定を推奨します。

### ① `git fetch` への自動 Prune 設定
以下のコマンドを実行すると、`git fetch` や `git pull` を行うたびに、リモートで消えたブランチのローカル追跡情報を自動的に削除します。

```bash
git config --global fetch.prune true
```

### ② ローカル `[gone]` ブランチを一括削除する Git エイリアス
追跡元リモートブランチが消えたローカルブランチを一括で整理・削除するコマンドを追加します。

```bash
# エイリアスの設定 (PowerShell / Git Bash 共通)
git config --global alias.cleanup "!git branch -vv | grep '\[gone\]' | awk '{print $1}' | xargs -r git branch -d"
```

#### 使い方
```bash
git cleanup
```
実行すると、リモートで既にマージ・削除されたローカルブランチが一括で安全に削除されます。
