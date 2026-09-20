# Mermaid CI 検証運用

## 目的

`docs/` 配下の Markdown に含まれる Mermaid 図を、GitHub Actions 上でレンダリング検証する。AI Agent を含む変更で Mermaid 記法が壊れた場合、マージ前にCIを失敗させる。

## 対象

- 対象ファイル: `docs/**/*.md`
- 対象ブロック: ` ```mermaid ` で開始するフェンスドコードブロック
- 検証対象の変更種別: 追加・変更・コピー・リネーム
- 削除されたファイルは検証しない。

## 検証方法

検証スクリプト `.github/scripts/validateMermaidMarkdown.mjs` は、PRまたはpushの比較元コミットとHEADコミットの差分から対象Markdownを取得する。各Mermaidブロックを一時 `.mmd` ファイルへ抽出し、固定バージョンの Mermaid CLI でSVGにレンダリングする。

構文不正、未対応の記法、括弧・引用符・矢印などの破損によりレンダリングに失敗した場合、CIは失敗する。エラーには元のMarkdownファイル、Mermaidブロック番号、開始行を表示する。

## 実行契機

Mermaid検証ワークフローは、次の場合に実行する。

- `main` または `develop` に対するPRで `docs/**/*.md` が変更された場合
- `main` または `develop` へのpushで `docs/**/*.md` が変更された場合
- `workflow_dispatch` による手動実行

通常のPR・pushでは差分ファイルだけを検証する。手動実行では全 `docs/**/*.md` を検証できるようにし、導入時のベースライン確認や定期的な全件確認に使用する。

## 依存関係と再現性

Mermaid CLIはJavaScript実装であるため、GitHub Actionsのジョブ内でのみNode.jsを利用する。開発者のローカル環境にNode.jsの導入は必須ではない。

CI内で `npx --yes --package @mermaid-js/mermaid-cli@11.4.2 mmdc` を実行し、Mermaid CLIの直接依存を完全固定する。開発者のローカル環境に `package.json`、`package-lock.json`、npmの導入は不要とする。`@latest` のような都度の最新版取得は使用せず、npmのキャッシュでCIの再実行を高速化する。

検証用の一時 `.mmd` とSVGはOSの一時ディレクトリにのみ出力し、リポジトリへ保存しない。

## 制約

この検証は図をレンダリングできることを保証するが、図の内容が業務・設計として正しいことまでは判断しない。矢印の方向、遷移漏れ、用語の誤りなどは、通常どおりPRレビューで確認する。

共通テーマやMermaid設定を将来導入した場合は、その設定変更時に差分検証ではなく全件検証を行う。
