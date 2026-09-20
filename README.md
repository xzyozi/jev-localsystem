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

- **基本設計書**: [JEV-BD-001_基本設計書.md](docs/design/JEV-BD-001_基本設計書.md)
- **設計ガイドライン**: [docs/design/TEMPLATE/README.md](docs/design/TEMPLATE/README.md)
- **Git Flow & ブランチ運用方針**: [docs/setup/git_flow_and_branch_policy.md](docs/setup/git_flow_and_branch_policy.md)
- **Mermaid CI 検証運用**: [docs/setup/mermaid_ci_validation.md](docs/setup/mermaid_ci_validation.md)

---

## ディレクトリ構成

```text
jev-localsystem/
├── docs/
│   ├── analysis/       # 分析・調査資料
│   ├── archive/        # アーカイブ
│   ├── design/         # 仕様書・設計書正本 (SSOT)
│   │   ├── JEV-BD-001_基本設計書.md
│   │   └── TEMPLATE/   # 設計書テンプレート群
│   ├── features/       # 機能仕様書
│   ├── how-to/         # 実装・操作手順
│   ├── logs/           # 運用ログ・検証記録
│   ├── review/         # レビュー記録
│   ├── setup/          # 環境構築・開発規約
│   └── test/           # テスト計画・結果
├── LICENSE
└── README.md
```
