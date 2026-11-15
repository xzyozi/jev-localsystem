# Layer 2 実機検証レポート: モデル能力向けメタモルフィック評価（極性反転・Choice巡回・ノイズ耐性）

- **関連 Issue**: 親 [#62](https://github.com/xzyozi/jev-localsystem/issues/62) / 子 [#65](https://github.com/xzyozi/jev-localsystem/issues/65)
- **評価スクリプト**: [verify_model_metamorphic.py](file:///Users/saigoutsubasa/Desktop/python/jev-localsystem/scripts/verify_model_metamorphic.py)
- **生データ**: [2026-11-14_model_metamorphic_raw.json](file:///Users/saigoutsubasa/Desktop/python/jev-localsystem/docs/logs/2026-11-14_model_metamorphic_raw.json)
- **検証日**: 2026-11-14

## 1. 検証目的

モデル自身の読解力・位置バイアス耐性・指示追従力に依存する Layer 2 の評価軸として、以下 3 項目の実測ベースラインを Tier 1（`qwen3:8b`）および Tier 2（`phi4-mini:latest`）で取得した。

1. **極性反転（Polarity Flip）**: Noul タスクにおける肯定・否定質問ペア 5 組（10 問）の `Yes`/`No` 反転率・正答率・マージン符号反転率
2. **3〜4択 Choice 巡回置換不変性**: 明確な正解を持つ 3〜4 択問題 4 問の全巡回シフト（計 14 回）での勝者ラベル一致率・正答率
3. **軽微な表記ゆれ・無関係ノイズ耐性**: Noul 2 問・Choice 2 問に対し、空白/全半角ゆれ・改行増減・末尾無関係注記の 3 変種（計 12 変種）を与えた際の判定維持率と信頼度/マージン変動量

## 2. 実測結果比較（`qwen3:8b` vs `phi4-mini:latest`）

| 評価軸 | 指標 | `qwen3:8b` (Tier 1) | `phi4-mini:latest` (Tier 2) |
| :--- | :--- | :--- | :--- |
| 極性反転 (5ペア) | `flip_consistency_rate` | 100.0% | 100.0% |
| 極性反転 (5ペア) | `both_correct_rate` | 100.0% | 100.0% |
| 極性反転 (5ペア) | `margin_sign_flip_rate` | 100.0% | 100.0% |
| 極性反転 (参考) | 肯定マージン範囲 / 否定マージン範囲 | `+13.7058 〜 +19.9909` / `-13.5834 〜 -19.9999` | `+3.0288 〜 +8.2305` / `-4.5005 〜 -9.4898` |
| Choice 巡回置換 (4問/14シフト) | `cyclic_consistency_rate` | 100.0% | 100.0% |
| Choice 巡回置換 (4問/14シフト) | `correct_cyclic_rate` | 100.0% | 100.0% |
| Choice 巡回置換 (4問/14シフト) | `shift_accuracy_rate` | 100.0% | 100.0% |
| ノイズ耐性 (4問/12変種) | `verdict_preservation_rate` | 100.0% | 100.0% |
| ノイズ耐性 (4問/12変種) | `max_confidence_delta` / `avg_confidence_delta` | `0.0` / `0.0` | `0.002` / `0.00037` |
| ノイズ耐性 (4問/12変種) | `max_margin_delta` | `0.6931` | `1.6819` |

## 3. 考察

1. **判定一貫性**:
   - 事実関係が明確なベンチマークセットにおいては、`qwen3:8b`・`phi4-mini:latest` ともに極性反転・3〜4択巡回置換・軽微ノイズの 3 軸すべてで **100.0%** の一貫性率を達成した。
2. **信頼度・マージン余裕の差**:
   - 一方で内部マージンに着目すると、`qwen3:8b` は極性反転時に `±13.5〜20.0pt` の明確な分離幅を示しノイズ下でも `max_confidence_delta = 0.0`（`max_margin_delta = 0.6931`）と極めて安定しているのに対し、`phi4-mini:latest` はマージン幅が `3.0〜9.5pt` とやや小さく、ノイズによる最大マージン変動も `1.6819pt` と相対的に大きい。
3. **pytest 品質ゲート**:
   - [test_model_metamorphic.py](file:///Users/saigoutsubasa/Desktop/python/jev-localsystem/tests/test_model_metamorphic.py) には `@pytest.mark.ollama` を付与し、一貫性率ベース（極性反転 `>= 80%`、Choice 巡回 `>= 75%`、ノイズ維持率 `>= 80%` かつ信頼度変動上限 `<= 0.1`）の品質ゲートとして組み込んだ。
