# モデル学習・予測ガイド

[← ユーザーガイドに戻る](./USER_GUIDE.md)

---

## 目次

1. [特徴量エンジニアリング](#1-特徴量エンジニアリング)
2. [モデル学習](#2-モデル学習)
3. [予測実行](#3-予測実行)

---

## 1. 特徴量エンジニアリング

### 1.1 特徴量グループ

システムは6グループ・100以上の特徴量を自動生成します。

#### テクニカル特徴量（technical）
```
- return_1m, return_3m, return_6m, return_1y（期間リターン）
- volatility_20d, volatility_60d（ボラティリティ）
- rsi_14, rsi_28（RSI）
- macd, macd_signal, macd_hist（MACD）
- bb_position（ボリンジャーバンド位置）
- momentum_10d, momentum_20d（モメンタム）
- drawdown_max_60d（最大ドローダウン）
```

#### ファンダメンタル特徴量（fundamental）
```
- per, pbr, pcfr（バリュエーション）
- roe, roa（収益性）
- net_margin, operating_margin（利益率）
- debt_equity_ratio（財務レバレッジ）
- revenue_growth_yoy, profit_growth_yoy（成長率）
- fcf_yield（フリーキャッシュフロー利回り）
- accruals_ratio（アクルーアルズ比率）
```

#### マーケット特徴量（market）
```
- market_cap, market_cap_log（時価総額）
- turnover_value_20d（売買代金）
- volume_ratio_20d（出来高比率）
- sector_relative_return（セクター相対リターン）
- price_to_52w_high（52週高値比）
```

#### EDINET特徴量（edinet）
```
- rd_ratio（研究開発費率）
- capex_ratio（設備投資率）
- working_capital_ratio（運転資本比率）
- sales_growth_edinet（売上成長率）
```

#### 開示特徴量（disclosure）
```
- earnings_revision_flag（業績修正フラグ）
- guidance_change（業績予想変更）
- buyback_flag（自社株買いフラグ）
- disclosure_count_30d（開示件数）
```

#### グローバル指数特徴量（global_index）
```
- sp500_correlation_60d（S&P500相関）
- nasdaq_beta（NASDAQベータ）
- vix_sensitivity（VIX感応度）
```

### 1.2 特徴量の確認・分析

```bash
# 特徴量重要度を分析
uv run python scripts/analyze_features.py

# 詳細統計を出力
uv run python scripts/analyze_features.py --model-path models/model_20251210_184936 --analyze-data
```

**出力ファイル:**
- `data/performance/feature_importance_YYYYMMDD.csv`
- `data/performance/feature_statistics_YYYYMMDD.csv`

### 1.3 特徴量のカスタマイズ

`src/features/builder.py`で特徴量グループを制御できます。

```python
# 特徴量グループの有効/無効
FEATURE_GROUPS = {
    'technical': True,
    'fundamental': True,
    'market': True,
    'edinet': True,
    'disclosure': True,
    'global_index': True,
    'trends': False,  # Google Trendsは通常無効
}
```

---

## 2. モデル学習

### 2.1 基本的な学習

```bash
# デフォルト設定で学習
uv run python scripts/train_model.py

# スリープ防止付きで実行（Mac）
caffeinate -i uv run python scripts/train_model.py --train-start 2022-09-01 --train-end 2025-09-30 --holding-days 10
```

**デフォルト設定:**
- 学習期間: 過去5年
- バリデーション: Walk-Forward CV
- モデル: LightGBM Regression
- 目的変数: 20営業日後リターン

### 2.2 学習オプション

```bash
# 期間を指定
uv run python scripts/train_model.py \
    --train-start 2020-01-01 \
    --train-end 2024-06-30

# バリデーションをスキップ（高速化）
uv run python scripts/train_model.py --skip-validation

# モデルタイプを変更
uv run python scripts/train_model.py --task binary   # 2値分類
uv run python scripts/train_model.py --task ranking  # ランキング

# 保有期間を変更
uv run python scripts/train_model.py --holding-days 10  # 10日後リターン
```

### 2.3 ターゲットタイプ（目的変数）

予測対象となるターゲット（目的変数）を選択できます。

```bash
# 対TOPIX超過リターン（デフォルト・推奨）
uv run python scripts/train_model.py --target-type excess

# 絶対リターン（従来方式）
uv run python scripts/train_model.py --target-type absolute

# リターン5分位（ランキング学習向け）
uv run python scripts/train_model.py --target-type quintile
```

| ターゲット | 説明 | 用途 |
|-----------|------|------|
| `excess` | 対TOPIX超過リターン | マーケットニュートラル戦略、銘柄選択力の評価 |
| `absolute` | 絶対リターン | ロングオンリー戦略 |
| `quintile` | クロスセクショナル5分位 | ランキング学習、相対評価 |

**推奨: `excess`（対TOPIX超過リターン）**

理由:
- 市場全体の動きを除去し、純粋な銘柄選択力を学習
- 上昇相場・下落相場で公平な評価が可能
- ロングショート戦略に適している

### 2.4 Walk-Forward クロスバリデーション

時系列データに適した検証方法を採用しています。

```
|-------- 学習期間 (3年) --------|-- 検証 (3ヶ月) --|
                                 |-- エンバーゴ (20日) --|

→ 1ヶ月ずつウィンドウをスライド
```

**パラメータ:**
- 学習期間: 756営業日（約3年）
- 検証期間: 63営業日（約3ヶ月）
- ステップ: 21営業日（約1ヶ月）
- エンバーゴ: 20営業日（ラベル漏洩防止）

### 2.5 ハイパーパラメータ最適化

Optunaを使用して最適なパラメータを探索します。

```bash
# 基本的な最適化（50試行）
uv run python scripts/optimize_params.py --n-trials 100

# 最適化メトリクスを指定
uv run python scripts/optimize_params.py --n-trials 100 --metric icir

# 特徴量グループも最適化
uv run python scripts/optimize_params.py --n-trials 100 --optimize-features
```

**探索パラメータ:**
- `num_leaves`: 5〜128
- `learning_rate`: 0.001〜0.3
- `feature_fraction`: 0.3〜1.0
- `bagging_fraction`: 0.5〜1.0
- `min_child_samples`: 5〜100

**最適化メトリクス:**
- `ic`: Information Coefficient（予測相関）
- `icir`: IC Information Ratio（IC安定性）
- `sharpe`: シャープレシオ
- `hit_rate`: 方向正解率

### 2.6 モデル出力

学習後のファイル:

```
data/models/
├── model_YYYYMMDD_HHMMSS.lgb  # 学習済みモデル
├── latest.lgb                  # 最新モデルへのコピー
└── validation_results_YYYYMMDD.csv  # バリデーション結果

models/
└── latest.lgb                  # プロダクションモデル
```

### 2.7 評価指標の見方

```
=== バリデーション結果 ===
IC (Information Coefficient): 0.045
  → 予測と実績の相関。0.03以上で有効、0.05以上で良好

ICIR (IC Information Ratio): 0.72
  → ICの安定性。0.5以上で安定

Hit Rate: 53.2%
  → 方向正解率。52%以上で有効

Sharpe Ratio: 1.25
  → リスク調整後リターン。1.0以上で良好

Spread: 2.3%
  → ロング-ショートポートフォリオの月間スプレッド
```

---

## 3. 予測実行

### 3.1 基本的な予測

```bash
# 本日の予測
uv run python scripts/predict.py

# 特定日の予測
uv run python scripts/predict.py --date 2024-12-06

# トップN銘柄数を指定
uv run python scripts/predict.py --top-n 50
```

### 3.2 予測オプション

```bash
# 使用モデルを指定
uv run python scripts/predict.py --model-path data/models/model_20241201.lgb

# フィルタ条件
uv run python scripts/predict.py \
    --min-price 500 \          # 最低株価
    --min-volume 50000 \       # 最低出来高
    --exclude-sectors "銀行業,保険業"  # 除外セクター
```

### 3.3 予測出力

**出力ファイル:** `data/predictions/daily/predictions_YYYYMMDD.csv`

```csv
rank,code,company_name,sector,market,score,price,market_cap
1,7203,トヨタ自動車,輸送用機器,プライム,0.847,2500,35000000
2,6758,ソニーグループ,電気機器,プライム,0.823,12000,15000000
...
```

**カラム説明:**
- `rank`: 予測スコア順位
- `code`: 銘柄コード
- `score`: 予測スコア（0〜1、高いほど有望）
- `price`: 直近株価
- `market_cap`: 時価総額（百万円）

### 3.4 予測結果の解釈

- **スコア 0.8以上**: 非常に有望（上位10%）
- **スコア 0.6〜0.8**: 有望（上位30%）
- **スコア 0.4〜0.6**: 中立
- **スコア 0.4未満**: 弱気

> **注意:** 予測は統計的な傾向であり、個別銘柄の投資判断は自己責任で行ってください。

---

[← データ収集ガイド](./DATA_COLLECTION.md) | [運用ガイド →](./OPERATIONS.md)
