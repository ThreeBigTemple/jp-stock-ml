# JP Stock ML - Claude Code 開発プロンプト

## プロジェクト概要

日本株の成長銘柄発掘を目的とした機械学習パイプライン。J-Quants API（スタンダードプラン）からデータを収集し、LightGBMで予測モデルを構築する。

## 現在の状態

**Phase 1: データ基盤構築 ✅ 完了**

以下が実装済み：
- J-Quants APIクライアント（認証、全エンドポイント対応）
- SQLiteデータベース（SQLAlchemy ORM）
- データ収集スクリプト（過去10年分一括、日次更新）

### 実装済みファイル構造

```
jp_stock_ml/
├── config/
│   ├── __init__.py
│   └── settings.py              # 設定・環境変数管理
├── src/
│   ├── collectors/
│   │   ├── jquants_client.py    # J-Quants APIクライアント
│   │   └── jquants_collector.py # データ収集ロジック
│   ├── database/
│   │   └── models.py            # SQLAlchemyモデル（Stock, Price, Financial等）
│   ├── features/                # ← Phase 2で実装
│   │   └── __init__.py
│   └── utils/
│       └── date_utils.py        # 日付ユーティリティ
├── scripts/
│   ├── init_database.py
│   ├── collect_historical.py
│   └── daily_update.py
└── pyproject.toml               # uv用
```

### データベーステーブル

| テーブル | 内容 | 主キー |
|---------|------|--------|
| stocks | 銘柄マスタ（コード、会社名、セクター、市場区分） | code |
| prices | 日次株価（OHLCV、調整済み価格） | code, date |
| financials | 財務データ（売上、利益、EPS等）Point-in-Time管理 | code, disclosed_date, period |
| trading_calendar | 取引カレンダー | date |
| topix | TOPIX日次データ | date |
| margin_balance | 信用取引週末残高 | code, date |
| short_selling | 業種別空売り比率 | date, sector_33_code |

## 次のタスク: Phase 2 - 特徴量エンジニアリング

### 実装すべきファイル

```
src/features/
├── __init__.py
├── technical.py      # テクニカル指標
├── fundamental.py    # ファンダメンタル指標
├── market.py         # 市場・需給指標
└── builder.py        # 特徴量マトリクス構築
```

### 特徴量候補

**テクニカル系（technical.py）**
```python
# モメンタム
- return_1m, return_3m, return_6m, return_12m  # 期間リターン
- momentum_20d, momentum_60d                    # モメンタム

# 相対強度
- rs_vs_topix      # TOPIX比リターン
- rs_rank_sector   # セクター内リターン順位（パーセンタイル）

# 移動平均
- ma_5_20_ratio    # 5日/20日MA比率
- ma_20_60_ratio   # 20日/60日MA比率
- price_vs_ma200   # 200日MA乖離率

# ボラティリティ
- volatility_20d, volatility_60d  # 標準偏差ベース
- atr_14d                          # ATR

# 出来高
- volume_ratio_20d   # 20日平均出来高比
- volume_ma_ratio    # 出来高移動平均比率
```

**ファンダメンタル系（fundamental.py）**
```python
# 成長性（最重要）
- revenue_growth_yoy      # 売上高成長率（前年同期比）
- revenue_growth_qoq      # 売上高成長率（前四半期比）
- op_income_growth_yoy    # 営業利益成長率
- eps_growth_yoy          # EPS成長率
- revenue_growth_3y_cagr  # 3年売上CAGR

# 収益性
- operating_margin        # 営業利益率
- operating_margin_change # 営業利益率変化
- roe, roa                # ROE, ROA
- gross_margin            # 粗利率（算出可能なら）

# バリュエーション
- per, pbr, psr           # PER, PBR, PSR
- ev_ebitda               # EV/EBITDA
- per_relative_sector     # セクター内相対PER

# 財務健全性
- equity_ratio            # 自己資本比率
- debt_equity_ratio       # D/Eレシオ

# 予想・サプライズ
- eps_surprise            # EPS実績 vs 予想
- guidance_revision       # 業績予想修正率
```

**市場・需給系（market.py）**
```python
# 信用取引
- margin_balance_ratio    # 信用倍率（買残/売残）
- margin_balance_change   # 信用残高変化率

# 空売り
- short_selling_ratio     # セクター空売り比率

# 市場全体
- topix_return_20d        # TOPIX 20日リターン（マーケット状態）
```

### 重要な実装ポイント

**1. Point-in-Time管理**
```python
# 財務データは disclosed_date（開示日）ベースで使用
# 予測時点で利用可能なデータのみ使用すること
def get_available_financials(code: str, as_of_date: date) -> pd.DataFrame:
    return df[df['disclosed_date'] <= as_of_date]
```

**2. 特徴量マトリクス構造**
```python
# 出力形式: 銘柄×日付のマトリクス
# columns: [code, date, feature_1, feature_2, ..., target]
# target: N日後リターン or 上昇フラグ
```

**3. ラグ処理**
```python
# 株価データは当日終値を使用可能
# 財務データは開示日の翌営業日から使用可能
```

### Phase 2完了後の次ステップ

**Phase 3: MLパイプライン**
- LightGBMモデル構築
- ウォークフォワード検証
- Optunaハイパーパラメータ最適化

**Phase 4: 運用**
- 日次予測スクリプト
- パフォーマンス監視

## 開発ルール

- パッケージ管理: `uv` を使用（pip不可）
- Python: 3.10以上
- 型ヒント推奨
- ログ出力: loggingモジュール使用

## 実行方法

```bash
# 環境セットアップ
uv sync

# DB初期化
uv run python scripts/init_database.py

# データ収集（初回）
uv run python scripts/collect_historical.py

# 日次更新
uv run python scripts/daily_update.py
```

## 参考: J-Quants APIレスポンス例

**財務データ（/fins/statements）**
```json
{
  "LocalCode": "72030",
  "DisclosedDate": "2024-05-08",
  "TypeOfCurrentPeriod": "FY",
  "NetSales": 45095325000000,
  "OperatingProfit": 5352934000000,
  "Profit": 4944933000000,
  "EarningsPerShare": 380.25,
  "ForecastNetSales": 46000000000000,
  "ChangesInNetSales": 0.082
}
```

**株価データ（/prices/daily_quotes）**
```json
{
  "Code": "72030",
  "Date": "2024-12-06",
  "Open": 2850.0,
  "High": 2875.0,
  "Low": 2840.0,
  "Close": 2865.0,
  "Volume": 12500000,
  "AdjustmentClose": 2865.0
}
```
