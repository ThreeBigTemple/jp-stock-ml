# データ収集ガイド

[← ユーザーガイドに戻る](./USER_GUIDE.md)

---

## 目次

1. [日次更新](#1-日次更新)
2. [ヒストリカルデータ収集](#2-ヒストリカルデータ収集)
3. [個別データソース収集](#3-個別データソース収集)
4. [データベース構造](#4-データベース構造)

---

## 1. 日次更新

毎日実行して最新データを取得します。

```bash
uv run python scripts/daily_update.py
```

**更新内容:**
- 直近7日分の株価
- 新規公開銘柄
- EDINET開示（過去30日）
- TDnet適時開示（過去30日）
- 海外指数

**オプション:**
```bash
# 更新日数を指定
uv run python scripts/daily_update.py --days 14

# 特定のデータのみ更新
uv run python scripts/daily_update.py --skip-edinet --skip-tdnet
```

---

## 2. ヒストリカルデータ収集

新規セットアップ時や、データを補完したい場合に使用します。

```bash
# 過去N年分を収集
uv run python scripts/collect_historical.py --years 5

# 期間を指定
uv run python scripts/collect_historical.py \
    --start-date 2019-01-01 \
    --end-date 2024-12-31
```

---

## 3. 個別データソース収集

### 3.1 EDINET（詳細財務）

EDINETデータ収集は2段階で行います：
1. **書類メタデータ取得**: 書類一覧（doc_id、提出日等）をDBに保存
2. **財務データ取得**: XBRLをダウンロード・パースして財務数値を抽出

```bash
# 過去5年分の書類メタデータを収集（デフォルト）
uv run python scripts/collect_edinet.py

# 特定の年のみ取得
uv run python scripts/collect_edinet.py --year 2024

# 過去N年分を取得
uv run python scripts/collect_edinet.py --years 3

# 取得状況を確認
uv run python scripts/collect_edinet.py --check

# 再取得（既存データを上書き）
uv run python scripts/collect_edinet.py --year 2024 --force
```

**財務データ（XBRL）の取得:**

書類メタデータのみでは財務数値（売上、営業利益等）は空です。XBRLをパースして財務データを取得する必要があります。

```bash
# 特定年のXBRLをダウンロード・パースして財務データを取得
uv run python scripts/collect_edinet.py --year 2021 --parse-xbrl

# 並列数を指定（デフォルト5、最大推奨10）
uv run python scripts/collect_edinet.py --year 2021 --parse-xbrl --workers 10

# 件数を制限してテスト
uv run python scripts/collect_edinet.py --year 2021 --parse-xbrl --parse-limit 100
```

> **注意:**
> - EDINET APIキーが必要です（環境変数 `EDINET_API_KEY` または `--api-key` オプションで指定）
> - XBRLパースは1年あたり数時間〜半日かかります
> - EDINET APIには1日あたりの呼び出し制限があります

**財務データ取得状況の確認:**

```bash
uv run python scripts/collect_edinet.py --check
```

出力例:
```
年度  | 書類数 | 銘柄数 | 状態
2021 |  15741 |   4262 | ✓ 取得済み
2022 |  15926 |   4281 | ✓ 取得済み

財務データ取得状況:
2021: 0/15741 (0.0%) 財務データあり    ← XBRLパースが必要
2022: 14459/15926 (90.8%) 財務データあり
```

### 3.2 TDnet（適時開示）

```bash
# TDnetデータを収集
uv run python scripts/collect_tdnet.py --days 30
```

> **注意:** TDnetは過去30日分のみ取得可能。毎日収集して履歴を蓄積してください。

### 3.3 海外指数

```bash
# S&P500、NASDAQ、VIX等を収集
uv run python scripts/collect_global_indices.py --days 365
```

### 3.4 投資部門別売買状況

```bash
# 過去1年分を収集（デフォルト）
uv run python scripts/collect_investor_trades.py

# 過去30日分を収集
uv run python scripts/collect_investor_trades.py --days 30
```

---

## 4. データベース構造

### 主要テーブル

| テーブル | 内容 | 主キー |
|----------|------|--------|
| stocks | 銘柄マスタ | code |
| prices | 日次株価 | code, date |
| financials | 四半期財務 | code, disclosed_date |
| topix | TOPIX終値 | date |
| edinet_financial | EDINET詳細財務 | code, doc_id |
| disclosure | TDnet開示 | code, date |
| global_index | 海外指数 | ticker, date |

データベースの場所: `data/jp_stock.db`

### データベース分析

データベースの状態を確認するためのスクリプト：

```bash
# 基本分析（テーブル一覧、レコード数、欠損率）
uv run python scripts/analyze_database.py

# 詳細分析（市場別・セクター別銘柄数、データ整合性）
uv run python scripts/analyze_database_v2.py

# 最終分析（年別充足率、データ品質サマリ、改善推奨事項）
uv run python scripts/analyze_database_final.py
```

---

[← ユーザーガイドに戻る](./USER_GUIDE.md) | [モデル学習ガイド →](./MODEL_TRAINING.md)
