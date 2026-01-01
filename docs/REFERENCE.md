# リファレンス

[← ユーザーガイドに戻る](./USER_GUIDE.md)

---

## 目次

1. [設定ファイル](#1-設定ファイル)
2. [スクリプト一覧](#2-スクリプト一覧)
3. [付録](#3-付録)

---

## 1. 設定ファイル

### 1.1 config/settings.py

```python
# API認証（.envから読み込み）
JQUANTS_MAIL_ADDRESS = os.getenv("JQUANTS_MAIL_ADDRESS")
JQUANTS_PASSWORD = os.getenv("JQUANTS_PASSWORD")

# データベース
DATABASE_PATH = "data/jp_stock.db"

# API設定
API_INTERVAL = 0.5  # リクエスト間隔（秒）
MAX_RETRIES = 3

# 対象市場
TARGET_MARKETS = ["プライム", "スタンダード", "グロース"]
```

### 1.2 config/operation_config.yaml

```yaml
prediction:
  portfolio_size: 50            # 予測出力銘柄数
  min_price: 100                # 最低株価
  min_volume: 10000             # 最低出来高
  exclude_sectors:              # 除外セクター
    - 銀行業
    - 保険業
    - 証券、商品先物取引業
  include_markets:              # 対象市場
    - プライム
    - スタンダード
    - グロース

rebalance:
  frequency: weekly             # リバランス頻度
  min_holding_days: 20          # 最低保有日数
  max_turnover: 0.3             # 最大回転率

retrain:
  schedule: monthly             # リトレーニング頻度
  train_years: 5                # 学習期間（年）
  optimize_params: false        # パラメータ最適化
  num_boost_round: 1000         # ブースティング回数

validation:
  min_ic_threshold: 0.02        # IC閾値
  min_sharpe_threshold: 0.5     # シャープ閾値
  min_hit_rate: 0.52            # ヒットレート閾値
```

### 1.3 config/alert_rules.yaml

```yaml
rules:
  - name: ic_low
    condition: ic < 0.01
    level: warning
    message: "IC低下: {ic:.4f}"

  - name: ic_critical
    condition: ic < -0.02
    level: error
    message: "IC急落: {ic:.4f}"

  - name: drawdown_warning
    condition: drawdown < -0.10
    level: warning
    message: "ドローダウン警告: {drawdown:.1%}"

  - name: drawdown_critical
    condition: drawdown < -0.15
    level: critical
    message: "ドローダウン危険: {drawdown:.1%}"

notifications:
  slack:
    enabled: true
    levels: [warning, error, critical]

  line:
    enabled: true
    levels: [critical]

  email:
    enabled: false
```

---

## 2. スクリプト一覧

### 2.1 データ収集

| スクリプト | 説明 | 主なオプション |
|-----------|------|---------------|
| `init_database.py` | DB初期化 | なし |
| `collect_historical.py` | 過去データ収集 | `--years`, `--start-date`, `--end-date` |
| `daily_update.py` | 日次更新 | `--days`, `--skip-edinet`, `--skip-tdnet` |
| `collect_edinet.py` | EDINET収集 | `--years`, `--year`, `--check`, `--force`, `--parse-xbrl`, `--workers`, `--api-key` |
| `collect_tdnet.py` | TDnet収集 | `--days` |
| `collect_global_indices.py` | 海外指数収集 | `--days` |
| `collect_investor_trades.py` | 投資部門別売買状況収集 | `--days` |

### 2.2 モデル関連

| スクリプト | 説明 | 主なオプション |
|-----------|------|---------------|
| `train_model.py` | モデル学習 | `--train-start`, `--train-end`, `--skip-validation`, `--task`, `--target-type`, `--holding-days` |
| `predict.py` | 予測実行 | `--date`, `--top-n`, `--model-path`, `--min-price`, `--min-volume` |
| `optimize_params.py` | パラメータ最適化 | `--n-trials`, `--metric`, `--optimize-features` |
| `analyze_features.py` | 特徴量分析 | `--output-stats` |

### 2.3 運用

| スクリプト | 説明 | 主なオプション |
|-----------|------|---------------|
| `daily_pipeline.py` | 日次パイプライン | `--skip-update`, `--skip-predict`, `--skip-notify` |
| `monthly_retrain.py` | 月次リトレーニング | `--promote`, `--optimize` |
| `check_health.py` | ヘルスチェック | `--verbose` |
| `weekly_report.py` | 週次レポート | `--output-path` |

### 2.4 データベース分析・メンテナンス

| スクリプト | 説明 | 主なオプション |
|-----------|------|---------------|
| `analyze_database.py` | DB基本分析（テーブル数、レコード数、欠損状況） | なし |
| `analyze_database_v2.py` | DB詳細分析（市場別・セクター別集計、整合性チェック） | なし |
| `analyze_database_final.py` | DB最終分析（年別充足率、品質サマリ、改善推奨） | なし |
| `fill_yoy_changes.py` | financialsの前年同期比を計算・補完 | `--dry-run` |
| `fix_stocks_master.py` | 銘柄マスタ不整合修正 | `--fix` |
| `init_operation_tables.py` | 運用系テーブル確認・サンプルデータ投入 | `--insert-samples` |

#### fill_yoy_changes.py（前年同期比計算）

financialsテーブルの`change_net_sales`、`change_operating_profit`等のカラムがNULLの場合、前年同期データから自動計算して補完します。

```bash
# 確認のみ（更新しない）
uv run python scripts/fill_yoy_changes.py --dry-run

# 実行（データベースを更新）
uv run python scripts/fill_yoy_changes.py
```

#### fix_stocks_master.py（銘柄マスタ修正）

pricesやfinancialsに存在するが、stocksマスタに登録されていない銘柄（上場廃止銘柄など）をJ-Quants APIから情報取得して追加します。

```bash
# 確認のみ（不整合銘柄数を表示）
uv run python scripts/fix_stocks_master.py

# 実行（銘柄マスタを更新）
uv run python scripts/fix_stocks_master.py --fix
```

#### init_operation_tables.py（運用テーブル初期化）

predictions、alert_history等の運用系テーブルの状況確認とサンプルデータ投入を行います。

```bash
# 現在のテーブル状況を確認
uv run python scripts/init_operation_tables.py

# サンプルデータを投入（動作確認用）
uv run python scripts/init_operation_tables.py --insert-samples
```

---

## 3. 付録

### A. パフォーマンス目安

| 処理 | 所要時間目安 |
|------|-------------|
| 日次更新 | 5〜10分 |
| 予測実行 | 30秒〜1分 |
| モデル学習（バリデーション付き） | 2〜4時間 |
| パラメータ最適化（100試行） | 12〜24時間 |
| ヒストリカル収集（5年分） | 4〜8時間 |

### B. ディスク使用量目安

| 項目 | サイズ |
|------|--------|
| データベース（5年分） | 1.5〜2GB |
| モデルファイル（1つ） | 5〜15MB |
| 予測結果（1日分） | 10〜50KB |
| ログ（1ヶ月分） | 10〜50MB |

### C. 推奨環境

- CPU: 4コア以上
- メモリ: 16GB以上
- ストレージ: SSD 20GB以上
- OS: macOS / Linux（Windowsも可）

---

[← 運用ガイド](./OPERATIONS.md) | [トラブルシューティング →](./TROUBLESHOOTING.md)
