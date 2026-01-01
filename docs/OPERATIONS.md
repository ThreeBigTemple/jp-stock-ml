# 運用ガイド

[← ユーザーガイドに戻る](./USER_GUIDE.md)

---

## 目次

1. [日次運用](#1-日次運用)
2. [月次リトレーニング](#2-月次リトレーニング)
3. [パフォーマンス監視](#3-パフォーマンス監視)

---

## 1. 日次運用

### 1.1 日次パイプライン

すべての日次タスクを一括実行します。

```bash
uv run python scripts/daily_pipeline.py
```

**実行内容:**
1. データ更新（daily_update）
2. 予測実行（predict）
3. パフォーマンス追跡（過去予測の検証）
4. ヘルスチェック
5. 通知送信（Slack/LINE）

### 1.2 スケジューリング（cron設定例）

```bash
# crontab -e で編集

# 毎営業日 18:00 に実行（東証終了後）
0 18 * * 1-5 cd /path/to/jp_stock_ml && uv run python scripts/daily_pipeline.py >> logs/daily.log 2>&1

# 毎朝 8:00 にヘルスチェックのみ
0 8 * * 1-5 cd /path/to/jp_stock_ml && uv run python scripts/check_health.py >> logs/health.log 2>&1
```

### 1.3 手動での日次運用

パイプラインを個別に実行したい場合:

```bash
# 1. データ更新
uv run python scripts/daily_update.py

# 2. 予測
uv run python scripts/predict.py --date $(date +%Y-%m-%d) --top-n 50

# 3. ヘルスチェック
uv run python scripts/check_health.py
```

### 1.4 日次レポートの確認

予測結果:
```bash
# 最新の予測を確認
cat data/predictions/daily/predictions_$(date +%Y%m%d).csv | head -20
```

パフォーマンス:
```bash
# 最新のパフォーマンスを確認
cat data/performance/daily_metrics_$(date +%Y%m%d).json
```

---

## 2. 月次リトレーニング

### 2.1 自動リトレーニング

```bash
# 基本実行
uv run python scripts/monthly_retrain.py

# バリデーション通過時に自動昇格
uv run python scripts/monthly_retrain.py --promote
```

**リトレーニングフロー:**
1. 過去5年分の特徴量を再構築
2. 新モデルを学習
3. 直近3ヶ月でバリデーション
4. 基準を満たせばプロダクション昇格

### 2.2 昇格基準

新モデルが以下を満たす場合、プロダクションに昇格:

- IC > 0.02
- シャープレシオ > 0.5
- 現行モデルより改善（または同等）

### 2.3 手動でのモデル入れ替え

```bash
# 特定モデルをプロダクションに昇格
cp data/models/candidate/model_20241201.lgb models/latest.lgb
```

### 2.4 スケジューリング（cron設定例）

```bash
# 毎月1日の深夜に実行
0 2 1 * * cd /path/to/jp_stock_ml && uv run python scripts/monthly_retrain.py --promote >> logs/retrain.log 2>&1
```

---

## 3. パフォーマンス監視

### 3.1 ヘルスチェック

```bash
uv run python scripts/check_health.py
```

**チェック項目:**
- データ鮮度（株価が3日以上古くないか）
- モデル鮮度（90日以上古くないか）
- IC推移（直近20日平均）
- アラート状況

### 3.2 週次レポート

```bash
uv run python scripts/weekly_report.py
```

**出力:** `data/performance/weekly_report_YYYYMMDD.json`

**レポート内容:**
- IC、ICIR（直近20日）
- ヒットレート
- ポートフォリオリターン（vs TOPIX）
- シャープレシオ
- 最大ドローダウン
- VaR (95%)

### 3.3 特徴量分析

```bash
uv run python scripts/analyze_features.py
```

**出力:**
- `feature_importance_YYYYMMDD.csv`: 特徴量重要度
- `feature_correlation.png`: 相関ヒートマップ
- `feature_statistics.csv`: 統計情報

### 3.4 アラートルール

`config/alert_rules.yaml`で定義:

| ルール | 条件 | レベル |
|--------|------|--------|
| IC低下 | IC < 0.01 | warning |
| IC急落 | IC < -0.02 | error |
| ドローダウン警告 | DD < -10% | warning |
| ドローダウン危険 | DD < -15% | critical |
| データ更新遅延 | > 2日 | warning |
| 特徴量欠損 | > 10% | warning |

### 3.5 通知設定

**Slack通知:**
```bash
# .envに設定
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/xxx/yyy/zzz
```

**LINE通知:**
```bash
# .envに設定
LINE_CHANNEL_ACCESS_TOKEN=xxx
LINE_USER_ID=Uxxx
```

---

[← モデル学習ガイド](./MODEL_TRAINING.md) | [リファレンス →](./REFERENCE.md)
