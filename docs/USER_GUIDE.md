# JP Stock ML ユーザーガイド

日本株機械学習パイプライン - 完全操作マニュアル

---

## ドキュメント構成

| ドキュメント | 内容 |
|-------------|------|
| **[このページ](#1-システム概要)** | システム概要・クイックスタート |
| **[データ収集ガイド](./DATA_COLLECTION.md)** | 日次更新、ヒストリカル収集、EDINET/TDnet等 |
| **[モデル学習ガイド](./MODEL_TRAINING.md)** | 特徴量、モデル学習、予測実行 |
| **[運用ガイド](./OPERATIONS.md)** | 日次/月次運用、パフォーマンス監視 |
| **[リファレンス](./REFERENCE.md)** | 設定ファイル、スクリプト一覧、付録 |
| **[トラブルシューティング](./TROUBLESHOOTING.md)** | エラー対処、リセット手順 |

---

## 1. システム概要

### 1.1 このシステムでできること

JP Stock MLは、日本株市場における成長株を機械学習で予測するシステムです。

**主な機能:**
- J-Quants API等から株価・財務データを自動収集
- 100以上の特徴量を自動生成（テクニカル、ファンダメンタル、マーケット等）
- LightGBMベースの予測モデルを学習・評価
- 日次で有望銘柄トップNを予測
- パフォーマンスを継続監視し、アラートを通知

### 1.2 システム構成

```
jp_stock_ml/
├── config/           # 設定ファイル
├── data/             # データ格納（DB、モデル、予測結果）
├── docs/             # ドキュメント
├── scripts/          # 実行スクリプト
├── src/              # コアモジュール
├── logs/             # ログファイル
└── notebooks/        # 開発用Jupyter
```

### 1.3 データソース

| ソース | 取得データ | 必須 |
|--------|-----------|------|
| J-Quants API | 株価、財務、TOPIX、信用残、空売り | はい |
| EDINET | 詳細財務諸表 | 推奨 |
| TDnet | 適時開示情報 | 推奨 |
| yfinance | 海外指数（S&P500等） | いいえ |
| Google Trends | 検索トレンド | いいえ |

### 1.4 技術スタック

- Python 3.10+
- SQLite（SQLAlchemy ORM）
- LightGBM（機械学習）
- Optuna（ハイパーパラメータ最適化）
- uv（パッケージ管理）

---

## 2. クイックスタート

### 2.1 前提条件

- Python 3.10以上
- uv（パッケージマネージャ）
- J-Quants APIアカウント（無料プランで可）

### 2.2 インストール手順

```bash
# 1. リポジトリをクローン/ダウンロード
cd /path/to/jp_stock_ml

# 2. 環境変数を設定
cp .env.example .env
# .envファイルを編集してJ-Quants認証情報を入力
```

**.envファイルの内容:**
```
JQUANTS_MAIL_ADDRESS=your-email@example.com
JQUANTS_PASSWORD=your-password

# オプション: 通知用
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/xxx
LINE_CHANNEL_ACCESS_TOKEN=xxx
LINE_USER_ID=xxx
```

```bash
# 3. 依存パッケージをインストール
uv sync

# 4. データベースを初期化
uv run python scripts/init_database.py
```

### 2.3 初回データ収集

```bash
# 過去5年分のデータを収集（数時間かかります）
uv run python scripts/collect_historical.py --years 5

# または最小限（1年分）で試す
uv run python scripts/collect_historical.py --years 1
```

**収集されるデータ:**
- 株価（日次OHLCV）
- 財務諸表（四半期）
- TOPIX
- 銘柄マスタ
- 信用残・空売り比率

### 2.4 初回モデル学習

```bash
# 基本的なモデル学習
uv run python scripts/train_model.py

# カスタム期間を指定
uv run python scripts/train_model.py \
    --train-start 2020-01-01 \
    --train-end 2024-06-30
```

### 2.5 動作確認

```bash
# 予測を実行してみる
uv run python scripts/predict.py --date 2024-12-06 --top-n 10
```

成功すると、`data/predictions/daily/predictions_YYYYMMDD.csv`に予測結果が出力されます。

---

## 3. 次のステップ

セットアップが完了したら、以下のガイドを参照してください：

1. **[データ収集ガイド](./DATA_COLLECTION.md)** - 日次更新の設定、EDINET/TDnetの収集方法
2. **[モデル学習ガイド](./MODEL_TRAINING.md)** - 特徴量の詳細、学習オプション、予測の使い方
3. **[運用ガイド](./OPERATIONS.md)** - 日次/月次パイプラインの自動化、監視設定

---

## 更新履歴

- 2024-12-10: ドキュメント分割（DATA_COLLECTION, MODEL_TRAINING, OPERATIONS, REFERENCE, TROUBLESHOOTING）
- 2024-12-10: データベース分析・メンテナンス系スクリプトのドキュメント追加
- 2024-12-10: ターゲットタイプオプション追加（対TOPIX超過リターン）、EDINETのXBRLパース手順を追記
- 2024-12-09: 初版作成

---

**免責事項:** このシステムは投資アドバイスを提供するものではありません。すべての投資判断は自己責任で行ってください。
