# JP Stock ML - 日本株機械学習パイプライン

日本株の成長銘柄発掘を目的とした機械学習パイプラインです。
J-Quants APIを使用してデータを収集し、LightGBMで予測モデルを構築します。

## 機能

- **データ収集**: J-Quants APIから株価・財務・信用取引データを収集
- **特徴量エンジニアリング**: テクニカル・ファンダメンタル指標の算出
- **機械学習**: LightGBMによる成長銘柄予測
- **バックテスト**: ウォークフォワード検証

## セットアップ

### 1. 環境構築

```bash
# リポジトリをクローン（またはディレクトリを作成）
cd jp_stock_ml

# uvで依存パッケージをインストール（仮想環境も自動作成）
uv sync

# 開発用パッケージも含める場合
uv sync --extra dev
```

### 2. J-Quants API認証設定

1. [J-Quants](https://jpx-jquants.com/)でアカウントを作成
2. `.env.example`をコピーして`.env`にリネーム
3. 認証情報を設定

```bash
cp .env.example .env
# .envファイルを編集してメールアドレスとパスワードを設定
```

### 3. データベース初期化

```bash
uv run python scripts/init_database.py
```

### 4. 過去データ収集

```bash
# 過去10年分のデータを収集（数時間かかる場合があります）
uv run python scripts/collect_historical.py

# オプション
uv run python scripts/collect_historical.py --years 5  # 5年分のみ
uv run python scripts/collect_historical.py --skip-prices  # 株価をスキップ
```

### 5. 日次更新

```bash
# 直近7日分を更新
uv run python scripts/daily_update.py

# 日数指定
uv run python scripts/daily_update.py --days 14
```

## プロジェクト構造

```
jp_stock_ml/
├── config/
│   └── settings.py          # 設定・APIキー管理
├── data/
│   ├── raw/                  # 生データバックアップ
│   └── jp_stock.db           # メインDB（SQLite）
├── src/
│   ├── collectors/           # データ収集
│   │   ├── jquants_client.py    # J-Quants APIクライアント
│   │   └── jquants_collector.py # 収集ロジック
│   ├── database/             # データベース
│   │   └── models.py            # SQLAlchemyモデル
│   ├── features/             # 特徴量エンジニアリング（Phase 2）
│   └── utils/                # ユーティリティ
├── scripts/                  # 実行スクリプト
│   ├── init_database.py
│   ├── collect_historical.py
│   └── daily_update.py
├── notebooks/                # Jupyter notebooks
├── requirements.txt
└── README.md
```

## データベーススキーマ

### stocks（銘柄マスタ）
| カラム | 型 | 説明 |
|--------|-----|------|
| code | TEXT | 証券コード（PK） |
| company_name | TEXT | 会社名 |
| sector_33_code | TEXT | 33業種コード |
| market_name | TEXT | 市場区分 |
| ... | ... | ... |

### prices（株価）
| カラム | 型 | 説明 |
|--------|-----|------|
| code | TEXT | 証券コード |
| date | DATE | 日付 |
| open/high/low/close | REAL | 四本値 |
| adjustment_close | REAL | 調整済み終値 |
| volume | REAL | 出来高 |
| ... | ... | ... |

### financials（財務データ）
| カラム | 型 | 説明 |
|--------|-----|------|
| code | TEXT | 証券コード |
| disclosed_date | DATE | 開示日（Point-in-Time） |
| net_sales | REAL | 売上高 |
| operating_profit | REAL | 営業利益 |
| ... | ... | ... |

## 開発ロードマップ

- [x] Phase 1: データ基盤構築
  - [x] J-Quants APIクライアント
  - [x] データベースモデル
  - [x] データ収集スクリプト
- [ ] Phase 2: 特徴量エンジニアリング
  - [ ] テクニカル指標
  - [ ] ファンダメンタル指標
  - [ ] Point-in-Timeフィーチャーストア
- [ ] Phase 3: 機械学習パイプライン
  - [ ] LightGBMモデル
  - [ ] ウォークフォワード検証
  - [ ] Optuna最適化
- [ ] Phase 4: 運用・改善
  - [ ] 日次予測自動化
  - [ ] パフォーマンス監視

## 注意事項

- J-Quants APIの利用規約を遵守してください
- スタンダードプランでは過去10年分のデータが利用可能です
- 大量のデータ取得時はAPI呼び出し間隔に注意してください

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。
