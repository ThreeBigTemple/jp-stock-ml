# トラブルシューティング

[← ユーザーガイドに戻る](./USER_GUIDE.md)

---

## 目次

1. [よくあるエラー](#1-よくあるエラー)
2. [データ整合性チェック](#2-データ整合性チェック)
3. [ログ確認](#3-ログ確認)
4. [リセット手順](#4-リセット手順)

---

## 1. よくあるエラー

### J-Quants認証エラー

```
Error: Authentication failed
```

**対処:**
1. `.env`のメールアドレス・パスワードを確認
2. J-Quantsサイトでアカウント状態を確認
3. トークンをリフレッシュ: `rm -f data/.jquants_token`

### データベースロックエラー

```
sqlite3.OperationalError: database is locked
```

**対処:**
1. 他のプロセスがDBを使用していないか確認
2. `ps aux | grep python` で確認
3. 必要に応じてプロセスを終了

### 特徴量構築エラー

```
Error: Not enough data for feature calculation
```

**対処:**
1. データ収集期間を確認
2. 該当銘柄のデータを確認: `uv run python -c "..."`
3. 必要に応じて再収集

### メモリ不足

```
MemoryError
```

**対処:**
1. 学習期間を短縮
2. 銘柄フィルタを厳しく
3. バッチサイズを調整

---

## 2. データ整合性チェック

### Pythonで直接確認

```python
import sqlite3
conn = sqlite3.connect('data/jp_stock.db')

# 株価データ件数
print(conn.execute("SELECT COUNT(*) FROM prices").fetchone())

# 最新データ日付
print(conn.execute("SELECT MAX(date) FROM prices").fetchone())

# 銘柄数
print(conn.execute("SELECT COUNT(*) FROM stocks WHERE is_active=1").fetchone())
```

### 分析スクリプトを使用

```bash
# 基本分析
uv run python scripts/analyze_database.py

# 詳細分析
uv run python scripts/analyze_database_v2.py

# 最終分析（推奨事項付き）
uv run python scripts/analyze_database_final.py
```

---

## 3. ログ確認

```bash
# 最新ログを確認
tail -100 logs/daily_pipeline.log

# エラーのみ抽出
grep -i error logs/*.log
```

---

## 4. リセット手順

### データベースリセット

**警告:** すべてのデータが削除されます。

```bash
# DBを削除して再初期化
rm data/jp_stock.db
uv run python scripts/init_database.py
uv run python scripts/collect_historical.py --years 5
```

### モデルリセット

```bash
# 全モデルを削除して再学習
rm -rf data/models/* models/*
uv run python scripts/train_model.py
```

### 銘柄マスタの修復

pricesやfinancialsに存在するがstocksに登録されていない銘柄を追加:

```bash
# 確認
uv run python scripts/fix_stocks_master.py

# 実行
uv run python scripts/fix_stocks_master.py --fix
```

### 財務データの補完

前年同期比（YoY）データを計算して補完:

```bash
# 確認
uv run python scripts/fill_yoy_changes.py --dry-run

# 実行
uv run python scripts/fill_yoy_changes.py
```

---

[← リファレンス](./REFERENCE.md) | [ユーザーガイドに戻る →](./USER_GUIDE.md)
