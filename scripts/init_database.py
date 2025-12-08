#!/usr/bin/env python
"""
データベース初期化スクリプト

使用方法:
    python scripts/init_database.py
"""
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import DB_PATH, DATA_DIR
from src.database import init_db


def main():
    """メイン処理"""
    print(f"データベースを初期化します: {DB_PATH}")
    
    # データディレクトリ作成
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # DB初期化
    engine = init_db(str(DB_PATH))
    
    print("データベースの初期化が完了しました")
    print(f"  場所: {DB_PATH}")
    
    # テーブル一覧表示
    from sqlalchemy import inspect
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    print(f"  テーブル数: {len(tables)}")
    for table in tables:
        print(f"    - {table}")


if __name__ == "__main__":
    main()
