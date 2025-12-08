#!/usr/bin/env python
"""
日次データ更新スクリプト

定期実行（cronやタスクスケジューラ）で使用します。
直近N日分のデータを更新します。

使用方法:
    python scripts/daily_update.py [--days 7]
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import DB_PATH, JQUANTS_CONFIG, COLLECTION_CONFIG, validate_config
from src.database import init_db, get_session
from src.collectors import JQuantsClient, JQuantsCollector


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description="日次データ更新")
    parser.add_argument("--days", type=int, default=7, 
                        help="更新する日数（デフォルト: 7）- 週末を考慮して余裕を持たせる")
    args = parser.parse_args()
    
    # 設定検証
    try:
        validate_config()
    except ValueError as e:
        print(f"エラー: {e}")
        sys.exit(1)
    
    # DB接続
    engine = init_db(str(DB_PATH))
    session = get_session(engine)
    
    # APIクライアント
    client = JQuantsClient(
        mail_address=JQUANTS_CONFIG["mail_address"],
        password=JQUANTS_CONFIG["password"],
        api_interval=COLLECTION_CONFIG["api_interval"]
    )
    
    collector = JQuantsCollector(client, session)
    
    # 日付範囲
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    from_str = start_date.strftime("%Y-%m-%d")
    to_str = end_date.strftime("%Y-%m-%d")
    
    print(f"=== 日次更新開始 ({from_str} ~ {to_str}) ===\n")
    
    try:
        # 銘柄マスタ更新
        print("[1/6] 銘柄マスタ更新...")
        collector.collect_stocks()
        
        # 株価更新
        print("[2/6] 株価データ更新...")
        collector.collect_prices(from_str, to_str)
        
        # 財務データ更新（決算シーズン対応）
        print("[3/6] 財務データ更新...")
        collector.collect_financials()
        
        # TOPIX更新
        print("[4/6] TOPIXデータ更新...")
        collector.collect_topix(from_str, to_str)
        
        # 信用取引残高更新
        print("[5/6] 信用取引残高更新...")
        collector.collect_margin_balance(from_str, to_str)
        
        # 空売り比率更新
        print("[6/6] 空売り比率更新...")
        collector.collect_short_selling(from_str, to_str)
        
        print("\n=== 日次更新完了 ===")
        
    except Exception as e:
        print(f"\nエラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        session.close()


if __name__ == "__main__":
    main()
