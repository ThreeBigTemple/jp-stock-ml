#!/usr/bin/env python
"""
過去データ一括収集スクリプト

初回実行時に過去10年分のデータを収集します。
時間がかかるため、途中経過をログ出力します。

使用方法:
    python scripts/collect_historical.py [--years 10]
    python scripts/collect_historical.py --resume  # 続きから取得
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
from src.database.models import Price, Financial, MarginBalance, ShortSelling
from sqlalchemy import func


def check_existing_data(session):
    """既存データの状態をチェック"""
    status = {}

    # 株価データの最新日付
    latest_price = session.query(func.max(Price.date)).scalar()
    price_count = session.query(func.count(Price.id)).scalar()
    status['prices'] = {
        'count': price_count,
        'latest_date': latest_price
    }

    # 財務データ
    financial_count = session.query(func.count(Financial.id)).scalar()
    status['financials'] = {'count': financial_count}

    # 信用取引残高
    margin_count = session.query(func.count(MarginBalance.id)).scalar()
    latest_margin = session.query(func.max(MarginBalance.date)).scalar()
    status['margin'] = {
        'count': margin_count,
        'latest_date': latest_margin
    }

    # 空売り比率
    short_count = session.query(func.count(ShortSelling.id)).scalar()
    latest_short = session.query(func.max(ShortSelling.date)).scalar()
    status['short_selling'] = {
        'count': short_count,
        'latest_date': latest_short
    }

    return status


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description="過去データ一括収集")
    parser.add_argument("--years", type=int, default=10, help="取得する年数（デフォルト: 10）")
    parser.add_argument("--resume", action="store_true", help="続きから取得（既存データをスキップ）")
    parser.add_argument("--skip-prices", action="store_true", help="株価データをスキップ")
    parser.add_argument("--skip-financials", action="store_true", help="財務データをスキップ")
    parser.add_argument("--skip-margin", action="store_true", help="信用取引データをスキップ")
    parser.add_argument("--skip-short-selling", action="store_true", help="空売り比率をスキップ")
    args = parser.parse_args()
    
    # 設定検証
    print("設定を検証中...")
    try:
        validate_config()
    except ValueError as e:
        print(f"エラー: {e}")
        print("\n.envファイルを作成して認証情報を設定してください:")
        print("  JQUANTS_MAIL_ADDRESS=your_email@example.com")
        print("  JQUANTS_PASSWORD=your_password")
        sys.exit(1)
    
    print("設定OK")
    
    # DB初期化
    print("\nデータベースを初期化中...")
    engine = init_db(str(DB_PATH))
    session = get_session(engine)
    
    # APIクライアント初期化
    print("J-Quants APIに接続中...")
    client = JQuantsClient(
        mail_address=JQUANTS_CONFIG["mail_address"],
        password=JQUANTS_CONFIG["password"],
        api_interval=COLLECTION_CONFIG["api_interval"]
    )
    
    # コレクター初期化
    collector = JQuantsCollector(client, session)
    
    # 収集開始
    start_time = datetime.now()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.years * 365)
    from_str = start_date.strftime("%Y-%m-%d")
    to_str = end_date.strftime("%Y-%m-%d")

    # --resume モードの場合、既存データをチェック
    existing_status = None
    if args.resume:
        print("\n既存データをチェック中...")
        existing_status = check_existing_data(session)
        print(f"  株価: {existing_status['prices']['count']:,}件 (最新: {existing_status['prices']['latest_date']})")
        print(f"  財務: {existing_status['financials']['count']:,}件")
        print(f"  信用残: {existing_status['margin']['count']:,}件 (最新: {existing_status['margin']['latest_date']})")
        print(f"  空売り: {existing_status['short_selling']['count']:,}件 (最新: {existing_status['short_selling']['latest_date']})")

    print(f"\n=== 過去{args.years}年分のデータ収集を開始 ({start_time.strftime('%Y-%m-%d %H:%M:%S')}) ===\n")

    try:
        # 1. 銘柄マスタ
        print("\n[1/7] 銘柄マスタを収集...")
        collector.collect_stocks()

        # 2. 取引カレンダー
        print("\n[2/7] 取引カレンダーを収集...")
        collector.collect_trading_calendar(from_str, to_str)

        # 3. TOPIX
        print("\n[3/7] TOPIXデータを収集...")
        collector.collect_topix(from_str, to_str)

        # 4. 株価データ（年単位で分割）
        skip_prices = args.skip_prices
        price_start_date = start_date

        if args.resume and existing_status and existing_status['prices']['latest_date']:
            # 既存データがある場合、最新日付の翌日から取得
            latest = existing_status['prices']['latest_date']
            if isinstance(latest, str):
                latest = datetime.strptime(latest, "%Y-%m-%d").date()
            price_start_date = datetime.combine(latest, datetime.min.time()) + timedelta(days=1)
            if price_start_date >= end_date:
                print(f"\n[4/7] 株価データはすでに最新です（最新: {latest}）")
                skip_prices = True
            else:
                print(f"\n[4/7] 株価データを続きから取得（{price_start_date.strftime('%Y-%m-%d')}～）")

        if not skip_prices:
            if not args.resume:
                print("\n[4/7] 株価データを収集（年単位で分割）...")
            current = price_start_date
            year_count = 0
            while current < end_date:
                year_count += 1
                year_end = min(current + timedelta(days=365), end_date)
                print(f"  {year_count}年目: {current.strftime('%Y-%m-%d')} ~ {year_end.strftime('%Y-%m-%d')}")
                collector.collect_prices(
                    current.strftime("%Y-%m-%d"),
                    year_end.strftime("%Y-%m-%d")
                )
                current = year_end + timedelta(days=1)
        elif not args.resume:
            print("\n[4/7] 株価データをスキップ")

        # 5. 財務データ
        skip_financials = args.skip_financials
        if args.resume and existing_status and existing_status['financials']['count'] > 0:
            # 財務データの再開処理
            latest = session.query(func.max(Financial.disclosed_date)).scalar()
            if latest:
                if isinstance(latest, str):
                    latest = datetime.strptime(latest, "%Y-%m-%d").date()
                financial_start = datetime.combine(latest, datetime.min.time()) + timedelta(days=1)
                
                if financial_start >= end_date:
                    print(f"\n[5/7] 財務データはすでに最新です（最新: {latest}）")
                    skip_financials = True
                else:
                    print(f"\n[5/7] 財務データを続きから取得（{financial_start.strftime('%Y-%m-%d')}～）")
            else:
                financial_start = start_date
        else:
            financial_start = start_date

        if not skip_financials:
            print(f"\n[5/7] 財務データを収集（{financial_start.strftime('%Y-%m-%d')} ~ {to_str}）...")
            # 年単位で分割して収集（進捗保存のため）
            current = financial_start
            year_count = 0
            while current < end_date:
                year_count += 1
                year_end = min(current + timedelta(days=365), end_date)
                print(f"  {year_count}年目: {current.strftime('%Y-%m-%d')} ~ {year_end.strftime('%Y-%m-%d')}")
                collector.collect_financials(
                    from_date=current.strftime("%Y-%m-%d"),
                    to_date=year_end.strftime("%Y-%m-%d")
                )
                current = year_end + timedelta(days=1)
        elif not args.resume:
            print("\n[5/7] 財務データをスキップ")

        # 6. 信用取引残高
        skip_margin = args.skip_margin
        margin_start = from_str

        if args.resume and existing_status and existing_status['margin']['latest_date']:
            latest = existing_status['margin']['latest_date']
            if isinstance(latest, str):
                latest_dt = datetime.strptime(latest, "%Y-%m-%d")
            else:
                latest_dt = datetime.combine(latest, datetime.min.time())
            margin_start_dt = latest_dt + timedelta(days=1)
            if margin_start_dt >= end_date:
                print(f"\n[6/7] 信用取引残高はすでに最新です（最新: {latest}）")
                skip_margin = True
            else:
                margin_start = margin_start_dt.strftime("%Y-%m-%d")
                print(f"\n[6/7] 信用取引残高を続きから取得（{margin_start}～）")

        if not skip_margin:
            if not args.resume or not existing_status or not existing_status['margin']['latest_date']:
                print("\n[6/7] 信用取引残高を収集...")
            collector.collect_margin_balance(margin_start, to_str)
        elif not args.resume:
            print("\n[6/7] 信用取引残高をスキップ")

        # 7. 空売り比率
        skip_short = args.skip_short_selling
        short_start = from_str

        if args.resume and existing_status and existing_status['short_selling']['latest_date']:
            latest = existing_status['short_selling']['latest_date']
            if isinstance(latest, str):
                latest_dt = datetime.strptime(latest, "%Y-%m-%d")
            else:
                latest_dt = datetime.combine(latest, datetime.min.time())
            short_start_dt = latest_dt + timedelta(days=1)
            if short_start_dt >= end_date:
                print(f"\n[7/7] 空売り比率はすでに最新です（最新: {latest}）")
                skip_short = True
            else:
                short_start = short_start_dt.strftime("%Y-%m-%d")
                print(f"\n[7/7] 空売り比率を続きから取得（{short_start}～）")

        if not skip_short:
            if not args.resume or not existing_status or not existing_status['short_selling']['latest_date']:
                print("\n[7/7] 空売り比率を収集...")
            collector.collect_short_selling(short_start, to_str)
        
        end_time = datetime.now()
        elapsed = end_time - start_time
        
        print(f"\n=== データ収集完了 ===")
        print(f"  開始: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  終了: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  所要時間: {elapsed}")
        
    except KeyboardInterrupt:
        print("\n\n収集が中断されました。")
        print("途中まで収集したデータはDBに保存されています。")
        print("再実行すると続きから収集されます（重複データは上書き）。")
        sys.exit(1)
    
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        session.close()


if __name__ == "__main__":
    main()
