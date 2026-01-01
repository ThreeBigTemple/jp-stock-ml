#!/usr/bin/env python
"""
予測実行スクリプト

学習済みモデルを使って指定日の予測を実行し、上位銘柄を出力する

使用方法:
    uv run python scripts/predict.py --date 2024-12-06 --top-n 50
"""
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys

import pandas as pd
from sqlalchemy import create_engine

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database.models import get_session, Stock
from src.features import FeatureBuilder
from src.ml import LightGBMModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='予測実行スクリプト')
    parser.add_argument('--date', type=str, required=True,
                        help='予測基準日 (YYYY-MM-DD)')
    parser.add_argument('--model-path', type=str, default='models/latest.lgb',
                        help='モデルファイルパス')
    parser.add_argument('--top-n', type=int, default=50,
                        help='出力銘柄数')
    parser.add_argument('--output', type=str, default='predictions/',
                        help='出力ディレクトリ')
    parser.add_argument('--db-path', type=str, default='data/jp_stock.db',
                        help='データベースパス')
    parser.add_argument('--include-scores', action='store_true',
                        help='予測スコアを出力に含める')
    parser.add_argument('--min-price', type=float, default=100,
                        help='最低株価フィルタ')
    parser.add_argument('--min-volume', type=float, default=10000,
                        help='最低出来高フィルタ')
    parser.add_argument('--exclude-sectors', type=str, nargs='*', default=None,
                        help='除外セクターコード（例: 7050 銀行）')
    args = parser.parse_args()

    # 出力ディレクトリ作成
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 予測日
    prediction_date = datetime.strptime(args.date, '%Y-%m-%d').date()

    # モデル読み込み
    logger.info(f"モデル読み込み: {args.model_path}")
    if not Path(args.model_path).exists():
        logger.error(f"モデルファイルが見つかりません: {args.model_path}")
        return

    model = LightGBMModel.load(args.model_path)
    logger.info(f"特徴量数: {len(model.feature_names)}")

    # データベース接続
    logger.info(f"データベース接続: {args.db_path}")
    engine = create_engine(f"sqlite:///{args.db_path}")
    session = get_session(engine)

    try:
        # 特徴量構築（予測日を含む直近期間）
        # 特徴量計算に必要なルックバック期間を確保
        lookback_start = (prediction_date - timedelta(days=365)).strftime('%Y-%m-%d')
        prediction_date_str = prediction_date.strftime('%Y-%m-%d')

        logger.info(f"特徴量構築: {lookback_start} - {prediction_date_str}")

        builder = FeatureBuilder(session)
        feature_df = builder.build_features(
            from_date=lookback_start,
            to_date=prediction_date_str,
            include_technical=True,
            include_fundamental=True,
            include_market=True,
            include_edinet=True,
            include_disclosure=True,
            include_global_indices=True,
            include_trends=False,
        )

        if len(feature_df) == 0:
            logger.error("特徴量データがありません")
            return

        # 予測日のデータのみ抽出
        feature_df['date'] = pd.to_datetime(feature_df['date'])
        target_df = feature_df[feature_df['date'] == pd.to_datetime(prediction_date)].copy()

        if len(target_df) == 0:
            # 予測日のデータがない場合、最新日を使用
            latest_date = feature_df['date'].max()
            logger.warning(f"予測日のデータがありません。最新日を使用: {latest_date.date()}")
            target_df = feature_df[feature_df['date'] == latest_date].copy()

        logger.info(f"予測対象銘柄数: {len(target_df)}")

        # フィルタリング
        if args.min_price > 0 and 'adjustment_close' in target_df.columns:
            target_df = target_df[target_df['adjustment_close'] >= args.min_price]

        if args.min_volume > 0 and 'adjustment_volume' in target_df.columns:
            target_df = target_df[target_df['adjustment_volume'] >= args.min_volume]

        if args.exclude_sectors and 'sector_33_code' in target_df.columns:
            target_df = target_df[~target_df['sector_33_code'].isin(args.exclude_sectors)]

        logger.info(f"フィルタ後銘柄数: {len(target_df)}")

        if len(target_df) == 0:
            logger.error("フィルタ後の銘柄がありません")
            return

        # 特徴量の整備
        missing_features = [f for f in model.feature_names if f not in target_df.columns]

        if missing_features:
            logger.warning(f"欠損特徴量: {len(missing_features)}個")
            # 欠損特徴量を0で埋める
            for f in missing_features:
                target_df[f] = 0

        # 欠損値処理
        X = target_df[model.feature_names].copy()
        for col in X.columns:
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val if pd.notna(median_val) else 0)
            X[col] = X[col].replace([float('inf'), float('-inf')], 0)

        # 予測実行
        logger.info("予測実行中...")
        scores = model.predict(X)
        target_df['score'] = scores

        # 上位銘柄を抽出
        top_stocks = target_df.nlargest(args.top_n, 'score')

        # 銘柄マスタから会社名を取得
        stocks_df = pd.read_sql(session.query(Stock).statement, session.bind)
        stocks_info = stocks_df[['code', 'company_name', 'sector_33_name', 'market_name']]

        top_stocks = top_stocks.merge(stocks_info, on='code', how='left')

        # 出力カラムを整理
        output_cols = ['code', 'company_name', 'sector_33_name', 'market_name']

        if args.include_scores:
            output_cols.append('score')

        # 主要な特徴量も出力
        key_features = [
            'return_1m', 'return_3m', 'per', 'pbr', 'roe',
            'revenue_growth_yoy', 'operating_margin', 'rd_ratio'
        ]
        for f in key_features:
            if f in top_stocks.columns:
                output_cols.append(f)

        # 価格情報
        if 'adjustment_close' in top_stocks.columns:
            output_cols.append('adjustment_close')

        result_df = top_stocks[output_cols].copy()
        result_df = result_df.reset_index(drop=True)
        result_df.index = result_df.index + 1  # 1から始める

        # 結果表示
        print("\n" + "=" * 80)
        print(f" 予測結果 - {prediction_date_str} - Top {args.top_n}")
        print("=" * 80)

        # 表示用にカラム名を短縮
        display_df = result_df.copy()
        display_df.columns = [
            c.replace('company_name', '会社名')
             .replace('sector_33_name', 'セクター')
             .replace('market_name', '市場')
             .replace('adjustment_close', '株価')
             .replace('score', 'スコア')
             .replace('return_1m', '1M騰落')
             .replace('return_3m', '3M騰落')
             .replace('revenue_growth_yoy', '売上成長')
             .replace('operating_margin', '営業利益率')
             .replace('rd_ratio', 'R&D比率')
            for c in display_df.columns
        ]

        print(display_df.to_string())

        # CSV出力
        output_filename = f'predictions_{prediction_date_str.replace("-", "")}.csv'
        output_path = output_dir / output_filename
        result_df.to_csv(output_path, index=True, index_label='rank', encoding='utf-8-sig')
        logger.info(f"\n予測結果保存: {output_path}")

        # サマリー出力
        print("\n" + "-" * 40)
        print(" サマリー")
        print("-" * 40)
        print(f"予測日:       {prediction_date_str}")
        print(f"対象銘柄数:   {len(target_df)}")
        print(f"出力銘柄数:   {len(result_df)}")

        if 'score' in result_df.columns:
            print(f"最高スコア:   {result_df['score'].max():.4f}")
            print(f"最低スコア:   {result_df['score'].min():.4f}")

        # セクター分布
        if 'sector_33_name' in result_df.columns:
            sector_dist = result_df['sector_33_name'].value_counts()
            print("\nセクター分布:")
            for sector, count in sector_dist.head(5).items():
                print(f"  {sector}: {count}銘柄")

        print("=" * 80)

    finally:
        session.close()


if __name__ == '__main__':
    main()
