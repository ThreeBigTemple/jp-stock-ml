#!/usr/bin/env python
"""
モデル学習スクリプト

ウォークフォワード検証でモデルを評価し、最終モデルを学習・保存する

使用方法:
    uv run python scripts/train_model.py --train-start 2015-01-01 --train-end 2023-12-31
"""
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd
from sqlalchemy import create_engine

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database.models import get_session, TradingCalendar
from src.features import FeatureBuilder, get_feature_columns
from src.ml import (
    LabelGenerator,
    DatasetBuilder,
    LightGBMModel,
    WalkForwardCV,
    WalkForwardValidator,
)
from src.ml.metrics import print_evaluation_report

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_trading_calendar(session) -> pd.DataFrame:
    """取引カレンダーを読み込み"""
    try:
        calendar_df = pd.read_sql(
            session.query(TradingCalendar).statement,
            session.bind
        )
        return calendar_df
    except Exception:
        logger.warning("取引カレンダーが見つかりません。デフォルトを使用します。")
        return None


def main():
    parser = argparse.ArgumentParser(description='モデル学習スクリプト')
    parser.add_argument('--train-start', type=str, default='2015-01-01',
                        help='学習開始日 (YYYY-MM-DD)')
    parser.add_argument('--train-end', type=str, default='2023-12-31',
                        help='学習終了日 (YYYY-MM-DD)')
    parser.add_argument('--output-dir', type=str, default='models/',
                        help='モデル出力ディレクトリ')
    parser.add_argument('--task', type=str, choices=['regression', 'binary', 'ranking'],
                        default='regression', help='タスクタイプ')
    parser.add_argument('--holding-days', type=int, default=20,
                        help='保有期間（営業日）')
    parser.add_argument('--top-n', type=int, default=50,
                        help='ポートフォリオ銘柄数')
    parser.add_argument('--db-path', type=str, default='data/jp_stock.db',
                        help='データベースパス')
    parser.add_argument('--skip-validation', action='store_true',
                        help='ウォークフォワード検証をスキップ')
    parser.add_argument('--params-file', type=str, default=None,
                        help='最適化済みパラメータファイル')
    args = parser.parse_args()

    # 出力ディレクトリ作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # タイムスタンプ
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # データベース接続
    logger.info(f"データベース接続: {args.db_path}")
    engine = create_engine(f"sqlite:///{args.db_path}")
    session = get_session(engine)

    try:
        # 1. 特徴量構築
        logger.info("=" * 50)
        logger.info("1. 特徴量構築")
        logger.info("=" * 50)

        builder = FeatureBuilder(session)
        feature_df = builder.build_features(
            from_date=args.train_start,
            to_date=args.train_end,
            include_technical=True,
            include_fundamental=True,
            include_market=True,
            include_edinet=True,
            include_disclosure=True,
            include_global_indices=True,
            include_trends=False,  # ノイズが多いためデフォルトオフ
        )

        if len(feature_df) == 0:
            logger.error("特徴量データがありません")
            return

        logger.info(f"特徴量データ: {len(feature_df)}行, {len(feature_df.columns)}列")

        # 2. ラベル生成
        logger.info("=" * 50)
        logger.info("2. ラベル生成")
        logger.info("=" * 50)

        # 株価データを取得
        prices_df = feature_df[['code', 'date', 'adjustment_close']].copy()

        # TOPIXデータを取得
        from src.database.models import Topix
        topix_df = pd.read_sql(session.query(Topix).statement, session.bind)

        label_gen = LabelGenerator(prices_df, topix_df)
        label_df = label_gen.forward_return(days=args.holding_days)
        label_df = label_df.rename(columns={'forward_return': 'target'})

        logger.info(f"ラベルデータ: {len(label_df)}行")

        # 3. 特徴量カラム取得
        feature_columns = get_feature_columns(feature_df)
        logger.info(f"特徴量数: {len(feature_columns)}")

        # 特徴量グループ
        dataset_builder = DatasetBuilder(feature_df, label_df['target'])
        feature_groups = dataset_builder.get_feature_groups()

        for group, cols in feature_groups.items():
            logger.info(f"  {group}: {len(cols)}特徴量")

        # 4. パラメータ読み込み
        if args.params_file and Path(args.params_file).exists():
            logger.info(f"パラメータ読み込み: {args.params_file}")
            with open(args.params_file, 'r') as f:
                params_data = json.load(f)
                model_params = params_data.get('best_params', {})
        else:
            model_params = {
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'min_child_samples': 20,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'max_depth': 6,
            }

        logger.info(f"モデルパラメータ: {model_params}")

        # 5. ウォークフォワード検証
        if not args.skip_validation:
            logger.info("=" * 50)
            logger.info("3. ウォークフォワード検証")
            logger.info("=" * 50)

            # 取引カレンダー読み込み
            trading_calendar = load_trading_calendar(session)

            # CV設定
            cv = WalkForwardCV(
                train_period_days=756,  # 約3年
                test_period_days=63,    # 約3ヶ月
                step_days=21,           # 約1ヶ月
                embargo_days=args.holding_days,  # 保有期間と同じ
            )

            # モデルファクトリ
            def model_factory():
                return LightGBMModel(params=model_params, task_type=args.task)

            # バリデーター作成・実行
            validator = WalkForwardValidator(
                cv=cv,
                model_factory=model_factory,
                feature_df=feature_df,
                label_df=label_df,
                feature_columns=feature_columns,
            )

            train_start = pd.to_datetime(args.train_start).date()
            train_end = pd.to_datetime(args.train_end).date()

            results_df = validator.run(
                start_date=train_start,
                end_date=train_end,
                trading_calendar=trading_calendar,
                verbose=True,
            )

            if len(results_df) == 0:
                logger.error("検証結果がありません")
                return

            # 全体評価
            metrics = validator.evaluate()
            print_evaluation_report(metrics, "ウォークフォワード検証結果")

            # 期間別評価
            period_metrics = validator.evaluate_by_period(freq='Q')
            logger.info("\n期間別評価（四半期）:")
            logger.info(period_metrics[['period', 'ic_mean', 'icir', 'hit_rate']].to_string())

            # 特徴量重要度
            fi = validator.get_feature_importance()
            logger.info("\nTop 20 特徴量重要度:")
            logger.info(fi.head(20).to_string())

            # グループ別重要度
            group_fi = validator.get_feature_importance_by_group(feature_groups)
            logger.info("\nグループ別特徴量重要度:")
            logger.info(group_fi.to_string())

            # 結果保存
            results_path = output_dir / f'validation_results_{timestamp}.csv'
            results_df.to_csv(results_path, index=False)
            logger.info(f"検証結果保存: {results_path}")

            metrics_path = output_dir / f'validation_metrics_{timestamp}.json'
            with open(metrics_path, 'w', encoding='utf-8') as f:
                # float32をfloatに変換
                metrics_clean = {k: float(v) if hasattr(v, 'item') else v
                                for k, v in metrics.items()}
                json.dump(metrics_clean, f, indent=2, ensure_ascii=False)
            logger.info(f"評価指標保存: {metrics_path}")

        # 6. 最終モデル学習
        logger.info("=" * 50)
        logger.info("4. 最終モデル学習")
        logger.info("=" * 50)

        # 全データでマージ
        merged_df = feature_df.merge(
            label_df[['code', 'date', 'target']],
            on=['code', 'date'],
            how='inner'
        )
        merged_df = merged_df.dropna(subset=['target'])

        # 欠損値処理
        for col in feature_columns:
            if col in merged_df.columns:
                median_val = merged_df[col].median()
                merged_df[col] = merged_df[col].fillna(median_val)
                merged_df[col] = merged_df[col].replace([float('inf'), float('-inf')], median_val)

        X = merged_df[feature_columns]
        y = merged_df['target']

        logger.info(f"学習データ: {len(X)}サンプル, {len(feature_columns)}特徴量")

        # モデル学習
        final_model = LightGBMModel(params=model_params, task_type=args.task)
        final_model.fit(
            X, y,
            num_boost_round=1000,
            early_stopping_rounds=None,  # 全データなのでEarly Stoppingなし
        )

        # モデル保存
        model_path = output_dir / f'model_{timestamp}.lgb'
        final_model.save(str(model_path))
        logger.info(f"モデル保存: {model_path}")

        # latest.lgbとしてコピー
        latest_path = output_dir / 'latest.lgb'
        final_model.save(str(latest_path))
        logger.info(f"最新モデル保存: {latest_path}")

        # 特徴量重要度保存
        fi = final_model.feature_importance()
        fi_path = output_dir / f'feature_importance_{timestamp}.csv'
        fi.to_csv(fi_path, index=False)
        logger.info(f"特徴量重要度保存: {fi_path}")

        # 学習情報保存
        train_info = {
            'train_start': args.train_start,
            'train_end': args.train_end,
            'task_type': args.task,
            'holding_days': args.holding_days,
            'top_n': args.top_n,
            'n_samples': len(X),
            'n_features': len(feature_columns),
            'params': model_params,
            'timestamp': timestamp,
        }
        info_path = output_dir / f'train_info_{timestamp}.json'
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(train_info, f, indent=2, ensure_ascii=False)
        logger.info(f"学習情報保存: {info_path}")

        logger.info("=" * 50)
        logger.info("学習完了!")
        logger.info("=" * 50)

    finally:
        session.close()


if __name__ == '__main__':
    main()
