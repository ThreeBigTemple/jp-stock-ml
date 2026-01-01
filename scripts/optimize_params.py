#!/usr/bin/env python
"""
ハイパーパラメータ最適化スクリプト

Optunaを使ってLightGBMのハイパーパラメータを最適化する

使用方法:
    uv run python scripts/optimize_params.py --n-trials 100 --timeout 7200 --metric icir
"""
import argparse
import logging
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd
from sqlalchemy import create_engine

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database.models import get_session
from src.features import FeatureBuilder, get_feature_columns
from src.ml import LabelGenerator
from src.ml.optimizer import HyperparameterOptimizer, FeatureSelector
from src.ml.walk_forward import WalkForwardCV

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='ハイパーパラメータ最適化スクリプト')
    parser.add_argument('--n-trials', type=int, default=100,
                        help='Optuna試行回数')
    parser.add_argument('--timeout', type=int, default=7200,
                        help='タイムアウト（秒）')
    parser.add_argument('--metric', type=str, default='icir',
                        choices=['ic', 'icir', 'sharpe', 'hit_rate', 'portfolio_sharpe'],
                        help='最適化指標')
    parser.add_argument('--from-date', type=str, default='2018-01-01',
                        help='最適化データ開始日')
    parser.add_argument('--to-date', type=str, default='2023-12-31',
                        help='最適化データ終了日')
    parser.add_argument('--holding-days', type=int, default=20,
                        help='保有期間（営業日）')
    parser.add_argument('--output', type=str, default='models/',
                        help='出力ディレクトリ')
    parser.add_argument('--db-path', type=str, default='data/jp_stock.db',
                        help='データベースパス')
    parser.add_argument('--feature-selection', action='store_true',
                        help='特徴量グループ選択も最適化する')
    parser.add_argument('--study-name', type=str, default=None,
                        help='Optuna study名（再開用）')
    parser.add_argument('--storage', type=str, default=None,
                        help='Optuna storage URL')
    args = parser.parse_args()

    # 出力ディレクトリ作成
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

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
            from_date=args.from_date,
            to_date=args.to_date,
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

        logger.info(f"特徴量データ: {len(feature_df)}行, {len(feature_df.columns)}列")

        # 2. ラベル生成
        logger.info("=" * 50)
        logger.info("2. ラベル生成")
        logger.info("=" * 50)

        prices_df = feature_df[['code', 'date', 'adjustment_close']].copy()

        from src.database.models import Topix
        topix_df = pd.read_sql(session.query(Topix).statement, session.bind)

        label_gen = LabelGenerator(prices_df, topix_df)
        label_df = label_gen.forward_return(days=args.holding_days)
        label_df = label_df.rename(columns={'forward_return': 'target'})

        logger.info(f"ラベルデータ: {len(label_df)}行")

        # 3. 特徴量カラム取得
        feature_columns = get_feature_columns(feature_df)
        logger.info(f"特徴量数: {len(feature_columns)}")

        # 4. CV設定（最適化用に短い期間）
        cv = WalkForwardCV(
            train_period_days=504,  # 約2年
            test_period_days=63,    # 約3ヶ月
            step_days=63,           # 約3ヶ月
            embargo_days=args.holding_days,
        )

        # 5. ハイパーパラメータ最適化
        logger.info("=" * 50)
        logger.info("3. ハイパーパラメータ最適化")
        logger.info("=" * 50)

        study_name = args.study_name or f'lgb_optimization_{timestamp}'

        optimizer = HyperparameterOptimizer(
            feature_df=feature_df,
            label_df=label_df,
            feature_columns=feature_columns,
            cv=cv,
            n_trials=args.n_trials,
            timeout=args.timeout,
            metric=args.metric,
            task_type='regression',
        )

        best_params = optimizer.optimize(
            study_name=study_name,
            storage=args.storage,
        )

        # 結果保存
        params_path = output_dir / f'optimized_params_{timestamp}.json'
        optimizer.save_results(str(params_path))

        # パラメータ重要度
        param_importance = optimizer.get_param_importance()
        print("\nパラメータ重要度:")
        print(param_importance.to_string())

        param_importance_path = output_dir / f'param_importance_{timestamp}.csv'
        param_importance.to_csv(param_importance_path, index=False)

        # 最適化履歴
        history = optimizer.get_optimization_history()
        history_path = output_dir / f'optimization_history_{timestamp}.csv'
        history.to_csv(history_path, index=False)

        # 6. 特徴量選択最適化（オプション）
        if args.feature_selection:
            logger.info("=" * 50)
            logger.info("4. 特徴量グループ選択最適化")
            logger.info("=" * 50)

            # 特徴量グループ定義
            from src.ml.dataset import DatasetBuilder
            ds_builder = DatasetBuilder(feature_df, label_df['target'])
            feature_groups = ds_builder.get_feature_groups()

            for group, cols in feature_groups.items():
                logger.info(f"  {group}: {len(cols)}特徴量")

            feature_selector = FeatureSelector(
                feature_df=feature_df,
                label_df=label_df,
                feature_groups=feature_groups,
                cv=cv,
                base_params=best_params,
                n_trials=50,
                metric=args.metric,
            )

            best_features = feature_selector.optimize()

            # グループ選択頻度
            group_importance = feature_selector.get_group_importance()
            print("\nグループ選択頻度:")
            print(group_importance.to_string())

            group_importance_path = output_dir / f'feature_group_selection_{timestamp}.csv'
            group_importance.to_csv(group_importance_path, index=False)

            logger.info(f"選択された特徴量数: {len(best_features)}")

        # サマリー表示
        print("\n" + "=" * 80)
        print(" 最適化完了サマリー")
        print("=" * 80)
        print(f"\n試行回数: {args.n_trials}")
        print(f"最適化指標: {args.metric}")
        print(f"最良スコア: {optimizer.study.best_value:.4f}")
        print("\n最適パラメータ:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")
        print(f"\n結果保存先: {params_path}")
        print("=" * 80)

    finally:
        session.close()


if __name__ == '__main__':
    main()
