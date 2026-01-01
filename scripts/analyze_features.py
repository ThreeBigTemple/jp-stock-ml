#!/usr/bin/env python
"""
特徴量分析スクリプト

学習済みモデルの特徴量重要度を分析し、レポートを出力する

使用方法:
    uv run python scripts/analyze_features.py --model-path models/latest.lgb
"""
import argparse
import json
import logging
from pathlib import Path
import sys

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database.models import get_session
from src.features import FeatureBuilder
from src.ml import LightGBMModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_feature_statistics(feature_df: pd.DataFrame,
                                  feature_columns: list) -> pd.DataFrame:
    """特徴量の統計情報を計算"""
    stats_list = []

    for col in feature_columns:
        if col not in feature_df.columns:
            continue

        values = feature_df[col].dropna()

        stats_list.append({
            'feature': col,
            'count': len(values),
            'missing_rate': 1 - len(values) / len(feature_df),
            'mean': values.mean(),
            'std': values.std(),
            'min': values.min(),
            'p25': values.quantile(0.25),
            'median': values.median(),
            'p75': values.quantile(0.75),
            'max': values.max(),
            'skew': values.skew() if len(values) > 2 else np.nan,
            'kurtosis': values.kurtosis() if len(values) > 3 else np.nan,
        })

    return pd.DataFrame(stats_list)


def calculate_feature_correlation(feature_df: pd.DataFrame,
                                   feature_columns: list,
                                   top_n: int = 50) -> pd.DataFrame:
    """特徴量間の相関を計算"""
    # 上位特徴量のみで計算（計算量削減）
    available = [c for c in feature_columns[:top_n] if c in feature_df.columns]

    corr_matrix = feature_df[available].corr()

    # 高相関ペアを抽出
    high_corr_pairs = []
    for i, col1 in enumerate(available):
        for j, col2 in enumerate(available):
            if i < j:
                corr_val = corr_matrix.loc[col1, col2]
                if abs(corr_val) > 0.7:  # 相関0.7以上
                    high_corr_pairs.append({
                        'feature_1': col1,
                        'feature_2': col2,
                        'correlation': corr_val
                    })

    return pd.DataFrame(high_corr_pairs).sort_values('correlation', ascending=False)


def main():
    parser = argparse.ArgumentParser(description='特徴量分析スクリプト')
    parser.add_argument('--model-path', type=str, required=True,
                        help='モデルファイルパス')
    parser.add_argument('--output', type=str, default='reports/',
                        help='出力ディレクトリ')
    parser.add_argument('--db-path', type=str, default='data/jp_stock.db',
                        help='データベースパス')
    parser.add_argument('--analyze-data', action='store_true',
                        help='データ統計も分析する')
    parser.add_argument('--from-date', type=str, default='2020-01-01',
                        help='データ分析開始日')
    parser.add_argument('--to-date', type=str, default='2023-12-31',
                        help='データ分析終了日')
    args = parser.parse_args()

    # 出力ディレクトリ作成
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # モデル読み込み
    logger.info(f"モデル読み込み: {args.model_path}")
    if not Path(args.model_path).exists():
        logger.error(f"モデルファイルが見つかりません: {args.model_path}")
        return

    model = LightGBMModel.load(args.model_path)

    print("\n" + "=" * 80)
    print(" 特徴量分析レポート")
    print("=" * 80)

    # 1. 特徴量重要度（Gain）
    print("\n" + "-" * 40)
    print(" 1. 特徴量重要度 (Gain)")
    print("-" * 40)

    fi_gain = model.feature_importance('gain')
    fi_gain['importance_pct'] = fi_gain['importance'] / fi_gain['importance'].sum() * 100

    print("\nTop 30 特徴量:")
    print(fi_gain.head(30).to_string())

    # 2. 特徴量重要度（Split）
    print("\n" + "-" * 40)
    print(" 2. 特徴量重要度 (Split)")
    print("-" * 40)

    fi_split = model.feature_importance('split')
    fi_split['importance_pct'] = fi_split['importance'] / fi_split['importance'].sum() * 100

    print("\nTop 30 特徴量:")
    print(fi_split.head(30).to_string())

    # 3. グループ別重要度
    print("\n" + "-" * 40)
    print(" 3. グループ別特徴量重要度")
    print("-" * 40)

    # 特徴量グループを定義
    feature_groups = {
        'technical': [f for f in model.feature_names
                      if any(f.startswith(p) for p in ['return_', 'momentum_', 'ma_', 'price_vs_',
                                                       'golden_cross', 'volatility_', 'atr_',
                                                       'volume_', 'turnover_', 'rs_'])],
        'fundamental': [f for f in model.feature_names
                        if any(f.startswith(p) for p in ['revenue_growth', 'op_income', 'net_income',
                                                         'eps_growth', 'operating_margin', 'roe',
                                                         'roa', 'per', 'pbr', 'psr', 'equity_ratio',
                                                         'debt_equity', 'guidance_', 'eps_revision'])],
        'edinet': [f for f in model.feature_names
                   if any(f.startswith(p) for p in ['rd_', 'capex_', 'gross_margin', 'sga_',
                                                    'interest_bearing', 'net_debt', 'interest_coverage',
                                                    'sales_per_', 'profit_per_', 'employee_', 'fcf'])],
        'event': [f for f in model.feature_names
                  if any(f.startswith(p) for p in ['earnings_rev', 'buyback_', 'dividend_rev',
                                                   'stock_split', 'disclosure_count', 'days_since'])],
        'supply_demand': [f for f in model.feature_names
                          if any(f.startswith(p) for p in ['margin_', 'short_selling'])],
        'market': [f for f in model.feature_names
                   if any(f.startswith(p) for p in ['sp500', 'nasdaq', 'vix', 'us10y',
                                                    'usdjpy', 'wti', 'gold', 'risk_on'])],
        'sentiment': [f for f in model.feature_names
                      if f.startswith('search_')],
    }

    # 「その他」グループ
    categorized = set()
    for features in feature_groups.values():
        categorized.update(features)
    feature_groups['other'] = [f for f in model.feature_names if f not in categorized]

    group_fi = model.feature_importance_by_group(feature_groups, 'gain')
    print("\nグループ別重要度 (Gain):")
    print(group_fi.to_string())

    # 4. 重要度分布
    print("\n" + "-" * 40)
    print(" 4. 特徴量重要度分布")
    print("-" * 40)

    print(f"\n総特徴量数: {len(model.feature_names)}")
    print(f"重要度 > 1%: {(fi_gain['importance_pct'] > 1).sum()}個")
    print(f"重要度 > 0.5%: {(fi_gain['importance_pct'] > 0.5).sum()}個")
    print(f"重要度 = 0: {(fi_gain['importance'] == 0).sum()}個")

    # Top 10の累積重要度
    top10_pct = fi_gain.head(10)['importance_pct'].sum()
    top20_pct = fi_gain.head(20)['importance_pct'].sum()
    top50_pct = fi_gain.head(50)['importance_pct'].sum()

    print(f"\nTop 10 累積重要度: {top10_pct:.1f}%")
    print(f"Top 20 累積重要度: {top20_pct:.1f}%")
    print(f"Top 50 累積重要度: {top50_pct:.1f}%")

    # 5. データ統計分析（オプション）
    if args.analyze_data:
        print("\n" + "-" * 40)
        print(" 5. データ統計分析")
        print("-" * 40)

        # データベース接続
        engine = create_engine(f"sqlite:///{args.db_path}")
        session = get_session(engine)

        try:
            logger.info("特徴量データ読み込み中...")
            builder = FeatureBuilder(session)
            feature_df = builder.build_features(
                from_date=args.from_date,
                to_date=args.to_date,
            )

            if len(feature_df) > 0:
                # 統計情報
                stats_df = calculate_feature_statistics(feature_df, model.feature_names)
                print("\n特徴量統計（上位20）:")
                print(stats_df[['feature', 'missing_rate', 'mean', 'std', 'median']].head(20).to_string())

                # 高欠損特徴量
                high_missing = stats_df[stats_df['missing_rate'] > 0.3]
                if len(high_missing) > 0:
                    print(f"\n欠損率30%超の特徴量: {len(high_missing)}個")
                    print(high_missing[['feature', 'missing_rate']].to_string())

                # 相関分析
                top_features = fi_gain.head(50)['feature'].tolist()
                corr_df = calculate_feature_correlation(feature_df, top_features)
                if len(corr_df) > 0:
                    print("\n高相関ペア（|r| > 0.7）:")
                    print(corr_df.head(20).to_string())

                # 統計情報を保存
                stats_path = output_dir / 'feature_statistics.csv'
                stats_df.to_csv(stats_path, index=False)
                logger.info(f"特徴量統計保存: {stats_path}")

        finally:
            session.close()

    # レポート保存
    print("\n" + "-" * 40)
    print(" レポート保存")
    print("-" * 40)

    # 重要度（Gain）保存
    fi_gain_path = output_dir / 'feature_importance_gain.csv'
    fi_gain.to_csv(fi_gain_path, index=False)
    logger.info(f"特徴量重要度(Gain)保存: {fi_gain_path}")

    # 重要度（Split）保存
    fi_split_path = output_dir / 'feature_importance_split.csv'
    fi_split.to_csv(fi_split_path, index=False)
    logger.info(f"特徴量重要度(Split)保存: {fi_split_path}")

    # グループ別重要度保存
    group_fi_path = output_dir / 'feature_importance_by_group.csv'
    group_fi.to_csv(group_fi_path, index=False)
    logger.info(f"グループ別重要度保存: {group_fi_path}")

    # サマリーJSON
    summary = {
        'total_features': len(model.feature_names),
        'important_features_1pct': int((fi_gain['importance_pct'] > 1).sum()),
        'zero_importance_features': int((fi_gain['importance'] == 0).sum()),
        'top10_cumulative_importance': float(top10_pct),
        'top20_cumulative_importance': float(top20_pct),
        'top_feature': fi_gain.iloc[0]['feature'],
        'top_feature_importance': float(fi_gain.iloc[0]['importance_pct']),
        'group_ranking': group_fi['group'].tolist(),
    }

    summary_path = output_dir / 'analysis_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"サマリー保存: {summary_path}")

    print("\n" + "=" * 80)
    print(" 分析完了!")
    print("=" * 80)


if __name__ == '__main__':
    main()
