"""
データセット作成モジュール

ラベル生成、train/test分割、欠損値処理などを担当
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import date
from typing import Optional, List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class Dataset:
    """学習用データセット"""
    X: pd.DataFrame
    y: pd.Series
    codes: pd.Series
    dates: pd.Series
    feature_names: List[str]

    def __len__(self) -> int:
        return len(self.X)

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """NumPy配列に変換"""
        return self.X.values, self.y.values


class LabelGenerator:
    """予測ターゲット生成"""

    def __init__(self, prices_df: pd.DataFrame, topix_df: Optional[pd.DataFrame] = None):
        """
        Args:
            prices_df: 株価DataFrame（code, date, adjustment_close）
            topix_df: TOPIXデータ（超過リターン計算用）
        """
        self.prices = prices_df.copy()
        self.prices['date'] = pd.to_datetime(self.prices['date'])
        self.prices = self.prices.sort_values(['code', 'date'])

        self.topix = None
        if topix_df is not None:
            self.topix = topix_df.copy()
            self.topix['date'] = pd.to_datetime(self.topix['date'])
            self.topix = self.topix.sort_values('date')

    def forward_return(self, days: int = 20) -> pd.DataFrame:
        """
        N営業日後リターン（回帰用）

        Args:
            days: 保有期間（営業日）

        Returns:
            DataFrame with columns: [code, date, forward_return]
        """
        result_list = []

        for code, group in self.prices.groupby('code'):
            group = group.sort_values('date').copy()
            future_close = group['adjustment_close'].shift(-days)
            group['forward_return'] = future_close / group['adjustment_close'] - 1
            result_list.append(group[['code', 'date', 'forward_return']])

        result = pd.concat(result_list, ignore_index=True)
        return result

    def forward_return_binary(self, days: int = 20,
                               threshold: float = 0.0) -> pd.DataFrame:
        """
        N営業日後リターンの2値分類

        Args:
            days: 保有期間
            threshold: 閾値（例: 0.05 = 5%以上で1）

        Returns:
            DataFrame with columns: [code, date, forward_return_binary]
        """
        forward_ret = self.forward_return(days)
        forward_ret['forward_return_binary'] = (
            forward_ret['forward_return'] > threshold
        ).astype(int)
        return forward_ret[['code', 'date', 'forward_return_binary']]

    def forward_return_quintile(self, days: int = 20) -> pd.DataFrame:
        """
        N営業日後リターンの5分位ラベル（ランキング用）

        各日付でクロスセクショナルに5分位に分類

        Returns:
            DataFrame with columns: [code, date, forward_return_quintile]
        """
        forward_ret = self.forward_return(days)

        def assign_quintile(group):
            try:
                group['forward_return_quintile'] = pd.qcut(
                    group['forward_return'],
                    q=5,
                    labels=[0, 1, 2, 3, 4],
                    duplicates='drop'
                ).astype(float)
            except ValueError:
                # 分位数が作れない場合（データが少ない等）
                group['forward_return_quintile'] = np.nan
            return group

        result = forward_ret.groupby('date', group_keys=False).apply(assign_quintile)
        return result[['code', 'date', 'forward_return_quintile']]

    def excess_return(self, days: int = 20,
                      benchmark: str = "topix") -> pd.DataFrame:
        """
        ベンチマーク超過リターン

        Args:
            days: 保有期間
            benchmark: "topix" or "sector"（セクターは未実装）

        Returns:
            DataFrame with columns: [code, date, excess_return]
        """
        forward_ret = self.forward_return(days)

        if benchmark == "topix" and self.topix is not None:
            # TOPIXのN日後リターンを計算
            topix = self.topix.copy()
            topix['topix_return'] = topix['close'].shift(-days) / topix['close'] - 1

            # マージ
            forward_ret = forward_ret.merge(
                topix[['date', 'topix_return']],
                on='date',
                how='left'
            )
            forward_ret['excess_return'] = (
                forward_ret['forward_return'] - forward_ret['topix_return']
            )
        else:
            # ベンチマークがない場合は生リターンを使用
            forward_ret['excess_return'] = forward_ret['forward_return']

        return forward_ret[['code', 'date', 'excess_return']]


class DatasetBuilder:
    """学習用データセット構築"""

    # 特徴量グループのプレフィックスマッピング
    FEATURE_GROUP_PREFIXES = {
        'technical': ['return_', 'momentum_', 'ma_', 'price_vs_', 'golden_cross',
                      'volatility_', 'atr_', 'volume_', 'turnover_', 'rs_'],
        'fundamental': ['revenue_growth', 'op_income', 'net_income', 'eps_growth',
                        'operating_margin', 'roe', 'roa', 'per', 'pbr', 'psr',
                        'equity_ratio', 'debt_equity', 'guidance_', 'eps_revision'],
        'edinet': ['rd_', 'capex_', 'gross_margin', 'sga_', 'interest_bearing',
                   'net_debt', 'interest_coverage', 'sales_per_', 'profit_per_',
                   'employee_', 'fcf'],
        'event': ['earnings_rev', 'buyback_', 'dividend_rev', 'stock_split',
                  'disclosure_count', 'days_since'],
        'supply_demand': ['margin_', 'short_selling'],
        'market': ['sp500', 'nasdaq', 'vix', 'us10y', 'usdjpy', 'wti', 'gold', 'risk_on'],
        'sentiment': ['search_'],
    }

    def __init__(self, feature_df: pd.DataFrame, label_series: pd.Series):
        """
        Args:
            feature_df: 特徴量DataFrame [code, date, features...]
            label_series: ラベルSeries（indexはfeature_dfと対応）
        """
        self.feature_df = feature_df.copy()
        self.feature_df['date'] = pd.to_datetime(self.feature_df['date'])

        self.label_series = label_series.copy()

        # 特徴量カラムの特定
        exclude_cols = {
            'code', 'date', 'target', 'sector_33_code', 'sector_33_name',
            'sector_17_code', 'sector_17_name', 'market_name', 'market_code',
            'company_name', 'company_name_english', 'listing_date', 'delisting_date',
            'is_active', 'open', 'high', 'low', 'close', 'volume', 'turnover_value',
            'adjustment_factor', 'adjustment_open', 'adjustment_high',
            'adjustment_low', 'adjustment_close', 'adjustment_volume',
            'id', 'updated_at', 'scale_category'
        }
        self.feature_columns = [c for c in feature_df.columns if c not in exclude_cols]
        self._filtered_stocks = None

    def create_dataset(self,
                       train_start: date,
                       train_end: date,
                       test_start: date,
                       test_end: date,
                       min_samples: int = 100) -> Tuple[Dataset, Dataset]:
        """
        学習・テストデータセット作成

        Args:
            train_start/end: 学習期間
            train_end/end: テスト期間
            min_samples: 銘柄あたり最小サンプル数

        Returns:
            (train_dataset, test_dataset)
        """
        df = self.feature_df.copy()

        # ラベルをマージ
        if isinstance(self.label_series, pd.DataFrame):
            label_col = [c for c in self.label_series.columns
                         if c not in ['code', 'date']][0]
            df = df.merge(
                self.label_series,
                on=['code', 'date'],
                how='left'
            )
            df['target'] = df[label_col]
        else:
            df['target'] = self.label_series.values

        # NaNを除去
        df = df.dropna(subset=['target'])

        # 日付変換
        train_start = pd.to_datetime(train_start)
        train_end = pd.to_datetime(train_end)
        test_start = pd.to_datetime(test_start)
        test_end = pd.to_datetime(test_end)

        # train/test分割
        train_mask = (df['date'] >= train_start) & (df['date'] <= train_end)
        test_mask = (df['date'] >= test_start) & (df['date'] <= test_end)

        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()

        # サンプル数フィルタ
        if min_samples > 0:
            stock_counts = train_df.groupby('code').size()
            valid_stocks = stock_counts[stock_counts >= min_samples].index
            train_df = train_df[train_df['code'].isin(valid_stocks)]
            test_df = test_df[test_df['code'].isin(valid_stocks)]

        logger.info(f"学習データ: {len(train_df)}行, テストデータ: {len(test_df)}行")

        # 使用する特徴量カラム（実際に存在するもののみ）
        available_features = [c for c in self.feature_columns if c in df.columns]

        train_dataset = Dataset(
            X=train_df[available_features],
            y=train_df['target'],
            codes=train_df['code'],
            dates=train_df['date'],
            feature_names=available_features
        )

        test_dataset = Dataset(
            X=test_df[available_features],
            y=test_df['target'],
            codes=test_df['code'],
            dates=test_df['date'],
            feature_names=available_features
        )

        return train_dataset, test_dataset

    def filter_stocks(self,
                      min_price: float = 100,
                      min_volume: float = 10000,
                      min_market_cap: Optional[float] = None,
                      exclude_sectors: Optional[List[str]] = None) -> 'DatasetBuilder':
        """
        銘柄フィルタリング

        Args:
            min_price: 最低株価
            min_volume: 最低出来高
            min_market_cap: 最低時価総額（未実装）
            exclude_sectors: 除外セクターリスト

        Returns:
            自身（メソッドチェーン用）
        """
        df = self.feature_df.copy()

        # 株価フィルタ
        if min_price > 0 and 'adjustment_close' in df.columns:
            df = df[df['adjustment_close'] >= min_price]

        # 出来高フィルタ
        if min_volume > 0 and 'adjustment_volume' in df.columns:
            df = df[df['adjustment_volume'] >= min_volume]

        # セクター除外
        if exclude_sectors and 'sector_33_code' in df.columns:
            df = df[~df['sector_33_code'].isin(exclude_sectors)]

        self.feature_df = df
        logger.info(f"フィルタ後データ: {len(df)}行")

        return self

    def handle_missing(self,
                       strategy: str = "fill_median",
                       max_missing_ratio: float = 0.3) -> 'DatasetBuilder':
        """
        欠損値処理

        Args:
            strategy: "fill_median", "fill_zero", "drop"
            max_missing_ratio: 欠損率がこれを超える特徴量は除外

        Returns:
            自身（メソッドチェーン用）
        """
        df = self.feature_df.copy()

        # 無限値をNaNに置換
        for col in self.feature_columns:
            if col in df.columns:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)

        # 欠損率チェック
        missing_rates = df[self.feature_columns].isnull().mean()
        high_missing_cols = missing_rates[missing_rates > max_missing_ratio].index.tolist()

        if high_missing_cols:
            logger.warning(f"欠損率{max_missing_ratio*100}%超の特徴量を除外: {len(high_missing_cols)}列")
            self.feature_columns = [c for c in self.feature_columns
                                    if c not in high_missing_cols]

        # 欠損値処理
        if strategy == "fill_median":
            for col in self.feature_columns:
                if col in df.columns:
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
        elif strategy == "fill_zero":
            for col in self.feature_columns:
                if col in df.columns:
                    df[col] = df[col].fillna(0)
        elif strategy == "drop":
            df = df.dropna(subset=self.feature_columns)

        self.feature_df = df
        return self

    def get_feature_groups(self) -> Dict[str, List[str]]:
        """特徴量をカテゴリ別にグループ化"""
        groups = {name: [] for name in self.FEATURE_GROUP_PREFIXES}
        groups['other'] = []

        for col in self.feature_columns:
            found = False
            for group_name, prefixes in self.FEATURE_GROUP_PREFIXES.items():
                for prefix in prefixes:
                    if col.startswith(prefix) or prefix in col:
                        groups[group_name].append(col)
                        found = True
                        break
                if found:
                    break
            if not found:
                groups['other'].append(col)

        # 空のグループを削除
        groups = {k: v for k, v in groups.items() if v}

        return groups

    def get_available_features(self) -> List[str]:
        """利用可能な特徴量リストを取得"""
        return [c for c in self.feature_columns if c in self.feature_df.columns]

    def describe_features(self) -> pd.DataFrame:
        """特徴量の統計情報を取得"""
        available = self.get_available_features()
        stats = self.feature_df[available].describe().T
        stats['missing_rate'] = self.feature_df[available].isnull().mean()
        return stats


def merge_labels(feature_df: pd.DataFrame, label_df: pd.DataFrame,
                 label_col: str = 'forward_return') -> pd.DataFrame:
    """
    特徴量DFとラベルDFをマージ

    Args:
        feature_df: 特徴量DataFrame
        label_df: ラベルDataFrame（code, date, label_col）
        label_col: ラベル列名

    Returns:
        マージ済みDataFrame
    """
    feature_df = feature_df.copy()
    feature_df['date'] = pd.to_datetime(feature_df['date'])

    label_df = label_df.copy()
    label_df['date'] = pd.to_datetime(label_df['date'])

    result = feature_df.merge(
        label_df[['code', 'date', label_col]],
        on=['code', 'date'],
        how='left'
    )
    result = result.rename(columns={label_col: 'target'})

    return result
