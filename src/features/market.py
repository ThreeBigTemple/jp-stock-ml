"""
市場・需給指標算出モジュール

信用取引、空売り比率などの市場需給データから指標を算出する
海外指数（S&P500、VIX等）も含む
"""
import numpy as np
import pandas as pd
from typing import Optional
import logging

from .global_index_features import GlobalIndexFeatures, merge_global_features, calculate_market_regime

logger = logging.getLogger(__name__)


class MarketFeatures:
    """市場・需給指標算出クラス"""

    def __init__(
        self,
        margin_df: Optional[pd.DataFrame] = None,
        short_selling_df: Optional[pd.DataFrame] = None,
        topix_df: Optional[pd.DataFrame] = None,
        global_indices_df: Optional[pd.DataFrame] = None
    ):
        """
        初期化

        Args:
            margin_df: 信用取引残高データ（columns: code, date, margin_buying_balance,
                       margin_selling_balance, margin_buying_value, margin_selling_value）
            short_selling_df: 空売り比率データ（columns: date, sector_33_code,
                              selling_value, short_selling_with_restrictions,
                              short_selling_without_restrictions）
            topix_df: TOPIXデータ（columns: date, open, high, low, close）
            global_indices_df: 海外指数データ（columns: index_name, date, open, high, low, close, volume）
        """
        self.margin = margin_df.copy() if margin_df is not None else None
        self.short_selling = short_selling_df.copy() if short_selling_df is not None else None
        self.topix = topix_df.copy() if topix_df is not None else None
        self.global_indices = global_indices_df.copy() if global_indices_df is not None else None

        # 日付でソート
        if self.margin is not None:
            self.margin = self.margin.sort_values(['code', 'date'])
        if self.short_selling is not None:
            self.short_selling = self.short_selling.sort_values(['date', 'sector_33_code'])
        if self.topix is not None:
            self.topix = self.topix.sort_values('date')
        if self.global_indices is not None:
            self.global_indices = self.global_indices.sort_values(['index_name', 'date'])

    def calculate_margin_features(self) -> pd.DataFrame:
        """
        信用取引関連指標を算出

        Returns:
            信用取引特徴量DataFrame（columns: code, date, features...）
        """
        if self.margin is None or len(self.margin) == 0:
            logger.warning("信用取引データがありません")
            return pd.DataFrame()

        df = self.margin.copy()

        # 銘柄ごとに計算
        features_list = []

        for code, group in df.groupby('code'):
            group = group.sort_values('date').copy()

            # 信用倍率（買残 / 売残）
            group['margin_balance_ratio'] = np.where(
                group['margin_selling_balance'] > 0,
                group['margin_buying_balance'] / group['margin_selling_balance'],
                np.nan
            )

            # 信用倍率（金額ベース）
            group['margin_value_ratio'] = np.where(
                group['margin_selling_value'] > 0,
                group['margin_buying_value'] / group['margin_selling_value'],
                np.nan
            )

            # 買残変化率（週次データなので1週間前との比較）
            group['margin_buying_change'] = group['margin_buying_balance'].pct_change(1)

            # 売残変化率
            group['margin_selling_change'] = group['margin_selling_balance'].pct_change(1)

            # 買残 - 売残（ネットポジション）の変化
            group['net_margin_balance'] = (
                group['margin_buying_balance'] - group['margin_selling_balance']
            )
            group['net_margin_change'] = group['net_margin_balance'].pct_change(1)

            # 信用倍率の4週移動平均
            group['margin_ratio_ma4w'] = group['margin_balance_ratio'].rolling(4, min_periods=2).mean()

            # 信用倍率の変化（現在 vs 4週平均）
            group['margin_ratio_vs_ma4w'] = (
                group['margin_balance_ratio'] / group['margin_ratio_ma4w'] - 1
            )

            features_list.append(group)

        result = pd.concat(features_list, ignore_index=True)

        # 必要なカラムのみ選択
        feature_cols = [
            'code', 'date',
            'margin_balance_ratio', 'margin_value_ratio',
            'margin_buying_change', 'margin_selling_change',
            'net_margin_balance', 'net_margin_change',
            'margin_ratio_ma4w', 'margin_ratio_vs_ma4w'
        ]
        result = result[[c for c in feature_cols if c in result.columns]]

        logger.info(f"信用取引指標算出完了: {len(result)}行")
        return result

    def calculate_short_selling_features(self) -> pd.DataFrame:
        """
        空売り比率関連指標を算出

        Returns:
            空売り特徴量DataFrame（columns: date, sector_33_code, features...）
        """
        if self.short_selling is None or len(self.short_selling) == 0:
            logger.warning("空売りデータがありません")
            return pd.DataFrame()

        df = self.short_selling.copy()

        # セクターごとに計算
        features_list = []

        for sector, group in df.groupby('sector_33_code'):
            group = group.sort_values('date').copy()

            # 空売り比率（規制あり + 規制なし）/ 売り金額全体
            total_short = (
                group['short_selling_with_restrictions'].fillna(0) +
                group['short_selling_without_restrictions'].fillna(0)
            )
            group['short_selling_ratio'] = np.where(
                group['selling_value'] > 0,
                total_short / group['selling_value'],
                np.nan
            )

            # 規制なし空売り比率
            group['short_selling_unrestricted_ratio'] = np.where(
                group['selling_value'] > 0,
                group['short_selling_without_restrictions'] / group['selling_value'],
                np.nan
            )

            # 空売り比率の変化
            group['short_ratio_change_5d'] = group['short_selling_ratio'].pct_change(5)

            # 空売り比率の20日移動平均
            group['short_ratio_ma20d'] = group['short_selling_ratio'].rolling(20, min_periods=10).mean()

            # 現在 vs 移動平均
            group['short_ratio_vs_ma20d'] = (
                group['short_selling_ratio'] / group['short_ratio_ma20d'] - 1
            )

            features_list.append(group)

        result = pd.concat(features_list, ignore_index=True)

        # 必要なカラムのみ選択
        feature_cols = [
            'date', 'sector_33_code', 'sector_33_name',
            'short_selling_ratio', 'short_selling_unrestricted_ratio',
            'short_ratio_change_5d', 'short_ratio_ma20d', 'short_ratio_vs_ma20d'
        ]
        result = result[[c for c in feature_cols if c in result.columns]]

        logger.info(f"空売り指標算出完了: {len(result)}行")
        return result

    def calculate_topix_features(self) -> pd.DataFrame:
        """
        TOPIX（市場全体）関連指標を算出

        Returns:
            TOPIX特徴量DataFrame（columns: date, features...）
        """
        if self.topix is None or len(self.topix) == 0:
            logger.warning("TOPIXデータがありません")
            return pd.DataFrame()

        df = self.topix.copy()
        df = df.sort_values('date')

        # リターン
        df['topix_return_5d'] = df['close'].pct_change(5)
        df['topix_return_20d'] = df['close'].pct_change(20)
        df['topix_return_60d'] = df['close'].pct_change(60)

        # ボラティリティ
        returns = df['close'].pct_change()
        df['topix_volatility_20d'] = returns.rolling(20, min_periods=20).std() * np.sqrt(252)

        # 移動平均との乖離
        ma20 = df['close'].rolling(20, min_periods=20).mean()
        ma60 = df['close'].rolling(60, min_periods=60).mean()
        df['topix_vs_ma20'] = df['close'] / ma20 - 1
        df['topix_vs_ma60'] = df['close'] / ma60 - 1

        # トレンド（20日MA vs 60日MA）
        df['topix_trend'] = ma20 / ma60 - 1

        # 必要なカラムのみ選択
        feature_cols = [
            'date', 'close',
            'topix_return_5d', 'topix_return_20d', 'topix_return_60d',
            'topix_volatility_20d', 'topix_vs_ma20', 'topix_vs_ma60', 'topix_trend'
        ]
        result = df[[c for c in feature_cols if c in df.columns]]
        result = result.rename(columns={'close': 'topix_close'})

        logger.info(f"TOPIX指標算出完了: {len(result)}行")
        return result

    def calculate_global_index_features(self) -> pd.DataFrame:
        """
        海外指数関連指標を算出

        Returns:
            海外指数特徴量DataFrame（columns: date, features...）
        """
        if self.global_indices is None or len(self.global_indices) == 0:
            logger.warning("海外指数データがありません")
            return pd.DataFrame()

        global_features = GlobalIndexFeatures(self.global_indices)
        result = global_features.calculate_all()

        # マーケットレジーム（市場環境）を追加
        if len(result) > 0:
            result = calculate_market_regime(result)

        logger.info(f"海外指数指標算出完了: {len(result)}行")
        return result

    def calculate_all(self) -> dict:
        """
        全市場・需給指標を算出

        Returns:
            各種特徴量DataFrameの辞書
        """
        return {
            'margin': self.calculate_margin_features(),
            'short_selling': self.calculate_short_selling_features(),
            'topix': self.calculate_topix_features(),
            'global_indices': self.calculate_global_index_features(),
        }


def merge_market_features(
    prices_df: pd.DataFrame,
    margin_features: pd.DataFrame,
    short_selling_features: pd.DataFrame,
    topix_features: pd.DataFrame,
    stocks_df: Optional[pd.DataFrame] = None,
    global_index_features: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    市場・需給指標を株価データにマージ

    Args:
        prices_df: 株価データ（code, date, ...）
        margin_features: 信用取引特徴量
        short_selling_features: 空売り特徴量
        topix_features: TOPIX特徴量
        stocks_df: 銘柄マスタ（code, sector_33_code）
        global_index_features: 海外指数特徴量

    Returns:
        マージされたDataFrame
    """
    result = prices_df.copy()

    # TOPIX指標をマージ（全銘柄共通）
    if len(topix_features) > 0:
        result = result.merge(topix_features, on='date', how='left')
        logger.info("TOPIX指標をマージしました")

    # 海外指数指標をマージ（全銘柄共通）
    if global_index_features is not None and len(global_index_features) > 0:
        result = merge_global_features(result, global_index_features)
        logger.info("海外指数指標をマージしました")

    # 信用取引指標をマージ（銘柄・日付でマージ）
    # 注: 信用残高は週次データなので、最新の利用可能なデータを使用
    if len(margin_features) > 0:
        # 日付を調整（週末データを平日に適用）
        margin_features = margin_features.copy()

        # forward fillで週次データを日次に拡張
        margin_pivot = margin_features.set_index(['code', 'date']).unstack(level='code')
        margin_pivot = margin_pivot.resample('D').ffill()  # 日次に拡張
        margin_filled = margin_pivot.stack(level='code').reset_index()

        result = result.merge(
            margin_filled,
            on=['code', 'date'],
            how='left',
            suffixes=('', '_margin')
        )
        logger.info("信用取引指標をマージしました")

    # 空売り比率をマージ（セクター・日付でマージ）
    if len(short_selling_features) > 0 and stocks_df is not None:
        # 銘柄マスタからセクターコードを取得
        if 'sector_33_code' not in result.columns:
            sector_map = stocks_df[['code', 'sector_33_code']].drop_duplicates()
            result = result.merge(sector_map, on='code', how='left')

        result = result.merge(
            short_selling_features,
            on=['date', 'sector_33_code'],
            how='left',
            suffixes=('', '_short')
        )
        logger.info("空売り指標をマージしました")

    return result
