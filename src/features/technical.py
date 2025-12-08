"""
テクニカル指標算出モジュール

株価データから各種テクニカル指標を算出する
"""
import numpy as np
import pandas as pd
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class TechnicalFeatures:
    """テクニカル指標算出クラス"""

    def __init__(self, prices_df: pd.DataFrame, topix_df: Optional[pd.DataFrame] = None):
        """
        初期化

        Args:
            prices_df: 株価データ（columns: code, date, open, high, low, close, volume,
                       adjustment_close, adjustment_volume）
            topix_df: TOPIXデータ（columns: date, close）
        """
        self.prices = prices_df.copy()
        self.topix = topix_df.copy() if topix_df is not None else None

        # 日付でソート
        self.prices = self.prices.sort_values(['code', 'date'])
        if self.topix is not None:
            self.topix = self.topix.sort_values('date')

    def calculate_all(self) -> pd.DataFrame:
        """
        全テクニカル指標を算出

        Returns:
            特徴量を追加したDataFrame
        """
        df = self.prices.copy()

        # 銘柄ごとにグループ化して計算
        features_list = []

        for code, group in df.groupby('code'):
            group = group.sort_values('date').copy()

            # リターン系
            group = self._calc_returns(group)

            # モメンタム
            group = self._calc_momentum(group)

            # 移動平均
            group = self._calc_moving_averages(group)

            # ボラティリティ
            group = self._calc_volatility(group)

            # 出来高
            group = self._calc_volume_features(group)

            # TOPIX相対強度
            if self.topix is not None:
                group = self._calc_relative_strength(group)

            features_list.append(group)

        result = pd.concat(features_list, ignore_index=True)

        logger.info(f"テクニカル指標算出完了: {len(result)}行, {len(result.columns)}列")
        return result

    def _calc_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """期間リターンを算出"""
        close = df['adjustment_close']

        # 日次リターン
        df['return_1d'] = close.pct_change(1)

        # 期間リターン（営業日ベース: 1ヶ月≒20日）
        df['return_5d'] = close.pct_change(5)
        df['return_20d'] = close.pct_change(20)    # 1ヶ月
        df['return_60d'] = close.pct_change(60)    # 3ヶ月
        df['return_120d'] = close.pct_change(120)  # 6ヶ月
        df['return_240d'] = close.pct_change(240)  # 12ヶ月

        return df

    def _calc_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """モメンタム指標を算出"""
        close = df['adjustment_close']

        # モメンタム（N日前との差分 / N日前）
        df['momentum_20d'] = (close - close.shift(20)) / close.shift(20)
        df['momentum_60d'] = (close - close.shift(60)) / close.shift(60)

        # RSI
        df['rsi_14d'] = self._calc_rsi(close, 14)

        return df

    def _calc_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """RSIを算出"""
        delta = series.diff()

        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calc_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """移動平均関連指標を算出"""
        close = df['adjustment_close']

        # 各種移動平均
        ma5 = close.rolling(5, min_periods=5).mean()
        ma20 = close.rolling(20, min_periods=20).mean()
        ma60 = close.rolling(60, min_periods=60).mean()
        ma200 = close.rolling(200, min_periods=200).mean()

        # 移動平均比率
        df['ma_5_20_ratio'] = ma5 / ma20 - 1
        df['ma_20_60_ratio'] = ma20 / ma60 - 1

        # 価格と移動平均の乖離率
        df['price_vs_ma20'] = close / ma20 - 1
        df['price_vs_ma60'] = close / ma60 - 1
        df['price_vs_ma200'] = close / ma200 - 1

        # ゴールデンクロス・デッドクロス（前日との比較）
        df['ma_5_above_20'] = (ma5 > ma20).astype(int)
        df['ma_20_above_60'] = (ma20 > ma60).astype(int)

        return df

    def _calc_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """ボラティリティ指標を算出"""
        close = df['adjustment_close']
        high = df['adjustment_high'] if 'adjustment_high' in df.columns else df['high']
        low = df['adjustment_low'] if 'adjustment_low' in df.columns else df['low']

        # 日次リターンの標準偏差（年率換算）
        returns = close.pct_change()
        df['volatility_20d'] = returns.rolling(20, min_periods=20).std() * np.sqrt(252)
        df['volatility_60d'] = returns.rolling(60, min_periods=60).std() * np.sqrt(252)

        # ATR (Average True Range)
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr_14d'] = true_range.rolling(14, min_periods=14).mean()

        # ATRの価格比（正規化ATR）
        df['atr_14d_pct'] = df['atr_14d'] / close

        return df

    def _calc_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """出来高関連指標を算出"""
        volume = df['adjustment_volume'] if 'adjustment_volume' in df.columns else df['volume']

        # 移動平均出来高
        vol_ma20 = volume.rolling(20, min_periods=20).mean()
        vol_ma60 = volume.rolling(60, min_periods=60).mean()

        # 出来高比率
        df['volume_ratio_20d'] = volume / vol_ma20
        df['volume_ma_20_60_ratio'] = vol_ma20 / vol_ma60

        # 出来高変化率
        df['volume_change_5d'] = volume.pct_change(5)

        return df

    def _calc_relative_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """TOPIX相対強度を算出"""
        if self.topix is None or len(self.topix) == 0:
            return df

        # TOPIXデータをマージ
        topix = self.topix[['date', 'close']].rename(columns={'close': 'topix_close'})
        df = df.merge(topix, on='date', how='left')

        # TOPIXリターン
        df['topix_return_20d'] = df['topix_close'].pct_change(20)
        df['topix_return_60d'] = df['topix_close'].pct_change(60)

        # 相対強度（銘柄リターン - TOPIXリターン）
        df['rs_vs_topix_20d'] = df['return_20d'] - df['topix_return_20d']
        df['rs_vs_topix_60d'] = df['return_60d'] - df['topix_return_60d']

        # TOPIXカラムを削除
        df = df.drop(columns=['topix_close', 'topix_return_20d', 'topix_return_60d'], errors='ignore')

        return df


def calculate_sector_rank(df: pd.DataFrame, sector_col: str = 'sector_33_code') -> pd.DataFrame:
    """
    セクター内リターン順位を算出

    Args:
        df: 株価データ（code, date, sector_33_code, return_20d等を含む）
        sector_col: セクターカラム名

    Returns:
        順位（パーセンタイル）を追加したDataFrame
    """
    if sector_col not in df.columns:
        logger.warning(f"{sector_col}カラムが存在しません。セクター順位はスキップします。")
        return df

    result = df.copy()

    # 日付×セクターでグループ化してパーセンタイル順位を算出
    for return_col in ['return_20d', 'return_60d']:
        if return_col not in result.columns:
            continue

        rank_col = f'rs_rank_sector_{return_col.split("_")[1]}'

        result[rank_col] = result.groupby(['date', sector_col])[return_col].transform(
            lambda x: x.rank(pct=True)
        )

    return result
