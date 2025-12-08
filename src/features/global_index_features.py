"""
海外指数・グローバルマーケット指標算出モジュール

S&P500、VIX、金利、コモディティ等の海外指数から
日本株予測に有用な市場環境指標を算出する
"""
import numpy as np
import pandas as pd
from datetime import date, timedelta
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)


class GlobalIndexFeatures:
    """海外指数・グローバルマーケット指標算出クラス"""

    # 指数のカテゴリ分類
    INDEX_CATEGORIES = {
        'equity': ['sp500', 'nasdaq', 'dow'],
        'volatility': ['vix'],
        'rates': ['us10y'],
        'fx': ['dxy'],
        'commodity': ['wti', 'gold'],
    }

    def __init__(self, global_indices_df: pd.DataFrame):
        """
        初期化

        Args:
            global_indices_df: 海外指数データ（columns: index_name, date,
                               open, high, low, close, volume）
        """
        self.indices = global_indices_df.copy()

        # 日付でソート
        self.indices = self.indices.sort_values(['index_name', 'date'])

    def calculate_all(self) -> pd.DataFrame:
        """
        全海外指数特徴量を算出

        Returns:
            特徴量DataFrame（columns: date, features...）
        """
        if len(self.indices) == 0:
            logger.warning("海外指数データがありません")
            return pd.DataFrame()

        # 各指数の特徴量を算出
        all_features = {}

        for index_name in self.indices['index_name'].unique():
            idx_data = self.indices[self.indices['index_name'] == index_name].copy()
            idx_data = idx_data.sort_values('date')

            features = self._calc_index_features(idx_data, index_name)
            all_features[index_name] = features

        # 日付でマージ
        result = None
        for index_name, features in all_features.items():
            if result is None:
                result = features
            else:
                result = result.merge(features, on='date', how='outer')

        if result is None:
            return pd.DataFrame()

        # 指数間の相関・乖離特徴量
        result = self._calc_cross_index_features(result)

        result = result.sort_values('date')

        logger.info(f"海外指数特徴量算出完了: {len(result)}行, {len(result.columns)}列")
        return result

    def _calc_index_features(self, df: pd.DataFrame, index_name: str) -> pd.DataFrame:
        """個別指数の特徴量を算出"""
        result = df[['date']].copy()
        close = df['close']

        prefix = index_name

        # === リターン ===
        result[f'{prefix}_return_1d'] = close.pct_change(1)
        result[f'{prefix}_return_5d'] = close.pct_change(5)
        result[f'{prefix}_return_20d'] = close.pct_change(20)
        result[f'{prefix}_return_60d'] = close.pct_change(60)

        # === ボラティリティ ===
        returns = close.pct_change()
        result[f'{prefix}_volatility_20d'] = returns.rolling(20, min_periods=10).std() * np.sqrt(252)

        # === モメンタム ===
        result[f'{prefix}_momentum_20d'] = (close - close.shift(20)) / close.shift(20)

        # === 移動平均乖離 ===
        ma20 = close.rolling(20, min_periods=10).mean()
        ma60 = close.rolling(60, min_periods=30).mean()
        result[f'{prefix}_vs_ma20'] = close / ma20 - 1
        result[f'{prefix}_vs_ma60'] = close / ma60 - 1

        # === トレンド ===
        result[f'{prefix}_trend'] = ma20 / ma60 - 1

        # === 直近の値（レベル情報）===
        result[f'{prefix}_close'] = close.values

        # VIX特有の特徴量
        if index_name == 'vix':
            # VIXレベル（水準自体が重要）
            result['vix_level'] = close.values

            # VIX急騰フラグ
            result['vix_spike'] = (close.pct_change() > 0.2).astype(int)

            # VIX高水準フラグ（20以上）
            result['vix_high'] = (close > 20).astype(int)

            # VIX低水準フラグ（15以下＝安心感）
            result['vix_low'] = (close < 15).astype(int)

        # 金利特有の特徴量
        if index_name == 'us10y':
            # 金利水準
            result['us10y_level'] = close.values

            # 金利上昇トレンド
            result['us10y_rising'] = (close > close.shift(5)).astype(int)

        # 原油特有の特徴量
        if index_name == 'wti':
            # 原油高フラグ
            result['wti_high'] = (close > close.rolling(60).mean() * 1.1).astype(int)

        return result

    def _calc_cross_index_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """指数間のクロス特徴量を算出"""
        result = df.copy()

        # === リスクオン・リスクオフ指標 ===
        # VIXが低い & S&P500が上昇 = リスクオン
        if 'vix_close' in result.columns and 'sp500_return_20d' in result.columns:
            result['risk_on_score'] = (
                -result['vix_close'].rank(pct=True) +
                result['sp500_return_20d'].rank(pct=True)
            ) / 2

        # === 株式・債券相関 ===
        if 'sp500_return_5d' in result.columns and 'us10y_return_5d' in result.columns:
            # 直近20日の相関（ローリング）
            sp500_ret = result['sp500_return_5d']
            us10y_ret = result['us10y_return_5d']

            correlation = sp500_ret.rolling(20).corr(us10y_ret)
            result['equity_bond_correlation'] = correlation

        # === ドル高・円安影響 ===
        if 'dxy_return_5d' in result.columns:
            # ドル高は日本株にプラス（円安効果）の傾向
            result['dxy_strength'] = result['dxy_return_5d'].rank(pct=True)

        # === コモディティセンチメント ===
        if 'wti_return_5d' in result.columns and 'gold_return_5d' in result.columns:
            # 金上昇 & 原油下落 = リスクオフ懸念
            result['commodity_risk_signal'] = (
                result['gold_return_5d'] - result['wti_return_5d']
            )

        # === グローバル株式モメンタム ===
        equity_returns = []
        for idx in self.INDEX_CATEGORIES['equity']:
            col = f'{idx}_return_20d'
            if col in result.columns:
                equity_returns.append(col)

        if equity_returns:
            result['global_equity_momentum'] = result[equity_returns].mean(axis=1)

        return result

    def get_lagged_features(self, df: pd.DataFrame, lag_days: int = 1) -> pd.DataFrame:
        """
        日本市場向けにラグ付き特徴量を生成

        米国市場は日本より約14時間遅れて引けるため、
        前日の米国市場データが当日の日本市場に影響する

        Args:
            df: 特徴量DataFrame
            lag_days: ラグ日数（デフォルト1日）

        Returns:
            ラグ付き特徴量DataFrame
        """
        result = df.copy()

        # 日付をlag_days日シフト（米国の前日データを日本の当日にマッピング）
        result['date'] = pd.to_datetime(result['date']) + pd.Timedelta(days=lag_days)

        # カラム名にラグを明示
        rename_cols = {}
        for col in result.columns:
            if col != 'date':
                rename_cols[col] = f'{col}_lag{lag_days}d'

        result = result.rename(columns=rename_cols)

        logger.info(f"ラグ{lag_days}日の特徴量生成完了")
        return result


def merge_global_features(prices_df: pd.DataFrame,
                          global_features: pd.DataFrame) -> pd.DataFrame:
    """
    海外指数特徴量を日本株価データにマージ

    Args:
        prices_df: 日本株価データ（code, date, ...）
        global_features: 海外指数特徴量

    Returns:
        マージされたDataFrame
    """
    if len(global_features) == 0:
        logger.warning("海外指数特徴量がありません")
        return prices_df

    result = prices_df.copy()

    # 日付型を統一
    result['date'] = pd.to_datetime(result['date'])
    global_features = global_features.copy()
    global_features['date'] = pd.to_datetime(global_features['date'])

    # マージ
    result = result.merge(global_features, on='date', how='left')

    logger.info("海外指数特徴量をマージしました")
    return result


def calculate_market_regime(global_features: pd.DataFrame) -> pd.DataFrame:
    """
    マーケットレジーム（市場環境）を分類

    Args:
        global_features: 海外指数特徴量

    Returns:
        レジームラベルを追加したDataFrame
    """
    result = global_features.copy()

    # レジーム判定条件
    conditions = []
    labels = []

    # リスクオン: VIX低い & 株式上昇
    if 'vix_close' in result.columns and 'sp500_return_20d' in result.columns:
        risk_on = (result['vix_close'] < 20) & (result['sp500_return_20d'] > 0)
        conditions.append(risk_on)
        labels.append('risk_on')

        # リスクオフ: VIX高い & 株式下落
        risk_off = (result['vix_close'] > 25) & (result['sp500_return_20d'] < -0.05)
        conditions.append(risk_off)
        labels.append('risk_off')

        # ニュートラル
        neutral = ~risk_on & ~risk_off
        conditions.append(neutral)
        labels.append('neutral')

        result['market_regime'] = np.select(
            conditions,
            labels,
            default='neutral'
        )

        # 数値エンコーディング
        regime_map = {'risk_off': -1, 'neutral': 0, 'risk_on': 1}
        result['market_regime_score'] = result['market_regime'].map(regime_map)

    return result
