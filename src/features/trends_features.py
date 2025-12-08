"""
Google Trends検索トレンド指標算出モジュール

企業・銘柄の検索トレンドから投資家関心度を測定する
（補助的指標、ノイズが多い可能性あり）
"""
import numpy as np
import pandas as pd
from datetime import date, timedelta
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class TrendsFeatures:
    """Google Trends検索トレンド指標算出クラス"""

    def __init__(self, trends_df: pd.DataFrame):
        """
        初期化

        Args:
            trends_df: 検索トレンドデータ（columns: code, week_start,
                       keyword, interest）
        """
        self.trends = trends_df.copy()

        # 週開始日でソート
        self.trends = self.trends.sort_values(['code', 'week_start'])

    def calculate_all(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        全検索トレンド特徴量を算出

        Args:
            prices_df: 株価データ（対象日付の特定用）

        Returns:
            特徴量DataFrame（columns: code, date, features...）
        """
        if len(self.trends) == 0:
            logger.warning("検索トレンドデータがありません")
            return pd.DataFrame()

        # 週次データを日次に拡張（forward fill）
        daily_trends = self._expand_to_daily()

        # 各銘柄の特徴量を算出
        results = []

        codes = daily_trends['code'].unique()
        target_dates = prices_df['date'].unique()

        for code in codes:
            code_trends = daily_trends[daily_trends['code'] == code]
            code_prices = prices_df[prices_df['code'] == code]

            for target_date in target_dates:
                if isinstance(target_date, str):
                    target_date = pd.to_datetime(target_date).date()
                elif isinstance(target_date, pd.Timestamp):
                    target_date = target_date.date()

                # 株価データが存在するか確認
                if len(code_prices[code_prices['date'] == target_date]) == 0:
                    continue

                # Point-in-Time: 基準日以前のトレンドのみ
                available = code_trends[code_trends['date'] <= target_date]

                if len(available) == 0:
                    continue

                features = self._calc_features(code, target_date, available)
                results.append(features)

        if len(results) == 0:
            return pd.DataFrame()

        result_df = pd.DataFrame(results)
        logger.info(f"検索トレンド特徴量算出完了: {len(result_df)}行")
        return result_df

    def _expand_to_daily(self) -> pd.DataFrame:
        """週次データを日次に拡張"""
        daily_data = []

        for code in self.trends['code'].unique():
            code_data = self.trends[self.trends['code'] == code].copy()

            # 週開始日から次の週開始日までforward fill
            for _, row in code_data.iterrows():
                week_start = row['week_start']
                if isinstance(week_start, pd.Timestamp):
                    week_start = week_start.date()

                # 週の各日にデータを割り当て
                for i in range(7):
                    daily_data.append({
                        'code': code,
                        'date': week_start + timedelta(days=i),
                        'interest': row['interest'],
                        'keyword': row.get('keyword', '')
                    })

        return pd.DataFrame(daily_data)

    def _calc_features(self, code: str, target_date: date,
                       trends: pd.DataFrame) -> dict:
        """検索トレンド特徴量を算出"""
        features = {
            'code': code,
            'date': target_date,
        }

        interest = trends['interest']

        # === 現在の検索関心度 ===
        current_interest = interest.iloc[-1] if len(interest) > 0 else 0
        features['search_interest'] = current_interest

        # === 検索関心度の変化 ===
        if len(interest) >= 4:  # 4週間分
            # 直近1週間 vs 前週
            recent = interest.iloc[-7:].mean() if len(interest) >= 7 else interest.mean()
            prev = interest.iloc[-14:-7].mean() if len(interest) >= 14 else interest.iloc[:-7].mean()

            if prev > 0:
                features['search_interest_change_1w'] = (recent - prev) / prev
            else:
                features['search_interest_change_1w'] = 0

        # === 検索関心度の移動平均 ===
        if len(interest) >= 28:  # 4週間
            ma4w = interest.iloc[-28:].mean()
            features['search_interest_ma4w'] = ma4w

            # 現在 vs 4週平均
            if ma4w > 0:
                features['search_vs_ma4w'] = current_interest / ma4w - 1

        # === 検索関心度の標準偏差（変動性）===
        if len(interest) >= 8:
            features['search_volatility'] = interest.iloc[-8:].std()

        # === 検索急増フラグ ===
        # 直近が過去平均の1.5倍以上
        if len(interest) >= 28:
            avg_interest = interest.iloc[:-7].mean()
            if avg_interest > 0:
                features['search_spike'] = int(current_interest > avg_interest * 1.5)
            else:
                features['search_spike'] = 0

        # === 検索関心度のトレンド（傾き）===
        if len(interest) >= 8:
            # 直近8週の傾き（線形回帰の傾き）
            x = np.arange(min(len(interest), 8))
            y = interest.iloc[-8:].values
            if len(y) > 1:
                slope = np.polyfit(x, y, 1)[0]
                features['search_trend_slope'] = slope

        # === 検索関心度のパーセンタイル ===
        if len(interest) >= 52:  # 1年分
            percentile = (interest.iloc[-1] > interest.iloc[-52:]).mean()
            features['search_percentile_52w'] = percentile

        return features


def merge_trends_features(prices_df: pd.DataFrame,
                          trends_features: pd.DataFrame) -> pd.DataFrame:
    """
    検索トレンド特徴量を株価データにマージ

    Args:
        prices_df: 株価データ（code, date, ...）
        trends_features: 検索トレンド特徴量

    Returns:
        マージされたDataFrame
    """
    if len(trends_features) == 0:
        logger.warning("検索トレンド特徴量がありません")
        return prices_df

    result = prices_df.copy()

    # 日付型を統一
    result['date'] = pd.to_datetime(result['date']).dt.date
    trends_features = trends_features.copy()
    trends_features['date'] = pd.to_datetime(trends_features['date']).dt.date

    # マージ
    result = result.merge(trends_features, on=['code', 'date'], how='left')

    logger.info("検索トレンド特徴量をマージしました")
    return result


def calculate_sector_trends(trends_df: pd.DataFrame,
                            stocks_df: pd.DataFrame) -> pd.DataFrame:
    """
    セクター別検索トレンドを算出

    Args:
        trends_df: 検索トレンドデータ
        stocks_df: 銘柄マスタ（sector_33_code含む）

    Returns:
        セクター別トレンド統計DataFrame
    """
    if len(trends_df) == 0:
        return pd.DataFrame()

    # 銘柄にセクター情報を付与
    merged = trends_df.merge(
        stocks_df[['code', 'sector_33_code']],
        on='code',
        how='left'
    )

    # セクター×週で集計
    sector_trends = merged.groupby(['week_start', 'sector_33_code']).agg({
        'interest': ['mean', 'std', 'max']
    }).reset_index()

    # カラム名をフラット化
    sector_trends.columns = [
        'week_start', 'sector_33_code',
        'sector_search_mean', 'sector_search_std', 'sector_search_max'
    ]

    logger.info(f"セクター別検索トレンド算出完了: {len(sector_trends)}行")
    return sector_trends


def calculate_relative_search_interest(df: pd.DataFrame,
                                       sector_col: str = 'sector_33_code') -> pd.DataFrame:
    """
    セクター内相対検索関心度を算出

    Args:
        df: 特徴量DataFrame（search_interest, sector_33_code含む）
        sector_col: セクターカラム名

    Returns:
        相対検索関心度を追加したDataFrame
    """
    if sector_col not in df.columns or 'search_interest' not in df.columns:
        return df

    result = df.copy()

    # セクター中央値
    sector_median = result.groupby(['date', sector_col])['search_interest'].transform('median')

    # 相対検索関心度
    result['search_interest_relative_sector'] = result['search_interest'] / sector_median.replace(0, np.nan) - 1

    # セクター内パーセンタイル
    result['search_interest_rank_sector'] = result.groupby(['date', sector_col])['search_interest'].transform(
        lambda x: x.rank(pct=True)
    )

    return result
