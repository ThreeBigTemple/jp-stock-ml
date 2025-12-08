"""
TDnet適時開示シグナル算出モジュール

適時開示情報（業績修正、自社株買い、配当修正等）から
株価予測に有用なシグナルを算出する
"""
import numpy as np
import pandas as pd
from datetime import date, timedelta
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class DisclosureFeatures:
    """TDnet適時開示シグナル算出クラス"""

    # 開示タイプの重要度ウェイト
    DISCLOSURE_WEIGHTS = {
        'earnings_revision': 1.0,       # 業績予想修正（最重要）
        'dividend_revision': 0.6,       # 配当予想修正
        'buyback': 0.8,                 # 自社株買い
        'buyback_result': 0.4,          # 自社株買い結果
        'stock_split': 0.3,             # 株式分割
        'new_stock': -0.2,              # 新株発行（希薄化）
    }

    def __init__(self, disclosure_df: pd.DataFrame):
        """
        初期化

        Args:
            disclosure_df: 適時開示データ（columns: code, date, time, title,
                           disclosure_type, revision_sales, revision_operating, revision_net）
        """
        self.disclosure = disclosure_df.copy()

        # 日付でソート
        self.disclosure = self.disclosure.sort_values(['code', 'date'])

    def calculate_all(self, prices_df: pd.DataFrame,
                      lookback_days: int = 60) -> pd.DataFrame:
        """
        全適時開示シグナルを算出

        Args:
            prices_df: 株価データ（対象日付の特定用）
            lookback_days: シグナル算出のルックバック期間

        Returns:
            特徴量DataFrame（columns: code, date, features...）
        """
        if len(self.disclosure) == 0:
            logger.warning("適時開示データがありません")
            return pd.DataFrame()

        target_dates = prices_df['date'].unique()
        codes = prices_df['code'].unique()

        results = []

        for code in codes:
            code_disclosures = self.disclosure[self.disclosure['code'] == code]
            code_prices = prices_df[prices_df['code'] == code]

            for target_date in target_dates:
                if isinstance(target_date, str):
                    target_date = pd.to_datetime(target_date).date()
                elif isinstance(target_date, pd.Timestamp):
                    target_date = target_date.date()

                # 株価データが存在するか確認
                if len(code_prices[code_prices['date'] == target_date]) == 0:
                    continue

                # Point-in-Time: 基準日以前の開示のみ
                lookback_start = target_date - timedelta(days=lookback_days)
                available = code_disclosures[
                    (code_disclosures['date'] <= target_date) &
                    (code_disclosures['date'] >= lookback_start)
                ]

                features = self._calc_features(code, target_date, available, lookback_days)
                results.append(features)

        if len(results) == 0:
            return pd.DataFrame()

        result_df = pd.DataFrame(results)
        logger.info(f"適時開示シグナル算出完了: {len(result_df)}行")
        return result_df

    def _calc_features(self, code: str, target_date: date,
                       disclosures: pd.DataFrame, lookback_days: int) -> dict:
        """適時開示シグナルを算出"""
        features = {
            'code': code,
            'date': target_date,
        }

        # === 開示回数関連 ===
        # 直近N日間の開示回数
        features['disclosure_count'] = len(disclosures)

        # 開示タイプ別回数
        for dtype in self.DISCLOSURE_WEIGHTS.keys():
            type_count = len(disclosures[disclosures['disclosure_type'] == dtype])
            features[f'disclosure_{dtype}_count'] = type_count

        # === 業績予想修正関連（最重要シグナル）===
        earnings_revisions = disclosures[disclosures['disclosure_type'] == 'earnings_revision']

        if len(earnings_revisions) > 0:
            # 直近の業績予想修正
            latest_revision = earnings_revisions.iloc[-1]

            # 業績修正があったことを示すフラグ
            features['has_earnings_revision'] = 1

            # 業績修正からの経過日数
            if isinstance(latest_revision['date'], pd.Timestamp):
                days_since = (pd.to_datetime(target_date) - latest_revision['date']).days
            else:
                days_since = (target_date - latest_revision['date']).days
            features['days_since_earnings_revision'] = days_since

            # 修正率（データがある場合）
            if self._is_valid(latest_revision.get('revision_sales')):
                features['latest_revision_sales'] = latest_revision['revision_sales']
            if self._is_valid(latest_revision.get('revision_operating')):
                features['latest_revision_operating'] = latest_revision['revision_operating']
            if self._is_valid(latest_revision.get('revision_net')):
                features['latest_revision_net'] = latest_revision['revision_net']

            # 複数回の業績修正（モメンタム）
            features['earnings_revision_count'] = len(earnings_revisions)
        else:
            features['has_earnings_revision'] = 0
            features['days_since_earnings_revision'] = lookback_days  # 最大値

        # === 配当予想修正関連 ===
        dividend_revisions = disclosures[disclosures['disclosure_type'] == 'dividend_revision']

        if len(dividend_revisions) > 0:
            features['has_dividend_revision'] = 1
            latest_div = dividend_revisions.iloc[-1]

            if isinstance(latest_div['date'], pd.Timestamp):
                days_since = (pd.to_datetime(target_date) - latest_div['date']).days
            else:
                days_since = (target_date - latest_div['date']).days
            features['days_since_dividend_revision'] = days_since
        else:
            features['has_dividend_revision'] = 0

        # === 自社株買い関連 ===
        buybacks = disclosures[disclosures['disclosure_type'] == 'buyback']

        if len(buybacks) > 0:
            features['has_buyback'] = 1
            features['buyback_count'] = len(buybacks)

            latest_bb = buybacks.iloc[-1]
            if isinstance(latest_bb['date'], pd.Timestamp):
                days_since = (pd.to_datetime(target_date) - latest_bb['date']).days
            else:
                days_since = (target_date - latest_bb['date']).days
            features['days_since_buyback'] = days_since
        else:
            features['has_buyback'] = 0
            features['buyback_count'] = 0

        # === 株式分割関連 ===
        splits = disclosures[disclosures['disclosure_type'] == 'stock_split']
        features['has_stock_split'] = 1 if len(splits) > 0 else 0

        # === 新株発行関連（ネガティブシグナル）===
        new_stocks = disclosures[disclosures['disclosure_type'] == 'new_stock']
        features['has_new_stock'] = 1 if len(new_stocks) > 0 else 0

        # === 統合スコア ===
        # 各開示タイプにウェイトを掛けて合算
        score = 0.0
        for dtype, weight in self.DISCLOSURE_WEIGHTS.items():
            type_disclosures = disclosures[disclosures['disclosure_type'] == dtype]
            if len(type_disclosures) > 0:
                # 直近の開示ほど重要（時間減衰）
                for _, disc in type_disclosures.iterrows():
                    if isinstance(disc['date'], pd.Timestamp):
                        days_ago = (pd.to_datetime(target_date) - disc['date']).days
                    else:
                        days_ago = (target_date - disc['date']).days
                    decay = np.exp(-days_ago / 30)  # 30日で約37%に減衰
                    score += weight * decay

        features['disclosure_score'] = score

        return features

    def _is_valid(self, value) -> bool:
        """値が有効かチェック"""
        if value is None:
            return False
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            return False
        return True


def calculate_disclosure_momentum(df: pd.DataFrame,
                                  disclosure_df: pd.DataFrame,
                                  windows: List[int] = [20, 60]) -> pd.DataFrame:
    """
    開示モメンタム指標を算出

    Args:
        df: 価格データ（code, date）
        disclosure_df: 開示データ
        windows: 集計ウィンドウ（日数）

    Returns:
        モメンタム指標を追加したDataFrame
    """
    result = df.copy()

    for window in windows:
        # ウィンドウ内の開示回数を算出
        disclosure_counts = []

        for _, row in result.iterrows():
            code = row['code']
            target_date = row['date']

            if isinstance(target_date, pd.Timestamp):
                target_date = target_date.date()

            lookback_start = target_date - timedelta(days=window)

            count = len(disclosure_df[
                (disclosure_df['code'] == code) &
                (disclosure_df['date'] >= lookback_start) &
                (disclosure_df['date'] <= target_date)
            ])
            disclosure_counts.append(count)

        result[f'disclosure_count_{window}d'] = disclosure_counts

    return result


def aggregate_sector_disclosures(disclosure_df: pd.DataFrame,
                                 stocks_df: pd.DataFrame) -> pd.DataFrame:
    """
    セクター別開示統計を算出

    Args:
        disclosure_df: 開示データ
        stocks_df: 銘柄マスタ（sector_33_code含む）

    Returns:
        セクター別統計DataFrame
    """
    if len(disclosure_df) == 0:
        return pd.DataFrame()

    # 銘柄にセクター情報を付与
    merged = disclosure_df.merge(
        stocks_df[['code', 'sector_33_code']],
        on='code',
        how='left'
    )

    # 日付×セクター×タイプで集計
    stats = merged.groupby(['date', 'sector_33_code', 'disclosure_type']).size()
    stats = stats.unstack(fill_value=0).reset_index()

    logger.info(f"セクター別開示統計算出完了: {len(stats)}行")
    return stats
