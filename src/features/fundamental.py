"""
ファンダメンタル指標算出モジュール

財務データから各種ファンダメンタル指標を算出する
Point-in-Time管理に対応
"""
import numpy as np
import pandas as pd
from datetime import date, timedelta
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class FundamentalFeatures:
    """ファンダメンタル指標算出クラス"""

    def __init__(self, financials_df: pd.DataFrame, prices_df: pd.DataFrame):
        """
        初期化

        Args:
            financials_df: 財務データ（columns: code, disclosed_date, fiscal_year,
                           fiscal_quarter, net_sales, operating_profit, profit, etc.）
            prices_df: 株価データ（columns: code, date, adjustment_close, volume）
        """
        self.financials = financials_df.copy()
        self.prices = prices_df.copy()

        # 日付でソート
        self.financials = self.financials.sort_values(['code', 'disclosed_date'])
        self.prices = self.prices.sort_values(['code', 'date'])

    def get_pit_financials(self, code: str, as_of_date: date) -> Optional[pd.Series]:
        """
        Point-in-Timeで利用可能な最新財務データを取得

        Args:
            code: 銘柄コード
            as_of_date: 基準日

        Returns:
            利用可能な最新の財務データ（Series）
        """
        mask = (
            (self.financials['code'] == code) &
            (self.financials['disclosed_date'] <= as_of_date)
        )
        available = self.financials[mask]

        if len(available) == 0:
            return None

        # 最新の開示日のデータを取得
        return available.iloc[-1]

    def get_yoy_data(self, code: str, as_of_date: date) -> tuple:
        """
        前年同期のデータを取得（YoY比較用）

        Returns:
            (current_data, yoy_data) のタプル
        """
        current = self.get_pit_financials(code, as_of_date)
        if current is None:
            return None, None

        # 前年同期を探す
        current_fy = current.get('fiscal_year')
        current_fq = current.get('fiscal_quarter')

        if current_fy is None or current_fq is None:
            return current, None

        mask = (
            (self.financials['code'] == code) &
            (self.financials['fiscal_year'] == current_fy - 1) &
            (self.financials['fiscal_quarter'] == current_fq) &
            (self.financials['disclosed_date'] <= as_of_date)
        )
        yoy_data = self.financials[mask]

        if len(yoy_data) == 0:
            return current, None

        return current, yoy_data.iloc[-1]

    def calculate_all(self, target_dates: Optional[list] = None) -> pd.DataFrame:
        """
        全ファンダメンタル指標を算出

        Args:
            target_dates: 計算対象日付リスト（省略時は株価データの全日付）

        Returns:
            特徴量DataFrame（columns: code, date, feature_1, ...）
        """
        if target_dates is None:
            target_dates = self.prices['date'].unique()

        results = []
        codes = self.financials['code'].unique()

        for code in codes:
            code_prices = self.prices[self.prices['code'] == code]

            for target_date in target_dates:
                if isinstance(target_date, str):
                    target_date = pd.to_datetime(target_date).date()
                elif isinstance(target_date, pd.Timestamp):
                    target_date = target_date.date()

                # 株価データが存在するか確認
                price_row = code_prices[code_prices['date'] == target_date]
                if len(price_row) == 0:
                    continue

                price = price_row.iloc[0]['adjustment_close']

                # 財務データ取得（Point-in-Time）
                current, yoy = self.get_yoy_data(code, target_date)

                if current is None:
                    continue

                # 特徴量算出
                features = self._calc_features(code, target_date, current, yoy, price)
                results.append(features)

        if len(results) == 0:
            return pd.DataFrame()

        result_df = pd.DataFrame(results)
        logger.info(f"ファンダメンタル指標算出完了: {len(result_df)}行")
        return result_df

    def _calc_features(self, code: str, target_date: date,
                       current: pd.Series, yoy: Optional[pd.Series],
                       price: float) -> dict:
        """各種ファンダメンタル指標を算出"""
        features = {
            'code': code,
            'date': target_date,
        }

        # === 成長性指標 ===
        if yoy is not None:
            # 売上高成長率（YoY）
            if self._is_valid(current.get('net_sales')) and self._is_valid(yoy.get('net_sales')):
                features['revenue_growth_yoy'] = (
                    current['net_sales'] - yoy['net_sales']
                ) / abs(yoy['net_sales']) if yoy['net_sales'] != 0 else np.nan

            # 営業利益成長率（YoY）
            if self._is_valid(current.get('operating_profit')) and self._is_valid(yoy.get('operating_profit')):
                if yoy['operating_profit'] != 0:
                    features['op_profit_growth_yoy'] = (
                        current['operating_profit'] - yoy['operating_profit']
                    ) / abs(yoy['operating_profit'])

            # EPS成長率（YoY）
            if self._is_valid(current.get('earnings_per_share')) and self._is_valid(yoy.get('earnings_per_share')):
                if yoy['earnings_per_share'] != 0:
                    features['eps_growth_yoy'] = (
                        current['earnings_per_share'] - yoy['earnings_per_share']
                    ) / abs(yoy['earnings_per_share'])

            # 純利益成長率（YoY）
            if self._is_valid(current.get('profit')) and self._is_valid(yoy.get('profit')):
                if yoy['profit'] != 0:
                    features['profit_growth_yoy'] = (
                        current['profit'] - yoy['profit']
                    ) / abs(yoy['profit'])

        # 会社予想との比較（サプライズ）
        if self._is_valid(current.get('change_net_sales')):
            features['revenue_change_guidance'] = current['change_net_sales']
        if self._is_valid(current.get('change_operating_profit')):
            features['op_profit_change_guidance'] = current['change_operating_profit']
        if self._is_valid(current.get('change_profit')):
            features['profit_change_guidance'] = current['change_profit']

        # === 収益性指標 ===
        # 営業利益率
        if self._is_valid(current.get('net_sales')) and self._is_valid(current.get('operating_profit')):
            if current['net_sales'] != 0:
                features['operating_margin'] = current['operating_profit'] / current['net_sales']

        # 純利益率
        if self._is_valid(current.get('net_sales')) and self._is_valid(current.get('profit')):
            if current['net_sales'] != 0:
                features['profit_margin'] = current['profit'] / current['net_sales']

        # ROE（純利益 / 自己資本）
        if self._is_valid(current.get('profit')) and self._is_valid(current.get('equity')):
            if current['equity'] != 0:
                features['roe'] = current['profit'] / current['equity']

        # ROA（純利益 / 総資産）
        if self._is_valid(current.get('profit')) and self._is_valid(current.get('total_assets')):
            if current['total_assets'] != 0:
                features['roa'] = current['profit'] / current['total_assets']

        # 営業利益率の変化
        if yoy is not None:
            if (self._is_valid(current.get('net_sales')) and
                self._is_valid(current.get('operating_profit')) and
                self._is_valid(yoy.get('net_sales')) and
                self._is_valid(yoy.get('operating_profit'))):

                current_margin = current['operating_profit'] / current['net_sales'] if current['net_sales'] != 0 else 0
                yoy_margin = yoy['operating_profit'] / yoy['net_sales'] if yoy['net_sales'] != 0 else 0
                features['operating_margin_change'] = current_margin - yoy_margin

        # === バリュエーション指標 ===
        # PER
        if self._is_valid(current.get('earnings_per_share')) and current['earnings_per_share'] > 0:
            features['per'] = price / current['earnings_per_share']

        # PBR
        if self._is_valid(current.get('book_value_per_share')) and current['book_value_per_share'] > 0:
            features['pbr'] = price / current['book_value_per_share']

        # PSR（株価売上高倍率）- 概算
        # 時価総額が必要だが、ここではEPSベースで概算
        if self._is_valid(current.get('net_sales')) and self._is_valid(current.get('earnings_per_share')):
            if current['earnings_per_share'] != 0:
                # 発行株数を逆算（純利益/EPS）
                shares = current.get('profit', 0) / current['earnings_per_share'] if current['earnings_per_share'] != 0 else None
                if shares and shares > 0 and current['net_sales'] > 0:
                    market_cap = price * shares
                    features['psr'] = market_cap / current['net_sales']

        # === 財務健全性 ===
        # 自己資本比率
        if self._is_valid(current.get('equity')) and self._is_valid(current.get('total_assets')):
            if current['total_assets'] != 0:
                features['equity_ratio'] = current['equity'] / current['total_assets']

        # === 予想関連 ===
        # 予想EPS成長率（実績EPSからの成長）
        if self._is_valid(current.get('forecast_earnings_per_share')) and self._is_valid(current.get('earnings_per_share')):
            if current['earnings_per_share'] != 0:
                features['forecast_eps_growth'] = (
                    current['forecast_earnings_per_share'] - current['earnings_per_share']
                ) / abs(current['earnings_per_share'])

        # 予想PER
        if self._is_valid(current.get('forecast_earnings_per_share')) and current['forecast_earnings_per_share'] > 0:
            features['forward_per'] = price / current['forecast_earnings_per_share']

        # 配当利回り
        if self._is_valid(current.get('dividend_per_share')) and price > 0:
            features['dividend_yield'] = current['dividend_per_share'] / price

        # === キャッシュフロー ===
        if self._is_valid(current.get('cash_flows_from_operating_activities')):
            features['operating_cf'] = current['cash_flows_from_operating_activities']

            # 営業CFマージン
            if self._is_valid(current.get('net_sales')) and current['net_sales'] != 0:
                features['operating_cf_margin'] = (
                    current['cash_flows_from_operating_activities'] / current['net_sales']
                )

        return features

    def _is_valid(self, value) -> bool:
        """値が有効かチェック"""
        if value is None:
            return False
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            return False
        return True


def calculate_sector_relative_valuation(df: pd.DataFrame,
                                        sector_col: str = 'sector_33_code') -> pd.DataFrame:
    """
    セクター内相対バリュエーションを算出

    Args:
        df: 特徴量DataFrame（code, date, sector_33_code, per, pbr等を含む）
        sector_col: セクターカラム名

    Returns:
        相対バリュエーション指標を追加したDataFrame
    """
    if sector_col not in df.columns:
        logger.warning(f"{sector_col}カラムが存在しません。相対バリュエーションはスキップします。")
        return df

    result = df.copy()

    for valuation_col in ['per', 'pbr', 'psr']:
        if valuation_col not in result.columns:
            continue

        # セクター中央値
        sector_median = result.groupby(['date', sector_col])[valuation_col].transform('median')

        # 相対バリュエーション（対セクター中央値）
        relative_col = f'{valuation_col}_relative_sector'
        result[relative_col] = result[valuation_col] / sector_median - 1

        # セクター内パーセンタイル順位
        rank_col = f'{valuation_col}_rank_sector'
        result[rank_col] = result.groupby(['date', sector_col])[valuation_col].transform(
            lambda x: x.rank(pct=True)
        )

    return result


def calculate_growth_cagr(financials_df: pd.DataFrame, years: int = 3) -> pd.DataFrame:
    """
    過去N年のCAGR（年平均成長率）を算出

    Args:
        financials_df: 財務データ
        years: CAGR計算期間（年）

    Returns:
        CAGRを追加したDataFrame
    """
    df = financials_df.copy()
    df = df.sort_values(['code', 'fiscal_year', 'fiscal_quarter'])

    results = []

    for code in df['code'].unique():
        code_data = df[df['code'] == code]

        # 通期データのみ使用（fiscal_quarter == 4）
        annual_data = code_data[code_data['fiscal_quarter'] == 4].copy()

        if len(annual_data) < years + 1:
            continue

        for idx in range(years, len(annual_data)):
            current = annual_data.iloc[idx]
            past = annual_data.iloc[idx - years]

            result = {
                'code': code,
                'fiscal_year': current['fiscal_year'],
                'disclosed_date': current['disclosed_date'],
            }

            # 売上高CAGR
            if current['net_sales'] and past['net_sales'] and past['net_sales'] > 0:
                result['revenue_cagr_3y'] = (
                    (current['net_sales'] / past['net_sales']) ** (1 / years) - 1
                )

            # 営業利益CAGR
            if (current['operating_profit'] and past['operating_profit'] and
                current['operating_profit'] > 0 and past['operating_profit'] > 0):
                result['op_profit_cagr_3y'] = (
                    (current['operating_profit'] / past['operating_profit']) ** (1 / years) - 1
                )

            # EPS CAGR
            if (current['earnings_per_share'] and past['earnings_per_share'] and
                current['earnings_per_share'] > 0 and past['earnings_per_share'] > 0):
                result['eps_cagr_3y'] = (
                    (current['earnings_per_share'] / past['earnings_per_share']) ** (1 / years) - 1
                )

            results.append(result)

    if len(results) == 0:
        return pd.DataFrame()

    return pd.DataFrame(results)
