"""
EDINET詳細財務指標算出モジュール

EDINETの詳細財務データ（研究開発費、設備投資等）から
成長銘柄発掘に有用な指標を算出する
"""
import numpy as np
import pandas as pd
from datetime import date
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class EdinetFeatures:
    """EDINET詳細財務指標算出クラス"""

    def __init__(self, edinet_df: pd.DataFrame, prices_df: pd.DataFrame):
        """
        初期化

        Args:
            edinet_df: EDINET財務データ（columns: code, submit_date, fiscal_year_end,
                       net_sales, operating_income, rd_expenses, capex, etc.）
            prices_df: 株価データ（columns: code, date, adjustment_close）
        """
        self.edinet = edinet_df.copy()
        self.prices = prices_df.copy()

        # 日付でソート
        self.edinet = self.edinet.sort_values(['code', 'submit_date'])
        self.prices = self.prices.sort_values(['code', 'date'])

    def get_pit_edinet(self, code: str, as_of_date: date) -> Optional[pd.Series]:
        """
        Point-in-Timeで利用可能な最新EDINET財務データを取得

        Args:
            code: 銘柄コード
            as_of_date: 基準日

        Returns:
            利用可能な最新の財務データ（Series）
        """
        mask = (
            (self.edinet['code'] == code) &
            (self.edinet['submit_date'] <= as_of_date)
        )
        available = self.edinet[mask]

        if len(available) == 0:
            return None

        return available.iloc[-1]

    def get_yoy_edinet(self, code: str, as_of_date: date) -> tuple:
        """
        前年同期のEDINETデータを取得

        Returns:
            (current_data, yoy_data) のタプル
        """
        current = self.get_pit_edinet(code, as_of_date)
        if current is None:
            return None, None

        current_fiscal_end = current.get('fiscal_year_end')
        if current_fiscal_end is None:
            return current, None

        # 前年の決算期末日を探す
        target_fiscal_end = current_fiscal_end - pd.DateOffset(years=1)

        mask = (
            (self.edinet['code'] == code) &
            (self.edinet['submit_date'] <= as_of_date) &
            (self.edinet['fiscal_year_end'] <= target_fiscal_end + pd.DateOffset(days=30)) &
            (self.edinet['fiscal_year_end'] >= target_fiscal_end - pd.DateOffset(days=30))
        )
        yoy_data = self.edinet[mask]

        if len(yoy_data) == 0:
            return current, None

        return current, yoy_data.iloc[-1]

    def calculate_all(self, target_dates: Optional[List] = None) -> pd.DataFrame:
        """
        全EDINET詳細財務指標を算出

        Args:
            target_dates: 計算対象日付リスト

        Returns:
            特徴量DataFrame（columns: code, date, feature_1, ...）
        """
        if len(self.edinet) == 0:
            logger.warning("EDINETデータがありません")
            return pd.DataFrame()

        if target_dates is None:
            target_dates = self.prices['date'].unique()

        results = []
        codes = self.edinet['code'].unique()

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

                # EDINETデータ取得（Point-in-Time）
                current, yoy = self.get_yoy_edinet(code, target_date)

                if current is None:
                    continue

                # 特徴量算出
                features = self._calc_features(code, target_date, current, yoy, price)
                results.append(features)

        if len(results) == 0:
            return pd.DataFrame()

        result_df = pd.DataFrame(results)
        logger.info(f"EDINET詳細財務指標算出完了: {len(result_df)}行")
        return result_df

    def _calc_features(self, code: str, target_date: date,
                       current: pd.Series, yoy: Optional[pd.Series],
                       price: float) -> dict:
        """EDINET詳細財務指標を算出"""
        features = {
            'code': code,
            'date': target_date,
        }

        # === 研究開発関連指標（成長銘柄発掘の重要指標）===
        # 研究開発費比率（売上高に対する割合）
        if self._is_valid(current.get('rd_expenses')) and self._is_valid(current.get('net_sales')):
            if current['net_sales'] > 0:
                features['rd_intensity'] = current['rd_expenses'] / current['net_sales']

        # 研究開発費成長率（YoY）
        if yoy is not None:
            if self._is_valid(current.get('rd_expenses')) and self._is_valid(yoy.get('rd_expenses')):
                if yoy['rd_expenses'] > 0:
                    features['rd_growth_yoy'] = (
                        current['rd_expenses'] - yoy['rd_expenses']
                    ) / yoy['rd_expenses']

        # === 設備投資関連指標 ===
        # 設備投資比率（売上高に対する割合）
        if self._is_valid(current.get('capex')) and self._is_valid(current.get('net_sales')):
            if current['net_sales'] > 0:
                features['capex_intensity'] = abs(current['capex']) / current['net_sales']

        # 設備投資成長率（YoY）
        if yoy is not None:
            if self._is_valid(current.get('capex')) and self._is_valid(yoy.get('capex')):
                if abs(yoy['capex']) > 0:
                    features['capex_growth_yoy'] = (
                        abs(current['capex']) - abs(yoy['capex'])
                    ) / abs(yoy['capex'])

        # 設備投資 / 減価償却費（拡大投資の指標）
        if self._is_valid(current.get('capex')) and self._is_valid(current.get('depreciation')):
            if current['depreciation'] > 0:
                features['capex_to_depreciation'] = abs(current['capex']) / current['depreciation']

        # === 売上原価・粗利関連 ===
        # 粗利率
        if self._is_valid(current.get('gross_profit')) and self._is_valid(current.get('net_sales')):
            if current['net_sales'] > 0:
                features['gross_margin'] = current['gross_profit'] / current['net_sales']

        # 粗利率変化（YoY）
        if yoy is not None:
            if (self._is_valid(current.get('gross_profit')) and
                self._is_valid(current.get('net_sales')) and
                self._is_valid(yoy.get('gross_profit')) and
                self._is_valid(yoy.get('net_sales'))):

                current_gm = current['gross_profit'] / current['net_sales'] if current['net_sales'] > 0 else 0
                yoy_gm = yoy['gross_profit'] / yoy['net_sales'] if yoy['net_sales'] > 0 else 0
                features['gross_margin_change'] = current_gm - yoy_gm

        # 売上原価率
        if self._is_valid(current.get('cost_of_sales')) and self._is_valid(current.get('net_sales')):
            if current['net_sales'] > 0:
                features['cost_of_sales_ratio'] = current['cost_of_sales'] / current['net_sales']

        # === 販管費関連 ===
        # 販管費率
        if self._is_valid(current.get('sga_expenses')) and self._is_valid(current.get('net_sales')):
            if current['net_sales'] > 0:
                features['sga_ratio'] = current['sga_expenses'] / current['net_sales']

        # 販管費率変化（YoY）
        if yoy is not None:
            if (self._is_valid(current.get('sga_expenses')) and
                self._is_valid(current.get('net_sales')) and
                self._is_valid(yoy.get('sga_expenses')) and
                self._is_valid(yoy.get('net_sales'))):

                current_sga = current['sga_expenses'] / current['net_sales'] if current['net_sales'] > 0 else 0
                yoy_sga = yoy['sga_expenses'] / yoy['net_sales'] if yoy['net_sales'] > 0 else 0
                features['sga_ratio_change'] = current_sga - yoy_sga

        # === 財務健全性（詳細）===
        # 有利子負債比率（総資産に対する割合）
        if self._is_valid(current.get('interest_bearing_debt')) and self._is_valid(current.get('total_assets')):
            if current['total_assets'] > 0:
                features['debt_to_assets'] = current['interest_bearing_debt'] / current['total_assets']

        # 有利子負債/株主資本（D/E レシオ詳細版）
        if self._is_valid(current.get('interest_bearing_debt')) and self._is_valid(current.get('shareholders_equity')):
            if current['shareholders_equity'] > 0:
                features['debt_equity_ratio'] = current['interest_bearing_debt'] / current['shareholders_equity']

        # 株主資本比率
        if self._is_valid(current.get('shareholders_equity')) and self._is_valid(current.get('total_assets')):
            if current['total_assets'] > 0:
                features['shareholders_equity_ratio'] = current['shareholders_equity'] / current['total_assets']

        # === キャッシュフロー詳細 ===
        # 営業CFマージン
        if self._is_valid(current.get('cf_operating')) and self._is_valid(current.get('net_sales')):
            if current['net_sales'] > 0:
                features['ocf_margin'] = current['cf_operating'] / current['net_sales']

        # フリーキャッシュフロー（営業CF + 投資CF）
        if self._is_valid(current.get('cf_operating')) and self._is_valid(current.get('cf_investing')):
            features['free_cash_flow'] = current['cf_operating'] + current['cf_investing']

            # FCFマージン
            if self._is_valid(current.get('net_sales')) and current['net_sales'] > 0:
                features['fcf_margin'] = (current['cf_operating'] + current['cf_investing']) / current['net_sales']

        # 財務CF比率（有利子負債の返済状況）
        if self._is_valid(current.get('cf_financing')) and self._is_valid(current.get('net_sales')):
            if current['net_sales'] > 0:
                features['financing_cf_ratio'] = current['cf_financing'] / current['net_sales']

        # === 従業員関連 ===
        # 従業員一人当たり売上高
        if self._is_valid(current.get('employees')) and self._is_valid(current.get('net_sales')):
            if current['employees'] > 0:
                features['sales_per_employee'] = current['net_sales'] / current['employees']

        # 従業員一人当たり営業利益
        if self._is_valid(current.get('employees')) and self._is_valid(current.get('operating_income')):
            if current['employees'] > 0:
                features['profit_per_employee'] = current['operating_income'] / current['employees']

        # 従業員数成長率（YoY）
        if yoy is not None:
            if self._is_valid(current.get('employees')) and self._is_valid(yoy.get('employees')):
                if yoy['employees'] > 0:
                    features['employee_growth_yoy'] = (
                        current['employees'] - yoy['employees']
                    ) / yoy['employees']

        # === 成長性詳細指標（EDINET特有）===
        if yoy is not None:
            # 営業利益成長率（EDINET版）
            if self._is_valid(current.get('operating_income')) and self._is_valid(yoy.get('operating_income')):
                if abs(yoy['operating_income']) > 0:
                    features['edinet_op_growth_yoy'] = (
                        current['operating_income'] - yoy['operating_income']
                    ) / abs(yoy['operating_income'])

            # 経常利益成長率（EDINET版）
            if self._is_valid(current.get('ordinary_income')) and self._is_valid(yoy.get('ordinary_income')):
                if abs(yoy['ordinary_income']) > 0:
                    features['edinet_ordinary_growth_yoy'] = (
                        current['ordinary_income'] - yoy['ordinary_income']
                    ) / abs(yoy['ordinary_income'])

            # 純利益成長率（EDINET版）
            if self._is_valid(current.get('net_income')) and self._is_valid(yoy.get('net_income')):
                if abs(yoy['net_income']) > 0:
                    features['edinet_net_income_growth_yoy'] = (
                        current['net_income'] - yoy['net_income']
                    ) / abs(yoy['net_income'])

        return features

    def _is_valid(self, value) -> bool:
        """値が有効かチェック"""
        if value is None:
            return False
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            return False
        return True


def calculate_rd_sector_relative(df: pd.DataFrame,
                                 sector_col: str = 'sector_33_code') -> pd.DataFrame:
    """
    セクター内相対R&D指標を算出

    Args:
        df: 特徴量DataFrame（rd_intensity等を含む）
        sector_col: セクターカラム名

    Returns:
        相対R&D指標を追加したDataFrame
    """
    if sector_col not in df.columns:
        logger.warning(f"{sector_col}カラムが存在しません")
        return df

    result = df.copy()

    if 'rd_intensity' in result.columns:
        # セクター中央値
        sector_median = result.groupby(['date', sector_col])['rd_intensity'].transform('median')

        # 相対R&D投資強度
        result['rd_intensity_relative_sector'] = result['rd_intensity'] / sector_median.replace(0, np.nan) - 1

        # セクター内パーセンタイル
        result['rd_intensity_rank_sector'] = result.groupby(['date', sector_col])['rd_intensity'].transform(
            lambda x: x.rank(pct=True)
        )

    return result
