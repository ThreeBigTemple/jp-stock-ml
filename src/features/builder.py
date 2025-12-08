"""
特徴量マトリクス構築モジュール

各種特徴量を統合し、銘柄×日付のDataFrameを生成する
EDINET詳細財務、TDnet適時開示、海外指数、Google Trendsに対応
"""
import numpy as np
import pandas as pd
from datetime import date, timedelta
from typing import Optional, List
import logging

from sqlalchemy.orm import Session

from .technical import TechnicalFeatures, calculate_sector_rank
from .fundamental import FundamentalFeatures, calculate_sector_relative_valuation
from .market import MarketFeatures, merge_market_features
from .edinet_features import EdinetFeatures, calculate_rd_sector_relative
from .disclosure_features import DisclosureFeatures
from .trends_features import TrendsFeatures, merge_trends_features, calculate_relative_search_interest

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """特徴量マトリクス構築クラス"""

    def __init__(self, session: Session):
        """
        初期化

        Args:
            session: SQLAlchemyセッション
        """
        self.session = session

    def load_data(
        self,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        codes: Optional[List[str]] = None,
        include_edinet: bool = True,
        include_disclosure: bool = True,
        include_global_indices: bool = True,
        include_trends: bool = False  # Google Trendsはデフォルトオフ（ノイズが多い）
    ) -> dict:
        """
        必要なデータをDBから読み込み

        Args:
            from_date: 開始日（YYYY-MM-DD）
            to_date: 終了日（YYYY-MM-DD）
            codes: 対象銘柄コードリスト
            include_edinet: EDINET詳細財務を含むか
            include_disclosure: TDnet適時開示を含むか
            include_global_indices: 海外指数を含むか
            include_trends: Google Trendsを含むか

        Returns:
            各種データのDataFrame辞書
        """
        from ..database.models import (
            Stock, Price, Financial, Topix, MarginBalance, ShortSelling,
            EdinetFinancial, Disclosure, GlobalIndex, SearchTrend
        )

        logger.info("データ読み込み開始...")

        # 株価データ
        price_query = self.session.query(Price)
        if from_date:
            price_query = price_query.filter(Price.date >= from_date)
        if to_date:
            price_query = price_query.filter(Price.date <= to_date)
        if codes:
            price_query = price_query.filter(Price.code.in_(codes))

        prices_df = pd.read_sql(price_query.statement, self.session.bind)
        logger.info(f"株価データ: {len(prices_df)}行")

        # 銘柄マスタ
        stocks_df = pd.read_sql(self.session.query(Stock).statement, self.session.bind)
        logger.info(f"銘柄マスタ: {len(stocks_df)}行")

        # 財務データ（J-Quants）
        fin_query = self.session.query(Financial)
        if codes:
            fin_query = fin_query.filter(Financial.code.in_(codes))
        financials_df = pd.read_sql(fin_query.statement, self.session.bind)
        logger.info(f"財務データ: {len(financials_df)}行")

        # TOPIXデータ
        topix_query = self.session.query(Topix)
        if from_date:
            topix_query = topix_query.filter(Topix.date >= from_date)
        if to_date:
            topix_query = topix_query.filter(Topix.date <= to_date)
        topix_df = pd.read_sql(topix_query.statement, self.session.bind)
        logger.info(f"TOPIXデータ: {len(topix_df)}行")

        # 信用取引残高
        margin_query = self.session.query(MarginBalance)
        if from_date:
            margin_query = margin_query.filter(MarginBalance.date >= from_date)
        if to_date:
            margin_query = margin_query.filter(MarginBalance.date <= to_date)
        if codes:
            margin_query = margin_query.filter(MarginBalance.code.in_(codes))
        margin_df = pd.read_sql(margin_query.statement, self.session.bind)
        logger.info(f"信用取引残高: {len(margin_df)}行")

        # 空売り比率
        short_query = self.session.query(ShortSelling)
        if from_date:
            short_query = short_query.filter(ShortSelling.date >= from_date)
        if to_date:
            short_query = short_query.filter(ShortSelling.date <= to_date)
        short_selling_df = pd.read_sql(short_query.statement, self.session.bind)
        logger.info(f"空売り比率: {len(short_selling_df)}行")

        result = {
            'prices': prices_df,
            'stocks': stocks_df,
            'financials': financials_df,
            'topix': topix_df,
            'margin': margin_df,
            'short_selling': short_selling_df,
        }

        # === 新規データソース ===

        # EDINET詳細財務データ
        if include_edinet:
            edinet_query = self.session.query(EdinetFinancial)
            if codes:
                edinet_query = edinet_query.filter(EdinetFinancial.code.in_(codes))
            edinet_df = pd.read_sql(edinet_query.statement, self.session.bind)
            result['edinet'] = edinet_df
            logger.info(f"EDINET財務データ: {len(edinet_df)}行")

        # TDnet適時開示データ
        if include_disclosure:
            disclosure_query = self.session.query(Disclosure)
            if from_date:
                # 開示はルックバック期間が必要なので、少し前から取得
                lookback_date = pd.to_datetime(from_date) - pd.Timedelta(days=90)
                disclosure_query = disclosure_query.filter(Disclosure.date >= lookback_date)
            if to_date:
                disclosure_query = disclosure_query.filter(Disclosure.date <= to_date)
            if codes:
                disclosure_query = disclosure_query.filter(Disclosure.code.in_(codes))
            disclosure_df = pd.read_sql(disclosure_query.statement, self.session.bind)
            result['disclosure'] = disclosure_df
            logger.info(f"TDnet適時開示: {len(disclosure_df)}行")

        # 海外指数データ
        if include_global_indices:
            global_query = self.session.query(GlobalIndex)
            if from_date:
                global_query = global_query.filter(GlobalIndex.date >= from_date)
            if to_date:
                global_query = global_query.filter(GlobalIndex.date <= to_date)
            global_df = pd.read_sql(global_query.statement, self.session.bind)
            result['global_indices'] = global_df
            logger.info(f"海外指数: {len(global_df)}行")

        # Google Trendsデータ
        if include_trends:
            trends_query = self.session.query(SearchTrend)
            if codes:
                trends_query = trends_query.filter(SearchTrend.code.in_(codes))
            trends_df = pd.read_sql(trends_query.statement, self.session.bind)
            result['trends'] = trends_df
            logger.info(f"Google Trends: {len(trends_df)}行")

        return result

    def build_features(
        self,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        codes: Optional[List[str]] = None,
        include_technical: bool = True,
        include_fundamental: bool = True,
        include_market: bool = True,
        include_edinet: bool = True,
        include_disclosure: bool = True,
        include_global_indices: bool = True,
        include_trends: bool = False
    ) -> pd.DataFrame:
        """
        特徴量マトリクスを構築

        Args:
            from_date: 開始日
            to_date: 終了日
            codes: 対象銘柄コードリスト
            include_technical: テクニカル指標を含むか
            include_fundamental: ファンダメンタル指標を含むか
            include_market: 市場・需給指標を含むか
            include_edinet: EDINET詳細財務指標を含むか
            include_disclosure: TDnet適時開示シグナルを含むか
            include_global_indices: 海外指数特徴量を含むか
            include_trends: Google Trends特徴量を含むか

        Returns:
            特徴量マトリクス（columns: code, date, feature_1, ...）
        """
        # データ読み込み
        data = self.load_data(
            from_date, to_date, codes,
            include_edinet=include_edinet,
            include_disclosure=include_disclosure,
            include_global_indices=include_global_indices,
            include_trends=include_trends
        )

        if len(data['prices']) == 0:
            logger.warning("株価データがありません")
            return pd.DataFrame()

        result = data['prices'].copy()

        # 銘柄マスタからセクター情報を追加
        if 'sector_33_code' not in result.columns:
            sector_map = data['stocks'][['code', 'sector_33_code', 'sector_33_name', 'market_name']]
            result = result.merge(sector_map, on='code', how='left')

        # === テクニカル指標 ===
        if include_technical and len(data['prices']) > 0:
            logger.info("テクニカル指標算出中...")
            tech_features = TechnicalFeatures(
                prices_df=data['prices'],
                topix_df=data['topix']
            )
            result = tech_features.calculate_all()

            # セクター情報を再追加
            if 'sector_33_code' not in result.columns:
                result = result.merge(
                    data['stocks'][['code', 'sector_33_code', 'sector_33_name', 'market_name']],
                    on='code', how='left'
                )

            # セクター内順位
            result = calculate_sector_rank(result, 'sector_33_code')

        # === ファンダメンタル指標（J-Quants）===
        if include_fundamental and len(data['financials']) > 0:
            logger.info("ファンダメンタル指標算出中...")
            fund_features = FundamentalFeatures(
                financials_df=data['financials'],
                prices_df=data['prices']
            )

            # 対象日付を取得
            target_dates = result['date'].unique().tolist()
            fund_df = fund_features.calculate_all(target_dates)

            if len(fund_df) > 0:
                # マージ
                result = result.merge(fund_df, on=['code', 'date'], how='left')

                # セクター相対バリュエーション
                result = calculate_sector_relative_valuation(result, 'sector_33_code')

        # === EDINET詳細財務指標 ===
        if include_edinet and 'edinet' in data and len(data['edinet']) > 0:
            logger.info("EDINET詳細財務指標算出中...")
            edinet_features = EdinetFeatures(
                edinet_df=data['edinet'],
                prices_df=data['prices']
            )

            target_dates = result['date'].unique().tolist()
            edinet_df = edinet_features.calculate_all(target_dates)

            if len(edinet_df) > 0:
                result = result.merge(edinet_df, on=['code', 'date'], how='left')

                # セクター相対R&D指標
                result = calculate_rd_sector_relative(result, 'sector_33_code')

        # === TDnet適時開示シグナル ===
        if include_disclosure and 'disclosure' in data and len(data['disclosure']) > 0:
            logger.info("TDnet適時開示シグナル算出中...")
            disclosure_features = DisclosureFeatures(disclosure_df=data['disclosure'])

            disclosure_df = disclosure_features.calculate_all(prices_df=data['prices'])

            if len(disclosure_df) > 0:
                result = result.merge(disclosure_df, on=['code', 'date'], how='left')

        # === 市場・需給指標（海外指数含む）===
        if include_market:
            logger.info("市場・需給指標算出中...")

            global_indices_df = data.get('global_indices') if include_global_indices else None

            market_features = MarketFeatures(
                margin_df=data['margin'],
                short_selling_df=data['short_selling'],
                topix_df=data['topix'],
                global_indices_df=global_indices_df
            )
            market_dict = market_features.calculate_all()

            result = merge_market_features(
                prices_df=result,
                margin_features=market_dict['margin'],
                short_selling_features=market_dict['short_selling'],
                topix_features=market_dict['topix'],
                stocks_df=data['stocks'],
                global_index_features=market_dict.get('global_indices')
            )

        # === Google Trends特徴量 ===
        if include_trends and 'trends' in data and len(data['trends']) > 0:
            logger.info("Google Trends特徴量算出中...")
            trends_features = TrendsFeatures(trends_df=data['trends'])

            trends_df = trends_features.calculate_all(prices_df=data['prices'])

            if len(trends_df) > 0:
                result = merge_trends_features(result, trends_df)

                # セクター相対検索関心度
                result = calculate_relative_search_interest(result, 'sector_33_code')

        # 不要なカラムを削除
        drop_cols = ['id', 'updated_at']
        result = result.drop(columns=[c for c in drop_cols if c in result.columns])

        logger.info(f"特徴量マトリクス構築完了: {len(result)}行, {len(result.columns)}列")
        return result

    def build_training_data(
        self,
        from_date: str,
        to_date: str,
        target_days: int = 20,
        target_type: str = 'return',
        codes: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        学習用データセットを構築

        Args:
            from_date: 開始日
            to_date: 終了日
            target_days: ターゲット期間（日数）
            target_type: ターゲット種別（'return': リターン, 'binary': 上昇フラグ）
            codes: 対象銘柄コードリスト

        Returns:
            学習用DataFrame（columns: code, date, features..., target）
        """
        # 特徴量構築
        features_df = self.build_features(from_date, to_date, codes)

        if len(features_df) == 0:
            return pd.DataFrame()

        # ターゲット変数を算出
        logger.info(f"ターゲット変数算出中... ({target_days}日後{target_type})")

        result = features_df.copy()

        # 銘柄ごとにN日後リターンを算出
        result = result.sort_values(['code', 'date'])

        target_list = []
        for code, group in result.groupby('code'):
            group = group.sort_values('date').copy()

            # N日後の終値
            future_close = group['adjustment_close'].shift(-target_days)

            # リターン
            group['target_return'] = future_close / group['adjustment_close'] - 1

            # 上昇フラグ
            group['target_binary'] = (group['target_return'] > 0).astype(int)

            target_list.append(group)

        result = pd.concat(target_list, ignore_index=True)

        # ターゲット列を選択
        if target_type == 'return':
            result['target'] = result['target_return']
        else:
            result['target'] = result['target_binary']

        # ターゲットがNaN（未来データがない）の行を削除
        result = result.dropna(subset=['target'])

        # 中間カラムを削除
        result = result.drop(columns=['target_return', 'target_binary'], errors='ignore')

        logger.info(f"学習用データセット構築完了: {len(result)}行")
        return result


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    特徴量カラムのリストを取得

    Args:
        df: 特徴量DataFrame

    Returns:
        特徴量カラム名のリスト
    """
    exclude_cols = {
        'code', 'date', 'target', 'target_return', 'target_binary',
        'company_name', 'company_name_english',
        'sector_33_name', 'sector_17_name', 'market_name',
        'listing_date', 'delisting_date', 'is_active',
        'open', 'high', 'low', 'close', 'volume', 'turnover_value',
        'adjustment_factor', 'adjustment_open', 'adjustment_high',
        'adjustment_low', 'adjustment_close', 'adjustment_volume',
        'id', 'updated_at'
    }

    feature_cols = [c for c in df.columns if c not in exclude_cols]
    return feature_cols


def clean_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    特徴量データのクリーニング

    Args:
        df: 特徴量DataFrame
        feature_cols: 特徴量カラムリスト

    Returns:
        クリーニング済みDataFrame
    """
    result = df.copy()

    # 無限値をNaNに置換
    for col in feature_cols:
        if col in result.columns:
            result[col] = result[col].replace([np.inf, -np.inf], np.nan)

    # 欠損値の割合を確認
    missing_rates = result[feature_cols].isnull().mean()
    high_missing = missing_rates[missing_rates > 0.5].index.tolist()

    if high_missing:
        logger.warning(f"欠損率50%超のカラム: {high_missing}")

    return result


def split_by_date(
    df: pd.DataFrame,
    train_end_date: str,
    val_end_date: Optional[str] = None
) -> tuple:
    """
    日付でデータを分割（ウォークフォワード検証用）

    Args:
        df: 特徴量DataFrame
        train_end_date: 学習データ終了日
        val_end_date: 検証データ終了日（省略時はテストデータなし）

    Returns:
        (train_df, val_df, test_df) のタプル
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    train_end = pd.to_datetime(train_end_date)

    train_df = df[df['date'] <= train_end]

    if val_end_date:
        val_end = pd.to_datetime(val_end_date)
        val_df = df[(df['date'] > train_end) & (df['date'] <= val_end)]
        test_df = df[df['date'] > val_end]
    else:
        val_df = df[df['date'] > train_end]
        test_df = pd.DataFrame()

    logger.info(f"データ分割: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    return train_df, val_df, test_df
