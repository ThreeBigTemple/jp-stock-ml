"""
統合データコレクター

全データソースを統合管理
"""
from datetime import date, timedelta
from typing import Optional
import logging

from sqlalchemy.orm import Session

from .jquants_collector import JQuantsCollector
from .edinet_collector import EdinetCollector
from .tdnet_collector import TdnetCollector
from .yfinance_client import YFinanceClient

logger = logging.getLogger(__name__)


class MasterCollector:
    """統合データコレクター"""

    def __init__(self, session: Session,
                 jquants_collector: JQuantsCollector,
                 edinet_collector: Optional[EdinetCollector] = None,
                 tdnet_collector: Optional[TdnetCollector] = None,
                 yfinance_client: Optional[YFinanceClient] = None):

        self.session = session
        self.jquants = jquants_collector
        self.edinet = edinet_collector
        self.tdnet = tdnet_collector
        self.yfinance = yfinance_client

    def collect_all_historical(self, years: int = 10):
        """
        全データソースから過去データを一括収集

        Args:
            years: 取得年数
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=years * 365)

        from_str = start_date.strftime("%Y-%m-%d")
        to_str = end_date.strftime("%Y-%m-%d")

        logger.info(f"=== 全データソース一括収集開始 ({from_str} ~ {to_str}) ===")

        # 1. J-Quants（メイン）
        logger.info("\n[1/4] J-Quants データ収集...")
        self.jquants.collect_all_historical(years)

        # 2. EDINET（詳細財務）
        if self.edinet:
            logger.info("\n[2/4] EDINET データ収集...")
            self.edinet.collect_financials(start_date, end_date)
        else:
            logger.info("\n[2/4] EDINET スキップ（未設定）")

        # 3. TDnet（適時開示）
        if self.tdnet:
            logger.info("\n[3/4] TDnet データ収集...")
            self.tdnet.collect_disclosures(start_date, end_date)
        else:
            logger.info("\n[3/4] TDnet スキップ（未設定）")

        # 4. 海外指数（yfinance）
        if self.yfinance:
            logger.info("\n[4/4] 海外指数データ収集...")
            self._collect_global_indices(start_date, end_date)
        else:
            logger.info("\n[4/4] 海外指数スキップ（未設定）")

        logger.info("\n=== 全データソース一括収集完了 ===")

    def _collect_global_indices(self, start_date: date, end_date: date):
        """海外指数を収集"""
        from ..database.models import GlobalIndex
        from sqlalchemy.dialects.sqlite import insert

        indices = self.yfinance.get_all_indices(start_date, end_date)

        count = 0
        for index_name, df in indices.items():
            for _, row in df.iterrows():
                record = {
                    "index_name": index_name,
                    "date": row["Date"].date() if hasattr(row["Date"], "date") else row["Date"],
                    "open": row.get("Open"),
                    "high": row.get("High"),
                    "low": row.get("Low"),
                    "close": row.get("Close"),
                    "volume": row.get("Volume"),
                }

                stmt = insert(GlobalIndex).values(**record)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["index_name", "date"],
                    set_={k: v for k, v in record.items() if k not in ["index_name", "date"]}
                )
                self.session.execute(stmt)
                count += 1

        self.session.commit()
        logger.info(f"海外指数: {count}件 保存完了")

    def daily_update(self, days: int = 7):
        """
        日次更新

        Args:
            days: 更新日数（バッファ）
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=days)

        logger.info(f"=== 日次更新開始 ({start_date} ~ {end_date}) ===")

        # J-Quants
        self.jquants.collect_stocks()
        self.jquants.collect_prices(
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )
        self.jquants.collect_financials()

        # EDINET（直近のみ）
        if self.edinet:
            self.edinet.collect_financials(start_date, end_date)

        # TDnet
        if self.tdnet:
            self.tdnet.collect_disclosures(start_date, end_date)

        # 海外指数
        if self.yfinance:
            self._collect_global_indices(start_date, end_date)

        logger.info("=== 日次更新完了 ===")
