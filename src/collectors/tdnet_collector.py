"""
TDnet データ収集ロジック
"""
from datetime import date
from typing import List
from tqdm import tqdm
import logging

from sqlalchemy.orm import Session
from sqlalchemy.dialects.sqlite import insert

from .tdnet_client import TdnetClient
from ..database.models import Disclosure

logger = logging.getLogger(__name__)


class TdnetCollector:
    """TDnetデータ収集クラス"""

    def __init__(self, client: TdnetClient, session: Session):
        self.client = client
        self.session = session

    def collect_disclosures(self, start_date: date, end_date: date,
                            types: List[str] = None) -> int:
        """
        期間指定で開示情報を収集

        Args:
            start_date: 開始日
            end_date: 終了日
            types: 対象タイプ（Noneなら全て）

        Returns:
            収集件数
        """
        logger.info(f"TDnet開示情報を取得中... ({start_date} ~ {end_date})")

        disclosures = self.client.get_disclosures_for_period(start_date, end_date)

        # タイプフィルタ
        if types:
            disclosures = [d for d in disclosures if d.get("disclosure_type") in types]

        count = 0
        for disc in tqdm(disclosures, desc="TDnet保存"):
            record = {
                "code": disc["code"],
                "date": disc["date"],
                "time": disc["time"],
                "title": disc["title"],
                "disclosure_type": disc["disclosure_type"],
                "pdf_url": disc["pdf_url"],
            }

            stmt = insert(Disclosure).values(**record)
            stmt = stmt.on_conflict_do_update(
                index_elements=["code", "date", "title"],
                set_=record
            )
            self.session.execute(stmt)
            count += 1

        self.session.commit()
        logger.info(f"TDnet開示情報: {count}件 保存完了")
        return count
