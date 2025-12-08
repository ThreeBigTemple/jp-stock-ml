"""
EDINET データ収集ロジック
"""
from datetime import date, timedelta
from typing import List, Optional
from tqdm import tqdm
import logging

from sqlalchemy.orm import Session
from sqlalchemy.dialects.sqlite import insert

from .edinet_client import EdinetClient
from .edinet_parser import EdinetXbrlParser
from ..database.models import EdinetFinancial, Stock

logger = logging.getLogger(__name__)


class EdinetCollector:
    """EDINETデータ収集クラス"""

    def __init__(self, client: EdinetClient, parser: EdinetXbrlParser,
                 session: Session):
        self.client = client
        self.parser = parser
        self.session = session

    def collect_financials(self, start_date: date, end_date: date,
                           doc_types: List[str] = None) -> int:
        """
        期間指定で財務データを収集

        Args:
            start_date: 開始日
            end_date: 終了日
            doc_types: 書類種別コード

        Returns:
            収集件数
        """
        # 書類一覧取得
        documents = self.client.get_documents_for_period(
            start_date, end_date, doc_types
        )

        count = 0
        for doc in tqdm(documents, desc="EDINET財務データ収集"):
            doc_id = doc.get("docID")
            edinet_code = doc.get("edinetCode")

            # 証券コードを取得（EDINETコードから変換）
            stock_code = self._edinet_to_stock_code(edinet_code)
            if not stock_code:
                continue

            # XBRLダウンロード
            zip_content = self.client.download_xbrl_zip(doc_id)
            if not zip_content:
                continue

            # パース
            parsed = self.parser.parse_zip(zip_content)
            if not parsed:
                continue

            # DB保存
            self._save_financial(stock_code, doc, parsed)
            count += 1

        self.session.commit()
        logger.info(f"EDINET財務データ: {count}件 保存完了")
        return count

    def _edinet_to_stock_code(self, edinet_code: str) -> Optional[str]:
        """EDINETコードを証券コードに変換"""
        # stocksテーブルにEDINETコードがあれば検索
        # 簡易実装: J-QuantsのstocksテーブルにEDINET情報がない場合は
        # EDINET→証券コードのマッピングが必要
        # ここでは仮実装として、EDINETコードをそのまま証券コードとして扱う
        # 実際にはEDINETのAPIから取得した会社名で突合する必要がある

        # TODO: 実装が必要
        # 1. EDINETコードでstocksテーブルを検索
        # 2. 見つからない場合は会社名で検索
        # 3. それでも見つからない場合はスキップ

        logger.debug(f"EDINET code {edinet_code} mapping not implemented yet")
        return None

    def _save_financial(self, stock_code: str, doc: dict, parsed: dict):
        """財務データをDBに保存"""
        financial = {
            "code": stock_code,
            "edinet_code": doc.get("edinetCode"),
            "doc_id": doc.get("docID"),
            "doc_type_code": doc.get("docTypeCode"),
            "submit_date": doc.get("submitDateTime", "")[:10] if doc.get("submitDateTime") else None,
            "fiscal_year_end": doc.get("periodEnd"),

            # 損益計算書
            "net_sales": parsed.get("net_sales"),
            "cost_of_sales": parsed.get("cost_of_sales"),
            "gross_profit": parsed.get("gross_profit"),
            "sga_expenses": parsed.get("sga_expenses"),
            "rd_expenses": parsed.get("rd_expenses"),
            "operating_income": parsed.get("operating_income"),
            "ordinary_income": parsed.get("ordinary_income"),
            "net_income": parsed.get("net_income"),

            # 貸借対照表
            "total_assets": parsed.get("total_assets"),
            "net_assets": parsed.get("net_assets"),
            "shareholders_equity": parsed.get("shareholders_equity"),
            "interest_bearing_debt": parsed.get("interest_bearing_debt"),

            # キャッシュフロー
            "cf_operating": parsed.get("cf_operating"),
            "cf_investing": parsed.get("cf_investing"),
            "cf_financing": parsed.get("cf_financing"),

            # 投資・その他
            "capex": parsed.get("capex"),
            "depreciation": parsed.get("depreciation"),
            "employees": parsed.get("employees"),
        }

        stmt = insert(EdinetFinancial).values(**financial)
        stmt = stmt.on_conflict_do_update(
            index_elements=["code", "doc_id"],
            set_={k: v for k, v in financial.items() if k not in ["code", "doc_id"]}
        )
        self.session.execute(stmt)
