"""
EDINET API クライアント

金融庁 EDINET API v2 対応
https://disclosure2.edinet-fsa.go.jp/weee0010.aspx
"""
import requests
import zipfile
import io
from typing import List, Dict, Optional
from datetime import date, datetime, timedelta
import time
import logging

logger = logging.getLogger(__name__)


class EdinetClient:
    """EDINET API クライアント"""

    BASE_URL = "https://api.edinet-fsa.go.jp/api/v2"

    # 取得対象の書類種別コード
    DOC_TYPES = {
        "annual_report": "120",           # 有価証券報告書
        "quarterly_report": "140",        # 四半期報告書
        "earnings_report": "160",         # 決算短信（サマリー）
    }

    # 提出者業種（銀行・保険等は会計基準が異なる）
    EXCLUDE_INDUSTRIES = [
        "銀行業", "保険業", "証券、商品先物取引業"
    ]

    def __init__(self, api_interval: float = 1.0):
        """
        Args:
            api_interval: API呼び出し間隔（秒）
        """
        self.api_interval = api_interval
        self._last_request_time = 0

    def _wait_for_rate_limit(self):
        """レートリミット対策"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.api_interval:
            time.sleep(self.api_interval - elapsed)
        self._last_request_time = time.time()

    def get_document_list(self, target_date: date,
                          doc_type_code: str = "2") -> List[Dict]:
        """
        指定日の開示書類一覧を取得

        Args:
            target_date: 対象日
            doc_type_code: "1"=メタデータのみ, "2"=書類一覧

        Returns:
            書類情報のリスト
        """
        self._wait_for_rate_limit()

        url = f"{self.BASE_URL}/documents.json"
        params = {
            "date": target_date.strftime("%Y-%m-%d"),
            "type": doc_type_code
        }

        response = requests.get(url, params=params)
        response.raise_for_status()

        data = response.json()

        if data.get("metadata", {}).get("status") != "200":
            logger.warning(f"EDINET API warning: {data.get('metadata')}")
            return []

        return data.get("results", [])

    def filter_target_documents(self, documents: List[Dict],
                                 doc_types: List[str] = None) -> List[Dict]:
        """
        対象書類をフィルタリング

        Args:
            documents: 書類一覧
            doc_types: 対象書類種別 ["120", "140"] など

        Returns:
            フィルタ後の書類リスト
        """
        if doc_types is None:
            doc_types = [self.DOC_TYPES["annual_report"],
                        self.DOC_TYPES["quarterly_report"]]

        filtered = []
        for doc in documents:
            # 書類種別チェック
            if doc.get("docTypeCode") not in doc_types:
                continue

            # 有価証券報告書等のみ（ordinanceCode="010"=金商法）
            if doc.get("ordinanceCode") != "010":
                continue

            # EDINETコード（企業識別子）があること
            if not doc.get("edinetCode"):
                continue

            # XBRLがあること
            if doc.get("xbrlFlag") != "1":
                continue

            filtered.append(doc)

        return filtered

    def download_xbrl_zip(self, doc_id: str) -> Optional[bytes]:
        """
        XBRL ZIPファイルをダウンロード

        Args:
            doc_id: 書類管理番号

        Returns:
            ZIPファイルのバイト列（失敗時はNone）
        """
        self._wait_for_rate_limit()

        url = f"{self.BASE_URL}/documents/{doc_id}"
        params = {"type": "1"}  # 1=XBRL

        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()

            if response.headers.get("Content-Type", "").startswith("application/json"):
                # エラーレスポンス
                logger.warning(f"XBRL download failed for {doc_id}: {response.json()}")
                return None

            return response.content

        except Exception as e:
            logger.error(f"XBRL download error for {doc_id}: {e}")
            return None

    def get_documents_for_period(self, start_date: date, end_date: date,
                                  doc_types: List[str] = None) -> List[Dict]:
        """
        期間指定で書類一覧を取得

        Args:
            start_date: 開始日
            end_date: 終了日
            doc_types: 対象書類種別

        Returns:
            書類情報のリスト（重複排除済み）
        """
        all_docs = []
        seen_doc_ids = set()

        current = start_date
        while current <= end_date:
            logger.info(f"EDINET: {current} の書類一覧を取得中...")

            docs = self.get_document_list(current)
            filtered = self.filter_target_documents(docs, doc_types)

            for doc in filtered:
                doc_id = doc.get("docID")
                if doc_id and doc_id not in seen_doc_ids:
                    seen_doc_ids.add(doc_id)
                    all_docs.append(doc)

            current += timedelta(days=1)

        logger.info(f"EDINET: {len(all_docs)} 件の書類を取得")
        return all_docs
