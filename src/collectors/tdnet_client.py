"""
TDnet スクレイパー

東証適時開示情報システムからデータを取得
https://www.release.tdnet.info/inbs/
"""
import requests
from bs4 import BeautifulSoup
import re
from datetime import date, datetime, timedelta
from typing import List, Dict, Optional
import time
import logging

logger = logging.getLogger(__name__)


class TdnetClient:
    """TDnet スクレイパー"""

    BASE_URL = "https://www.release.tdnet.info/inbs"

    # 対象開示タイプ
    DISCLOSURE_TYPES = {
        "earnings_revision": "業績予想の修正",
        "dividend_revision": "配当予想の修正",
        "buyback": "自己株式の取得",
        "buyback_result": "自己株式の取得結果",
        "stock_split": "株式分割",
        "new_stock": "新株式発行",
    }

    def __init__(self, api_interval: float = 1.0):
        self.api_interval = api_interval
        self._last_request_time = 0
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; research bot)"
        })

    def _wait_for_rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self.api_interval:
            time.sleep(self.api_interval - elapsed)
        self._last_request_time = time.time()

    def get_disclosures(self, target_date: date) -> List[Dict]:
        """
        指定日の開示一覧を取得

        Args:
            target_date: 対象日

        Returns:
            開示情報のリスト
        """
        self._wait_for_rate_limit()

        # TDnetの日付別一覧ページ
        date_str = target_date.strftime("%Y%m%d")
        url = f"{self.BASE_URL}/I_list_{date_str}.html"

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            response.encoding = 'utf-8'

            return self._parse_disclosure_list(response.text, target_date)

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                # 開示がない日
                return []
            raise

    def _parse_disclosure_list(self, html: str, target_date: date) -> List[Dict]:
        """開示一覧HTMLをパース"""
        soup = BeautifulSoup(html, 'html.parser')
        disclosures = []

        # テーブル行を解析
        for row in soup.select('tr'):
            cells = row.select('td')
            if len(cells) < 4:
                continue

            try:
                time_str = cells[0].get_text(strip=True)
                code = cells[1].get_text(strip=True)
                company = cells[2].get_text(strip=True)
                title = cells[3].get_text(strip=True)

                # PDFリンク取得
                link = cells[3].select_one('a')
                pdf_url = link.get('href') if link else None

                # 証券コード抽出（4桁）
                code_match = re.match(r'(\d{4})', code)
                if not code_match:
                    continue

                disclosure = {
                    "date": target_date,
                    "time": time_str,
                    "code": code_match.group(1),
                    "company_name": company,
                    "title": title,
                    "pdf_url": pdf_url,
                    "disclosure_type": self._classify_disclosure(title),
                }

                disclosures.append(disclosure)

            except Exception as e:
                logger.debug(f"Parse error: {e}")
                continue

        return disclosures

    def _classify_disclosure(self, title: str) -> Optional[str]:
        """開示タイトルからタイプを分類"""
        title_lower = title.lower()

        if "業績予想" in title and "修正" in title:
            return "earnings_revision"
        elif "配当予想" in title and "修正" in title:
            return "dividend_revision"
        elif "自己株式" in title and "取得" in title:
            if "結果" in title:
                return "buyback_result"
            return "buyback"
        elif "株式分割" in title:
            return "stock_split"
        elif "新株式発行" in title or "増資" in title:
            return "new_stock"

        return None

    def get_disclosures_for_period(self, start_date: date,
                                    end_date: date) -> List[Dict]:
        """期間指定で開示一覧を取得"""
        all_disclosures = []

        current = start_date
        while current <= end_date:
            # 土日はスキップ
            if current.weekday() < 5:
                disclosures = self.get_disclosures(current)
                all_disclosures.extend(disclosures)
                logger.debug(f"TDnet {current}: {len(disclosures)} 件")

            current += timedelta(days=1)

        return all_disclosures

    def parse_earnings_revision(self, pdf_url: str) -> Optional[Dict]:
        """
        業績予想修正PDFから修正率を抽出

        注: PDFパースは複雑なため、簡易実装ではタイトルのみ分析
        本格実装では pdfplumber 等を使用
        """
        # TODO: PDF解析実装
        pass
