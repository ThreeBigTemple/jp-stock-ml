"""
Google Trends クライアント

企業名・銘柄の検索トレンドを取得
"""
from pytrends.request import TrendReq
import pandas as pd
from datetime import date, timedelta
from typing import List, Optional, Dict
import time
import logging

logger = logging.getLogger(__name__)


class GoogleTrendsClient:
    """Google Trends クライアント"""

    def __init__(self, language: str = "ja-JP", timezone: int = 540):
        """
        Args:
            language: 言語コード
            timezone: タイムゾーン（分）、日本=540
        """
        self.pytrends = TrendReq(hl=language, tz=timezone)
        self._last_request_time = 0
        self.api_interval = 2.0  # Google Trendsは制限が厳しい

    def _wait_for_rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self.api_interval:
            time.sleep(self.api_interval - elapsed)
        self._last_request_time = time.time()

    def get_interest_over_time(self, keywords: List[str],
                                start_date: date,
                                end_date: date) -> Optional[pd.DataFrame]:
        """
        キーワードの検索トレンドを取得

        Args:
            keywords: 検索キーワード（最大5個）
            start_date: 開始日
            end_date: 終了日

        Returns:
            検索ボリュームのDataFrame（週次）
        """
        if len(keywords) > 5:
            logger.warning("Google Trends supports max 5 keywords")
            keywords = keywords[:5]

        self._wait_for_rate_limit()

        timeframe = f"{start_date.strftime('%Y-%m-%d')} {end_date.strftime('%Y-%m-%d')}"

        try:
            self.pytrends.build_payload(
                keywords,
                cat=0,  # カテゴリ: 全て
                timeframe=timeframe,
                geo="JP",  # 日本
            )

            df = self.pytrends.interest_over_time()

            if df.empty:
                return None

            # isPartialカラムを除去
            if "isPartial" in df.columns:
                df = df.drop(columns=["isPartial"])

            return df.reset_index()

        except Exception as e:
            logger.warning(f"Google Trends error: {e}")
            return None

    def get_related_queries(self, keyword: str) -> Dict[str, pd.DataFrame]:
        """関連クエリを取得"""
        self._wait_for_rate_limit()

        try:
            self.pytrends.build_payload([keyword], geo="JP")
            return self.pytrends.related_queries()
        except Exception as e:
            logger.warning(f"Related queries error: {e}")
            return {}

    def get_stock_trend(self, company_name: str, stock_code: str,
                        start_date: date, end_date: date) -> Optional[pd.DataFrame]:
        """
        銘柄の検索トレンドを取得

        企業名と証券コードの両方で検索し、合算
        """
        keywords = [company_name, f"{stock_code} 株価"]
        return self.get_interest_over_time(keywords, start_date, end_date)
