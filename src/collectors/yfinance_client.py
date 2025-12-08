"""
yfinance ラッパー

株価データのバックアップ・補完用
海外指数（S&P500等）の取得
"""
import yfinance as yf
import pandas as pd
from datetime import date, datetime
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)


class YFinanceClient:
    """yfinance ラッパー"""

    # 海外指数シンボル
    INDICES = {
        "sp500": "^GSPC",
        "nasdaq": "^IXIC",
        "dow": "^DJI",
        "vix": "^VIX",
        "us10y": "^TNX",      # 米10年国債利回り
        "dxy": "DX-Y.NYB",    # ドルインデックス
        "wti": "CL=F",        # WTI原油
        "gold": "GC=F",       # 金
    }

    # 日本株のサフィックス
    JP_SUFFIX = ".T"

    def __init__(self):
        pass

    def get_jp_stock_prices(self, code: str, start_date: date,
                            end_date: date) -> Optional[pd.DataFrame]:
        """
        日本株の株価を取得

        Args:
            code: 証券コード（4桁）
            start_date: 開始日
            end_date: 終了日

        Returns:
            株価DataFrame
        """
        symbol = f"{code}{self.JP_SUFFIX}"

        try:
            df = yf.download(
                symbol,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                progress=False
            )

            if df.empty:
                return None

            df["code"] = code
            return df.reset_index()

        except Exception as e:
            logger.warning(f"yfinance error for {symbol}: {e}")
            return None

    def get_index_prices(self, index_name: str, start_date: date,
                         end_date: date) -> Optional[pd.DataFrame]:
        """
        指数価格を取得

        Args:
            index_name: 指数名（"sp500", "vix" 等）
            start_date: 開始日
            end_date: 終了日

        Returns:
            指数DataFrame
        """
        symbol = self.INDICES.get(index_name)
        if not symbol:
            logger.warning(f"Unknown index: {index_name}")
            return None

        try:
            df = yf.download(
                symbol,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                progress=False
            )

            if df.empty:
                return None

            df["index_name"] = index_name
            return df.reset_index()

        except Exception as e:
            logger.warning(f"yfinance error for {symbol}: {e}")
            return None

    def get_all_indices(self, start_date: date,
                        end_date: date) -> Dict[str, pd.DataFrame]:
        """全指数を取得"""
        results = {}

        for name in self.INDICES.keys():
            df = self.get_index_prices(name, start_date, end_date)
            if df is not None:
                results[name] = df

        return results

    def get_fx_rate(self, pair: str, start_date: date,
                    end_date: date) -> Optional[pd.DataFrame]:
        """
        為替レートを取得

        Args:
            pair: 通貨ペア（"USDJPY", "EURJPY" 等）
        """
        symbol = f"{pair}=X"

        try:
            df = yf.download(
                symbol,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                progress=False
            )
            return df.reset_index() if not df.empty else None

        except Exception as e:
            logger.warning(f"yfinance FX error for {pair}: {e}")
            return None
