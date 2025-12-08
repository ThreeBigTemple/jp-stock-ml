"""
J-Quants データ収集ロジック

APIからデータを取得してDBに保存
"""
import logging
from datetime import datetime, timedelta, date
from typing import Optional, List
from tqdm import tqdm

from sqlalchemy.orm import Session
from sqlalchemy.dialects.sqlite import insert

from .jquants_client import JQuantsClient
from ..database.models import (
    Stock, Price, Financial, TradingCalendar, Topix,
    MarginBalance, ShortSelling, InvestorTrades
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JQuantsCollector:
    """J-Quantsデータ収集クラス"""
    
    def __init__(self, client: JQuantsClient, session: Session):
        """
        初期化
        
        Args:
            client: J-Quants APIクライアント
            session: SQLAlchemyセッション
        """
        self.client = client
        self.session = session
    
    def _safe_float(self, value) -> Optional[float]:
        """安全なfloat変換"""
        if value is None or value == "" or value == "-":
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _safe_int(self, value) -> Optional[int]:
        """安全なint変換"""
        if value is None or value == "" or value == "-":
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None
    
    def _safe_date(self, value) -> Optional[date]:
        """安全なdate変換"""
        if value is None or value == "" or value == "-":
            return None
        try:
            if isinstance(value, date):
                return value
            return datetime.strptime(value, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            return None
    
    # ===================
    # 銘柄マスタ
    # ===================
    
    def collect_stocks(self, target_date: Optional[str] = None) -> int:
        """
        銘柄マスタを収集
        
        Args:
            target_date: 基準日（省略時は当日）
        
        Returns:
            取得件数
        """
        logger.info("銘柄マスタを取得中...")
        
        data = self.client.get_listed_info(date=target_date)
        
        count = 0
        for item in tqdm(data, desc="銘柄マスタ保存"):
            stock = Stock(
                code=item.get("Code"),
                company_name=item.get("CompanyName"),
                company_name_english=item.get("CompanyNameEnglish"),
                sector_17_code=item.get("Sector17Code"),
                sector_17_name=item.get("Sector17CodeName"),
                sector_33_code=item.get("Sector33Code"),
                sector_33_name=item.get("Sector33CodeName"),
                scale_category=item.get("ScaleCategory"),
                market_code=item.get("MarketCode"),
                market_name=item.get("MarketCodeName"),
                listing_date=self._safe_date(item.get("Date")),
                is_active=True,
            )
            
            # UPSERT
            self.session.merge(stock)
            count += 1
        
        self.session.commit()
        logger.info(f"銘柄マスタ: {count}件 保存完了")
        return count
    
    # ===================
    # 株価データ
    # ===================
    
    def collect_prices(self, from_date: str, to_date: str,
                       codes: Optional[List[str]] = None) -> int:
        """
        株価データを収集

        Args:
            from_date: 開始日（YYYY-MM-DD）
            to_date: 終了日（YYYY-MM-DD）
            codes: 銘柄コードリスト（省略時は全銘柄）

        Returns:
            取得件数
        """
        logger.info(f"株価データを取得中... ({from_date} ~ {to_date})")

        if codes:
            # 銘柄指定
            all_data = []
            for code in tqdm(codes, desc="銘柄別株価取得"):
                data = self.client.get_prices_daily(
                    code=code, from_date=from_date, to_date=to_date
                )
                all_data.extend(data)
        else:
            # 全銘柄（日付単位で取得 - J-Quants APIの制限対応）
            from datetime import datetime, timedelta
            start = datetime.strptime(from_date, "%Y-%m-%d")
            end = datetime.strptime(to_date, "%Y-%m-%d")

            all_data = []
            current = start
            date_list = []
            while current <= end:
                date_list.append(current.strftime("%Y-%m-%d"))
                current += timedelta(days=1)

            for date_str in tqdm(date_list, desc="日別株価取得"):
                try:
                    data = self.client.get_prices_daily(date=date_str)
                    all_data.extend(data)
                except Exception as e:
                    # 休日等でデータがない場合はスキップ
                    logger.debug(f"{date_str}: {e}")
        
        count = 0
        for item in tqdm(all_data, desc="株価保存"):
            price = {
                "code": item.get("Code"),
                "date": self._safe_date(item.get("Date")),
                "open": self._safe_float(item.get("Open")),
                "high": self._safe_float(item.get("High")),
                "low": self._safe_float(item.get("Low")),
                "close": self._safe_float(item.get("Close")),
                "volume": self._safe_float(item.get("Volume")),
                "turnover_value": self._safe_float(item.get("TurnoverValue")),
                "adjustment_factor": self._safe_float(item.get("AdjustmentFactor")),
                "adjustment_open": self._safe_float(item.get("AdjustmentOpen")),
                "adjustment_high": self._safe_float(item.get("AdjustmentHigh")),
                "adjustment_low": self._safe_float(item.get("AdjustmentLow")),
                "adjustment_close": self._safe_float(item.get("AdjustmentClose")),
                "adjustment_volume": self._safe_float(item.get("AdjustmentVolume")),
            }
            
            # UPSERT (SQLite)
            stmt = insert(Price).values(**price)
            stmt = stmt.on_conflict_do_update(
                index_elements=["code", "date"],
                set_={k: v for k, v in price.items() if k not in ["code", "date"]}
            )
            self.session.execute(stmt)
            count += 1
        
        self.session.commit()
        logger.info(f"株価データ: {count}件 保存完了")
        return count
    
    # ===================
    # 財務データ
    # ===================
    
    def collect_financials(self, from_date: Optional[str] = None, 
                           to_date: Optional[str] = None,
                           codes: Optional[List[str]] = None) -> int:
        """
        財務データを収集
        
        Args:
            from_date: 開始日（YYYY-MM-DD）
            to_date: 終了日（YYYY-MM-DD）
            codes: 銘柄コードリスト（省略時は全銘柄）
        
        Returns:
            取得件数
        """
        logger.info(f"財務データを取得中... ({from_date if from_date else 'All'} ~ {to_date if to_date else 'All'})")
        
        all_data = []
        
        if codes:
            # 銘柄指定（既存ロジック）
            for code in tqdm(codes, desc="銘柄別財務取得"):
                data = self.client.get_financial_statements(code=code)
                all_data.extend(data)
        elif from_date and to_date:
            # 日付範囲指定（API制限対応のため日次ループ）
            start = datetime.strptime(from_date, "%Y-%m-%d")
            end = datetime.strptime(to_date, "%Y-%m-%d")
            
            date_list = []
            current = start
            while current <= end:
                date_list.append(current.strftime("%Y-%m-%d"))
                current += timedelta(days=1)
            
            for date_str in tqdm(date_list, desc="日別財務取得"):
                try:
                    data = self.client.get_financial_statements(date=date_str)
                    if data:
                        all_data.extend(data)
                except Exception as e:
                    # エラーログは出すが処理は継続
                    logger.debug(f"{date_str}: {e}")
        else:
            # 引数なしはエラーになるため、今日の日付で取得などのフォールバックが必要だが、
            # 基本的には呼び出し元で日付を指定すべき
            logger.warning("日付範囲または銘柄指定がありません。本日分のデータを取得します。")
            today = datetime.now().strftime("%Y-%m-%d")
            all_data = self.client.get_financial_statements(date=today)
        
        count = 0
        for item in tqdm(all_data, desc="財務保存"):
            # 四半期判定
            period_type = item.get("TypeOfCurrentPeriod", "")
            if "1Q" in period_type:
                fiscal_quarter = 1
            elif "2Q" in period_type:
                fiscal_quarter = 2
            elif "3Q" in period_type:
                fiscal_quarter = 3
            else:
                fiscal_quarter = 4  # 通期
            
            # 会計年度
            fiscal_year_end = item.get("CurrentFiscalYearEndDate")
            fiscal_year = None
            if fiscal_year_end:
                try:
                    fiscal_year = int(fiscal_year_end[:4])
                except:
                    pass
            
            financial = {
                "code": item.get("LocalCode"),
                "disclosed_date": self._safe_date(item.get("DisclosedDate")),
                "disclosed_time": item.get("DisclosedTime"),
                "type_of_document": item.get("TypeOfDocument"),
                "type_of_current_period": period_type,
                "fiscal_year": fiscal_year,
                "fiscal_quarter": fiscal_quarter,
                "current_period_start_date": self._safe_date(item.get("CurrentPeriodStartDate")),
                "current_period_end_date": self._safe_date(item.get("CurrentPeriodEndDate")),
                "current_fiscal_year_start_date": self._safe_date(item.get("CurrentFiscalYearStartDate")),
                "current_fiscal_year_end_date": self._safe_date(item.get("CurrentFiscalYearEndDate")),
                "net_sales": self._safe_float(item.get("NetSales")),
                "operating_profit": self._safe_float(item.get("OperatingProfit")),
                "ordinary_profit": self._safe_float(item.get("OrdinaryProfit")),
                "profit": self._safe_float(item.get("Profit")),
                "total_assets": self._safe_float(item.get("TotalAssets")),
                "equity": self._safe_float(item.get("Equity")),
                "cash_flows_from_operating_activities": self._safe_float(item.get("CashFlowsFromOperatingActivities")),
                "cash_flows_from_investing_activities": self._safe_float(item.get("CashFlowsFromInvestingActivities")),
                "cash_flows_from_financing_activities": self._safe_float(item.get("CashFlowsFromFinancingActivities")),
                "earnings_per_share": self._safe_float(item.get("EarningsPerShare")),
                "book_value_per_share": self._safe_float(item.get("BookValuePerShare")),
                "dividend_per_share": self._safe_float(item.get("ResultDividendPerShare1stQuarter") or 
                                                        item.get("ResultDividendPerShare2ndQuarter") or
                                                        item.get("ResultDividendPerShareFiscalYearEnd")),
                "forecast_net_sales": self._safe_float(item.get("ForecastNetSales")),
                "forecast_operating_profit": self._safe_float(item.get("ForecastOperatingProfit")),
                "forecast_ordinary_profit": self._safe_float(item.get("ForecastOrdinaryProfit")),
                "forecast_profit": self._safe_float(item.get("ForecastProfit")),
                "forecast_earnings_per_share": self._safe_float(item.get("ForecastEarningsPerShare")),
                "forecast_dividend_per_share": self._safe_float(item.get("ForecastDividendPerShareFiscalYearEnd")),
                "change_net_sales": self._safe_float(item.get("ChangesInNetSales")),
                "change_operating_profit": self._safe_float(item.get("ChangesInOperatingProfit")),
                "change_ordinary_profit": self._safe_float(item.get("ChangesInOrdinaryProfit")),
                "change_profit": self._safe_float(item.get("ChangesInProfit")),
            }
            
            # UPSERT
            stmt = insert(Financial).values(**financial)
            stmt = stmt.on_conflict_do_update(
                index_elements=["code", "disclosed_date", "type_of_current_period"],
                set_={k: v for k, v in financial.items() 
                      if k not in ["code", "disclosed_date", "type_of_current_period"]}
            )
            self.session.execute(stmt)
            count += 1
        
        self.session.commit()
        logger.info(f"財務データ: {count}件 保存完了")
        return count
    
    # ===================
    # 取引カレンダー
    # ===================
    
    def collect_trading_calendar(self, from_date: str, to_date: str) -> int:
        """取引カレンダーを収集"""
        logger.info("取引カレンダーを取得中...")
        
        data = self.client.get_trading_calendar(from_date=from_date, to_date=to_date)
        
        count = 0
        for item in data:
            calendar = TradingCalendar(
                date=self._safe_date(item.get("Date")),
                is_trading_day=item.get("HolidayDivision") == "1"
            )
            self.session.merge(calendar)
            count += 1
        
        self.session.commit()
        logger.info(f"取引カレンダー: {count}件 保存完了")
        return count
    
    # ===================
    # TOPIX
    # ===================
    
    def collect_topix(self, from_date: str, to_date: str) -> int:
        """TOPIXデータを収集"""
        logger.info("TOPIXデータを取得中...")
        
        data = self.client.get_topix_daily(from_date=from_date, to_date=to_date)
        
        count = 0
        for item in data:
            topix = Topix(
                date=self._safe_date(item.get("Date")),
                open=self._safe_float(item.get("Open")),
                high=self._safe_float(item.get("High")),
                low=self._safe_float(item.get("Low")),
                close=self._safe_float(item.get("Close")),
            )
            self.session.merge(topix)
            count += 1
        
        self.session.commit()
        logger.info(f"TOPIX: {count}件 保存完了")
        return count
    
    # ===================
    # 信用取引
    # ===================
    
    def collect_margin_balance(self, from_date: str, to_date: str,
                               codes: Optional[List[str]] = None) -> int:
        """信用取引週末残高を収集"""
        logger.info(f"信用取引残高を取得中... ({from_date} ~ {to_date})")
        
        target_codes = codes
        if not target_codes:
            # 銘柄マスタから有効な銘柄を取得
            stocks = self.session.query(Stock.code).filter(Stock.is_active == True).all()
            target_codes = [s[0] for s in stocks]
            logger.info(f"全銘柄対象: {len(target_codes)}件")

        # 日付範囲を分割するためのヘルパー
        def split_date_range(start_str, end_str, years=5):
            s = datetime.strptime(start_str, "%Y-%m-%d")
            e = datetime.strptime(end_str, "%Y-%m-%d")
            chunks = []
            current = s
            while current < e:
                next_chunk = min(current + timedelta(days=365*years), e)
                chunks.append((current.strftime("%Y-%m-%d"), next_chunk.strftime("%Y-%m-%d")))
                current = next_chunk + timedelta(days=1)
            return chunks

        date_chunks = split_date_range(from_date, to_date)
        
        count = 0
        # 銘柄ごとにループ
        for code in tqdm(target_codes, desc="銘柄別信用残高取得"):
            stock_data = []
            for chunk_start, chunk_end in date_chunks:
                try:
                    data = self.client.get_margin_trades(
                        code=code, from_date=chunk_start, to_date=chunk_end
                    )
                    stock_data.extend(data)
                except Exception as e:
                    # エラーログは出すが処理は継続
                    logger.debug(f"Error fetching {code} ({chunk_start}~{chunk_end}): {e}")
            
            # 各銘柄の全期間データを保存
            for item in stock_data:
                margin = {
                    "code": item.get("Code"),
                    "date": self._safe_date(item.get("Date")),
                    "margin_buying_balance": self._safe_float(item.get("MarginBuyingBalance")),
                    "margin_buying_value": self._safe_float(item.get("MarginBuyingValue")),
                    "margin_selling_balance": self._safe_float(item.get("MarginSellingBalance")),
                    "margin_selling_value": self._safe_float(item.get("MarginSellingValue")),
                }
                
                stmt = insert(MarginBalance).values(**margin)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["code", "date"],
                    set_={k: v for k, v in margin.items() if k not in ["code", "date"]}
                )
                self.session.execute(stmt)
                count += 1
            
            # 一定間隔でコミット（メモリ節約）
            if count % 1000 == 0:
                self.session.commit()
        
        self.session.commit()
        logger.info(f"信用取引残高: {count}件 保存完了")
        return count
    
    # ===================
    # 空売り比率
    # ===================
    
    def collect_short_selling(self, from_date: str, to_date: str) -> int:
        """業種別空売り比率を収集"""
        logger.info("空売り比率を取得中...")
        
        data = self.client.get_short_selling(from_date=from_date, to_date=to_date)
        
        count = 0
        for item in data:
            short = {
                "date": self._safe_date(item.get("Date")),
                "sector_33_code": item.get("Sector33Code"),
                "sector_33_name": item.get("Sector33CodeName"),
                "selling_value": self._safe_float(item.get("SellingValue")),
                "short_selling_with_restrictions": self._safe_float(item.get("ShortSellingWithRestrictions")),
                "short_selling_without_restrictions": self._safe_float(item.get("ShortSellingWithoutRestrictions")),
            }
            
            stmt = insert(ShortSelling).values(**short)
            stmt = stmt.on_conflict_do_update(
                index_elements=["date", "sector_33_code"],
                set_={k: v for k, v in short.items() if k not in ["date", "sector_33_code"]}
            )
            self.session.execute(stmt)
            count += 1
        
        self.session.commit()
        logger.info(f"空売り比率: {count}件 保存完了")
        return count
    
    # ===================
    # 一括収集
    # ===================
    
    def collect_all_historical(self, years: int = 10):
        """
        過去データを一括収集
        
        Args:
            years: 取得する年数
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        from_str = start_date.strftime("%Y-%m-%d")
        to_str = end_date.strftime("%Y-%m-%d")
        
        logger.info(f"=== 過去データ一括収集開始 ({from_str} ~ {to_str}) ===")
        
        # 1. 銘柄マスタ
        self.collect_stocks()
        
        # 2. 取引カレンダー
        self.collect_trading_calendar(from_str, to_str)
        
        # 3. TOPIX
        self.collect_topix(from_str, to_str)
        
        # 4. 株価（年単位で分割）
        current = start_date
        while current < end_date:
            year_end = min(current + timedelta(days=365), end_date)
            self.collect_prices(
                current.strftime("%Y-%m-%d"),
                year_end.strftime("%Y-%m-%d")
            )
            current = year_end + timedelta(days=1)
        
        # 5. 財務データ
        self.collect_financials()
        
        # 6. 信用取引残高
        self.collect_margin_balance(from_str, to_str)
        
        # 7. 空売り比率
        self.collect_short_selling(from_str, to_str)
        
        logger.info("=== 過去データ一括収集完了 ===")
