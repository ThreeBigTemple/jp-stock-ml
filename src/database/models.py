"""
データベースモデル定義

SQLAlchemy ORM モデル
"""
from datetime import datetime, date
from typing import Optional
from sqlalchemy import (
    create_engine, Column, String, Integer, Float, Date, DateTime,
    Boolean, Text, Index, UniqueConstraint, ForeignKey
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

Base = declarative_base()


class Stock(Base):
    """銘柄マスタ"""
    __tablename__ = "stocks"
    
    code = Column(String(10), primary_key=True)  # 証券コード
    company_name = Column(String(200))           # 会社名
    company_name_english = Column(String(200))   # 会社名（英語）
    sector_17_code = Column(String(10))          # 17業種コード
    sector_17_name = Column(String(50))          # 17業種名
    sector_33_code = Column(String(10))          # 33業種コード
    sector_33_name = Column(String(50))          # 33業種名
    scale_category = Column(String(20))          # 規模区分
    market_code = Column(String(10))             # 市場コード
    market_name = Column(String(50))             # 市場名（プライム等）
    
    listing_date = Column(Date)                  # 上場日
    delisting_date = Column(Date)                # 上場廃止日
    is_active = Column(Boolean, default=True)    # アクティブフラグ
    
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # リレーション
    prices = relationship("Price", back_populates="stock")
    financials = relationship("Financial", back_populates="stock")


class Price(Base):
    """日次株価データ"""
    __tablename__ = "prices"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(10), ForeignKey("stocks.code"), nullable=False)
    date = Column(Date, nullable=False)
    
    open = Column(Float)                    # 始値
    high = Column(Float)                    # 高値
    low = Column(Float)                     # 安値
    close = Column(Float)                   # 終値
    volume = Column(Float)                  # 出来高
    turnover_value = Column(Float)          # 売買代金
    
    adjustment_factor = Column(Float)       # 調整係数
    adjustment_open = Column(Float)         # 調整済始値
    adjustment_high = Column(Float)         # 調整済高値
    adjustment_low = Column(Float)          # 調整済安値
    adjustment_close = Column(Float)        # 調整済終値
    adjustment_volume = Column(Float)       # 調整済出来高
    
    # リレーション
    stock = relationship("Stock", back_populates="prices")
    
    __table_args__ = (
        UniqueConstraint("code", "date", name="uq_price_code_date"),
        Index("ix_prices_date", "date"),
        Index("ix_prices_code_date", "code", "date"),
    )


class Financial(Base):
    """財務データ（Point-in-Time管理）"""
    __tablename__ = "financials"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(10), ForeignKey("stocks.code"), nullable=False)
    
    # 期間情報
    disclosed_date = Column(Date)                  # 開示日（Point-in-Time用）
    disclosed_time = Column(String(10))            # 開示時刻
    
    type_of_document = Column(String(100))         # 書類種別
    type_of_current_period = Column(String(20))    # 期種別（1Q, 2Q, 3Q, FY等）
    
    fiscal_year = Column(Integer)                  # 会計年度
    fiscal_quarter = Column(Integer)               # 四半期（1-4）
    
    current_period_start_date = Column(Date)       # 当期開始日
    current_period_end_date = Column(Date)         # 当期終了日
    current_fiscal_year_start_date = Column(Date)  # 当会計年度開始日
    current_fiscal_year_end_date = Column(Date)    # 当会計年度終了日
    
    # 損益計算書
    net_sales = Column(Float)                      # 売上高
    operating_profit = Column(Float)               # 営業利益
    ordinary_profit = Column(Float)                # 経常利益
    profit = Column(Float)                         # 当期純利益
    
    # 貸借対照表
    total_assets = Column(Float)                   # 総資産
    equity = Column(Float)                         # 純資産
    
    # キャッシュフロー
    cash_flows_from_operating_activities = Column(Float)   # 営業CF
    cash_flows_from_investing_activities = Column(Float)   # 投資CF
    cash_flows_from_financing_activities = Column(Float)   # 財務CF
    
    # 1株当たり指標
    earnings_per_share = Column(Float)             # EPS
    book_value_per_share = Column(Float)           # BPS
    dividend_per_share = Column(Float)             # DPS
    
    # 予想値
    forecast_net_sales = Column(Float)             # 売上高予想
    forecast_operating_profit = Column(Float)      # 営業利益予想
    forecast_ordinary_profit = Column(Float)       # 経常利益予想
    forecast_profit = Column(Float)                # 純利益予想
    forecast_earnings_per_share = Column(Float)    # EPS予想
    forecast_dividend_per_share = Column(Float)    # DPS予想
    
    # 前期比
    change_net_sales = Column(Float)               # 売上高前期比
    change_operating_profit = Column(Float)        # 営業利益前期比
    change_ordinary_profit = Column(Float)         # 経常利益前期比
    change_profit = Column(Float)                  # 純利益前期比
    
    # リレーション
    stock = relationship("Stock", back_populates="financials")
    
    __table_args__ = (
        UniqueConstraint("code", "disclosed_date", "type_of_current_period", 
                        name="uq_financial_code_date_period"),
        Index("ix_financials_disclosed_date", "disclosed_date"),
        Index("ix_financials_code_disclosed_date", "code", "disclosed_date"),
    )


class TradingCalendar(Base):
    """取引カレンダー"""
    __tablename__ = "trading_calendar"
    
    date = Column(Date, primary_key=True)
    is_trading_day = Column(Boolean, default=True)
    
    __table_args__ = (
        Index("ix_trading_calendar_date", "date"),
    )


class Topix(Base):
    """TOPIX日次データ"""
    __tablename__ = "topix"
    
    date = Column(Date, primary_key=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)


class MarginBalance(Base):
    """信用取引週末残高"""
    __tablename__ = "margin_balance"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(10), nullable=False)
    date = Column(Date, nullable=False)
    
    margin_buying_balance = Column(Float)      # 信用買残高（株数）
    margin_buying_value = Column(Float)        # 信用買残高（金額）
    margin_selling_balance = Column(Float)     # 信用売残高（株数）
    margin_selling_value = Column(Float)       # 信用売残高（金額）
    
    # 信用倍率は算出可能
    
    __table_args__ = (
        UniqueConstraint("code", "date", name="uq_margin_code_date"),
        Index("ix_margin_balance_code_date", "code", "date"),
    )


class ShortSelling(Base):
    """業種別空売り比率"""
    __tablename__ = "short_selling"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False)
    sector_33_code = Column(String(10))
    sector_33_name = Column(String(50))
    
    selling_value = Column(Float)              # 売り金額
    short_selling_with_restrictions = Column(Float)    # 空売り金額（価格規制あり）
    short_selling_without_restrictions = Column(Float) # 空売り金額（価格規制なし）
    
    __table_args__ = (
        UniqueConstraint("date", "sector_33_code", name="uq_short_selling_date_sector"),
        Index("ix_short_selling_date", "date"),
    )


class InvestorTrades(Base):
    """投資部門別売買状況"""
    __tablename__ = "investor_trades"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    published_date = Column(Date, nullable=False)
    start_date = Column(Date)
    end_date = Column(Date)
    
    section = Column(String(50))               # 部門名
    
    # 売買金額
    total_sell_value = Column(Float)
    total_buy_value = Column(Float)
    
    __table_args__ = (
        Index("ix_investor_trades_date", "published_date"),
    )


def init_db(db_path: str):
    """データベース初期化"""
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    return engine


class EdinetFinancial(Base):
    """EDINET詳細財務データ"""
    __tablename__ = "edinet_financials"

    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(10), nullable=False)           # 証券コード
    edinet_code = Column(String(10))                    # EDINETコード
    doc_id = Column(String(20), nullable=False)         # 書類管理番号
    doc_type_code = Column(String(10))                  # 書類種別
    submit_date = Column(Date)                          # 提出日
    fiscal_year_end = Column(Date)                      # 決算期末日

    # 損益計算書（詳細）
    net_sales = Column(Float)                           # 売上高
    cost_of_sales = Column(Float)                       # 売上原価
    gross_profit = Column(Float)                        # 売上総利益
    sga_expenses = Column(Float)                        # 販管費
    rd_expenses = Column(Float)                         # 研究開発費
    operating_income = Column(Float)                    # 営業利益
    ordinary_income = Column(Float)                     # 経常利益
    net_income = Column(Float)                          # 純利益

    # 貸借対照表
    total_assets = Column(Float)                        # 総資産
    net_assets = Column(Float)                          # 純資産
    shareholders_equity = Column(Float)                 # 株主資本
    interest_bearing_debt = Column(Float)               # 有利子負債

    # キャッシュフロー
    cf_operating = Column(Float)                        # 営業CF
    cf_investing = Column(Float)                        # 投資CF
    cf_financing = Column(Float)                        # 財務CF

    # 投資・その他
    capex = Column(Float)                               # 設備投資
    depreciation = Column(Float)                        # 減価償却費
    employees = Column(Integer)                         # 従業員数

    __table_args__ = (
        UniqueConstraint("code", "doc_id", name="uq_edinet_code_doc"),
        Index("ix_edinet_financials_code_date", "code", "submit_date"),
    )


class Disclosure(Base):
    """適時開示情報（TDnet）"""
    __tablename__ = "disclosures"

    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(10), nullable=False)           # 証券コード
    date = Column(Date, nullable=False)                 # 開示日
    time = Column(String(10))                           # 開示時刻
    title = Column(Text)                                # 開示タイトル
    disclosure_type = Column(String(50))                # 開示タイプ
    pdf_url = Column(Text)                              # PDFのURL

    # 業績予想修正の場合の詳細（将来拡張用）
    revision_sales = Column(Float)                      # 売上修正率
    revision_operating = Column(Float)                  # 営業利益修正率
    revision_net = Column(Float)                        # 純利益修正率

    __table_args__ = (
        UniqueConstraint("code", "date", "title", name="uq_disclosure"),
        Index("ix_disclosures_code_date", "code", "date"),
        Index("ix_disclosures_type", "disclosure_type"),
    )


class GlobalIndex(Base):
    """海外指数データ"""
    __tablename__ = "global_indices"

    id = Column(Integer, primary_key=True, autoincrement=True)
    index_name = Column(String(20), nullable=False)     # sp500, vix等
    date = Column(Date, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)

    __table_args__ = (
        UniqueConstraint("index_name", "date", name="uq_global_index"),
        Index("ix_global_indices_date", "date"),
    )


class SearchTrend(Base):
    """Google Trends検索トレンド"""
    __tablename__ = "search_trends"

    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(10), nullable=False)           # 証券コード
    week_start = Column(Date, nullable=False)           # 週の開始日
    keyword = Column(String(100))                       # 検索キーワード
    interest = Column(Integer)                          # 検索ボリューム（0-100）

    __table_args__ = (
        UniqueConstraint("code", "week_start", "keyword", name="uq_search_trend"),
        Index("ix_search_trends_code_week", "code", "week_start"),
    )


def get_session(engine):
    """セッション取得"""
    Session = sessionmaker(bind=engine)
    return Session()
