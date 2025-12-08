# JP Stock ML - 無料データソース全活用コレクター実装プロンプト

## 概要

J-Quants APIに加えて、無料で利用可能な全てのデータソースを活用したコレクターモジュールを実装する。

## 対象データソース

| データソース | 取得データ | 実装優先度 |
|-------------|-----------|-----------|
| J-Quants API | 株価、財務、信用残高 | ✅ 実装済み |
| EDINET API | 詳細財務（XBRL） | ⭐⭐⭐ 高 |
| TDnet | 適時開示（業績修正等） | ⭐⭐⭐ 高 |
| yfinance | 株価バックアップ、海外指数 | ⭐⭐ 中 |
| Google Trends | 検索トレンド | ⭐ 低 |

---

## 実装ファイル構造

```
src/collectors/
├── __init__.py
├── jquants_client.py        # 既存
├── jquants_collector.py     # 既存
├── edinet_client.py         # 新規: EDINET APIクライアント
├── edinet_collector.py      # 新規: EDINET収集ロジック
├── edinet_parser.py         # 新規: XBRLパーサー
├── tdnet_client.py          # 新規: TDnetスクレイパー
├── tdnet_collector.py       # 新規: TDnet収集ロジック
├── yfinance_client.py       # 新規: yfinanceラッパー
├── trends_client.py         # 新規: Google Trendsクライアント
└── master_collector.py      # 新規: 統合コレクター

src/database/models.py       # テーブル追加
```

---

## 1. EDINET API 実装

### 概要

金融庁のEDINET APIから有価証券報告書・四半期報告書のXBRLデータを取得し、詳細財務情報を抽出する。

### APIエンドポイント

```
Base URL: https://api.edinet-fsa.go.jp/api/v2

GET /documents.json?date={YYYY-MM-DD}&type=2
  → 指定日の開示書類一覧

GET /documents/{docID}?type=1
  → XBRL ZIP ダウンロード
```

### edinet_client.py

```python
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
```

### edinet_parser.py

```python
"""
EDINET XBRLパーサー

XBRLファイルから財務データを抽出
"""
import zipfile
import io
import re
import xml.etree.ElementTree as ET
from typing import Dict, Optional, List, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class EdinetXbrlParser:
    """EDINET XBRL パーサー"""
    
    # 抽出対象の勘定科目（XBRL要素名）
    # 日本基準（jppfs）と IFRS（ifrs）両対応
    TARGET_ELEMENTS = {
        # 売上高
        "net_sales": [
            "jppfs_cor:NetSales",
            "jppfs_cor:OperatingRevenue1",
            "ifrs-full:Revenue",
        ],
        # 売上原価
        "cost_of_sales": [
            "jppfs_cor:CostOfSales",
            "ifrs-full:CostOfSales",
        ],
        # 売上総利益
        "gross_profit": [
            "jppfs_cor:GrossProfit",
            "ifrs-full:GrossProfit",
        ],
        # 販管費
        "sga_expenses": [
            "jppfs_cor:SellingGeneralAndAdministrativeExpenses",
        ],
        # 研究開発費
        "rd_expenses": [
            "jppfs_cor:ResearchAndDevelopmentExpenses",
            "jpcrp_cor:ResearchAndDevelopmentExpensesTextBlock",  # 注記から
        ],
        # 営業利益
        "operating_income": [
            "jppfs_cor:OperatingIncome",
            "ifrs-full:ProfitLossFromOperatingActivities",
        ],
        # 経常利益
        "ordinary_income": [
            "jppfs_cor:OrdinaryIncome",
        ],
        # 純利益
        "net_income": [
            "jppfs_cor:ProfitLoss",
            "ifrs-full:ProfitLoss",
        ],
        # 総資産
        "total_assets": [
            "jppfs_cor:Assets",
            "ifrs-full:Assets",
        ],
        # 純資産
        "net_assets": [
            "jppfs_cor:NetAssets",
            "ifrs-full:Equity",
        ],
        # 自己資本（株主資本）
        "shareholders_equity": [
            "jppfs_cor:ShareholdersEquity",
            "ifrs-full:EquityAttributableToOwnersOfParent",
        ],
        # 有利子負債
        "interest_bearing_debt": [
            "jppfs_cor:ShortTermLoansPayable",
            "jppfs_cor:LongTermLoansPayable",
        ],
        # 営業CF
        "cf_operating": [
            "jppfs_cor:NetCashProvidedByUsedInOperatingActivities",
            "ifrs-full:CashFlowsFromUsedInOperatingActivities",
        ],
        # 投資CF
        "cf_investing": [
            "jppfs_cor:NetCashProvidedByUsedInInvestingActivities",
            "ifrs-full:CashFlowsFromUsedInInvestingActivities",
        ],
        # 財務CF
        "cf_financing": [
            "jppfs_cor:NetCashProvidedByUsedInFinancingActivities",
            "ifrs-full:CashFlowsFromUsedInFinancingActivities",
        ],
        # 設備投資額
        "capex": [
            "jppfs_cor:PurchaseOfPropertyPlantAndEquipmentAndIntangibleAssets",
            "jpcrp_cor:CapitalExpendituresEtcTextBlock",
        ],
        # 減価償却費
        "depreciation": [
            "jppfs_cor:DepreciationAndAmortization",
            "ifrs-full:DepreciationAndAmortisationExpense",
        ],
        # 従業員数
        "employees": [
            "jpcrp_cor:NumberOfEmployees",
        ],
    }
    
    # 名前空間
    NAMESPACES = {
        "xbrli": "http://www.xbrl.org/2003/instance",
        "jppfs_cor": "http://disclosure.edinet-fsa.go.jp/taxonomy/jppfs/2023-12-01/jppfs_cor",
        "jpcrp_cor": "http://disclosure.edinet-fsa.go.jp/taxonomy/jpcrp/2023-12-01/jpcrp_cor",
        "ifrs-full": "http://xbrl.ifrs.org/taxonomy/2023-03-23/ifrs-full",
    }
    
    def __init__(self):
        self.parsed_data = {}
    
    def parse_zip(self, zip_content: bytes) -> Optional[Dict[str, Any]]:
        """
        XBRLのZIPファイルをパース
        
        Args:
            zip_content: ZIPファイルのバイト列
        
        Returns:
            抽出した財務データの辞書
        """
        try:
            with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
                # XBRLインスタンスファイルを探す
                xbrl_files = [f for f in zf.namelist() 
                              if f.endswith('.xbrl') and 'AuditDoc' not in f]
                
                if not xbrl_files:
                    logger.warning("XBRLファイルが見つかりません")
                    return None
                
                # メインのXBRLファイル（通常は最も大きいもの）
                main_xbrl = max(xbrl_files, key=lambda f: zf.getinfo(f).file_size)
                
                with zf.open(main_xbrl) as f:
                    return self.parse_xbrl(f.read())
                    
        except Exception as e:
            logger.error(f"ZIP parse error: {e}")
            return None
    
    def parse_xbrl(self, xbrl_content: bytes) -> Dict[str, Any]:
        """
        XBRLインスタンスをパース
        
        Args:
            xbrl_content: XBRLファイルの内容
        
        Returns:
            抽出した財務データ
        """
        result = {}
        
        try:
            # 名前空間を動的に検出
            root = ET.fromstring(xbrl_content)
            
            # コンテキスト（期間情報）を取得
            contexts = self._parse_contexts(root)
            
            # 各勘定科目を抽出
            for field_name, element_names in self.TARGET_ELEMENTS.items():
                value = self._extract_value(root, element_names, contexts)
                if value is not None:
                    result[field_name] = value
            
            # メタ情報
            result["_contexts"] = contexts
            
        except Exception as e:
            logger.error(f"XBRL parse error: {e}")
        
        return result
    
    def _parse_contexts(self, root: ET.Element) -> Dict[str, Dict]:
        """コンテキスト（期間情報）をパース"""
        contexts = {}
        
        for ctx in root.findall(".//{http://www.xbrl.org/2003/instance}context"):
            ctx_id = ctx.get("id")
            if not ctx_id:
                continue
            
            period = ctx.find("{http://www.xbrl.org/2003/instance}period")
            if period is None:
                continue
            
            instant = period.find("{http://www.xbrl.org/2003/instance}instant")
            start_date = period.find("{http://www.xbrl.org/2003/instance}startDate")
            end_date = period.find("{http://www.xbrl.org/2003/instance}endDate")
            
            contexts[ctx_id] = {
                "instant": instant.text if instant is not None else None,
                "start_date": start_date.text if start_date is not None else None,
                "end_date": end_date.text if end_date is not None else None,
            }
        
        return contexts
    
    def _extract_value(self, root: ET.Element, element_names: List[str],
                       contexts: Dict) -> Optional[float]:
        """
        指定した勘定科目の値を抽出
        
        当期実績値を優先して取得
        """
        for elem_name in element_names:
            # 名前空間付きで検索
            for elem in root.iter():
                tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
                full_name = elem_name.split(":")[-1]
                
                if tag == full_name:
                    ctx_ref = elem.get("contextRef", "")
                    
                    # 当期・連結を優先
                    if "Current" in ctx_ref and "Consolidated" in ctx_ref:
                        try:
                            return float(elem.text) if elem.text else None
                        except (ValueError, TypeError):
                            pass
        
        return None


class EdinetSegmentParser:
    """セグメント情報パーサー"""
    
    def parse_segments(self, xbrl_content: bytes) -> List[Dict]:
        """
        セグメント別売上・利益を抽出
        
        Returns:
            [{"segment_name": "xxx", "sales": 1000, "profit": 100}, ...]
        """
        # セグメント情報は複雑なため、簡略化した実装
        # 本格実装では jpcrp_cor:SegmentInformationTextBlock などを解析
        pass
```

### edinet_collector.py

```python
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
        # なければEDINETコードマッピングテーブルを参照
        # 簡易実装: EDINETコードの一部が証券コード
        pass
    
    def _save_financial(self, stock_code: str, doc: dict, parsed: dict):
        """財務データをDBに保存"""
        financial = {
            "code": stock_code,
            "edinet_code": doc.get("edinetCode"),
            "doc_id": doc.get("docID"),
            "doc_type_code": doc.get("docTypeCode"),
            "submit_date": doc.get("submitDateTime", "")[:10],
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
```

---

## 2. TDnet 適時開示 実装

### 概要

東証の適時開示情報システム（TDnet）から業績予想修正、自社株買い、配当予想などのイベント情報を取得。

### tdnet_client.py

```python
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
```

### tdnet_collector.py

```python
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
```

---

## 3. yfinance 実装

### yfinance_client.py

```python
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
```

---

## 4. Google Trends 実装

### trends_client.py

```python
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
```

---

## 5. データベースモデル追加

### models.py に追加

```python
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
```

---

## 6. 統合コレクター

### master_collector.py

```python
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
```

---

## 7. pyproject.toml 追加依存関係

```toml
[project]
dependencies = [
    # 既存...
    
    # EDINET/TDnet
    "beautifulsoup4>=4.12.0",
    "lxml>=4.9.0",
    
    # yfinance
    "yfinance>=0.2.0",
    
    # Google Trends
    "pytrends>=4.9.0",
]
```

---

## 実装優先順位

1. **EDINET** - 詳細財務データは成長銘柄発掘に直結（研究開発費、設備投資等）
2. **TDnet** - 業績修正は強力なシグナル
3. **yfinance** - 海外指数は市場環境の把握に有用
4. **Google Trends** - 補助的（ノイズが多い可能性）

---

## 実行コマンド

```bash
# 環境更新
uv sync

# 全データソース一括収集
uv run python scripts/collect_all_sources.py --years 10

# 日次更新
uv run python scripts/daily_update_all.py
```
