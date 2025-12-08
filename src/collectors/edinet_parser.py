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
