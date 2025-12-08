"""
J-Quants API クライアント

認証フローとAPIエンドポイントへのアクセスを提供
"""
import time
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JQuantsClient:
    """J-Quants API クライアント"""
    
    BASE_URL = "https://api.jquants.com/v1"
    
    def __init__(self, mail_address: str, password: str, api_interval: float = 0.5):
        """
        初期化
        
        Args:
            mail_address: J-Quants登録メールアドレス
            password: パスワード
            api_interval: API呼び出し間隔（秒）
        """
        self.mail_address = mail_address
        self.password = password
        self.api_interval = api_interval
        
        self._refresh_token: Optional[str] = None
        self._id_token: Optional[str] = None
        self._token_expires_at: Optional[datetime] = None
        
        self._last_request_time: float = 0
    
    def _wait_for_rate_limit(self):
        """レートリミット対策の待機"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.api_interval:
            time.sleep(self.api_interval - elapsed)
        self._last_request_time = time.time()
    
    def _get_refresh_token(self) -> str:
        """リフレッシュトークンを取得"""
        url = f"{self.BASE_URL}/token/auth_user"
        response = requests.post(url, json={
            "mailaddress": self.mail_address,
            "password": self.password
        })
        response.raise_for_status()
        self._refresh_token = response.json()["refreshToken"]
        return self._refresh_token
    
    def _get_id_token(self) -> str:
        """IDトークンを取得（API認証用）"""
        if not self._refresh_token:
            self._get_refresh_token()
        
        url = f"{self.BASE_URL}/token/auth_refresh"
        params = {"refreshtoken": self._refresh_token}
        response = requests.post(url, params=params)
        response.raise_for_status()
        
        self._id_token = response.json()["idToken"]
        # トークンは約24時間有効だが、安全のため23時間で更新
        self._token_expires_at = datetime.now() + timedelta(hours=23)
        
        return self._id_token
    
    def _ensure_token(self):
        """有効なトークンを確保"""
        if (self._id_token is None or 
            self._token_expires_at is None or 
            datetime.now() >= self._token_expires_at):
            self._get_id_token()
            logger.info("トークンを更新しました")
    
    @property
    def headers(self) -> Dict[str, str]:
        """認証ヘッダー"""
        self._ensure_token()
        return {"Authorization": f"Bearer {self._id_token}"}
    
    def _request(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        API リクエスト実行
        
        Args:
            endpoint: エンドポイントパス（例: "/listed/info"）
            params: クエリパラメータ
        
        Returns:
            レスポンスJSON
        """
        self._wait_for_rate_limit()
        
        url = f"{self.BASE_URL}{endpoint}"
        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        
        return response.json()
    
    def _paginated_request(self, endpoint: str, params: Optional[Dict] = None, 
                           data_key: str = None) -> List[Dict]:
        """
        ページネーション対応リクエスト
        
        Args:
            endpoint: エンドポイントパス
            params: クエリパラメータ
            data_key: レスポンス内のデータキー
        
        Returns:
            全ページのデータを結合したリスト
        """
        all_data = []
        params = params or {}
        
        while True:
            response = self._request(endpoint, params)
            
            if data_key:
                data = response.get(data_key, [])
            else:
                # データキーを自動検出
                data_keys = [k for k in response.keys() if k != "pagination_key"]
                data_key = data_keys[0] if data_keys else None
                data = response.get(data_key, []) if data_key else []
            
            all_data.extend(data)
            
            # 次ページがあるか確認
            pagination_key = response.get("pagination_key")
            if not pagination_key:
                break
            
            params["pagination_key"] = pagination_key
            logger.debug(f"次ページ取得: {pagination_key}")
        
        return all_data
    
    # ===================
    # 上場銘柄情報
    # ===================
    
    def get_listed_info(self, code: Optional[str] = None, 
                        date: Optional[str] = None) -> List[Dict]:
        """
        上場銘柄一覧を取得
        
        Args:
            code: 銘柄コード（指定時は単一銘柄）
            date: 基準日（YYYY-MM-DD）
        
        Returns:
            銘柄情報のリスト
        """
        params = {}
        if code:
            params["code"] = code
        if date:
            params["date"] = date
        
        return self._paginated_request("/listed/info", params, "info")
    
    # ===================
    # 株価データ
    # ===================
    
    def get_prices_daily(self, code: Optional[str] = None,
                         date: Optional[str] = None,
                         from_date: Optional[str] = None,
                         to_date: Optional[str] = None) -> List[Dict]:
        """
        日次株価データを取得
        
        Args:
            code: 銘柄コード
            date: 特定日（YYYY-MM-DD）
            from_date: 開始日（YYYY-MM-DD）
            to_date: 終了日（YYYY-MM-DD）
        
        Returns:
            株価データのリスト
        """
        params = {}
        if code:
            params["code"] = code
        if date:
            params["date"] = date
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        
        return self._paginated_request("/prices/daily_quotes", params, "daily_quotes")
    
    # ===================
    # 財務データ
    # ===================
    
    def get_financial_statements(self, code: Optional[str] = None,
                                  date: Optional[str] = None) -> List[Dict]:
        """
        財務諸表データを取得
        
        Args:
            code: 銘柄コード
            date: 開示日（YYYY-MM-DD）
        
        Returns:
            財務データのリスト
        """
        params = {}
        if code:
            params["code"] = code
        if date:
            params["date"] = date
        
        return self._paginated_request("/fins/statements", params, "statements")
    
    def get_financial_announcement(self, code: Optional[str] = None,
                                    from_date: Optional[str] = None,
                                    to_date: Optional[str] = None) -> List[Dict]:
        """
        決算発表予定日を取得
        
        Args:
            code: 銘柄コード
            from_date: 開始日
            to_date: 終了日
        
        Returns:
            決算発表予定のリスト
        """
        params = {}
        if code:
            params["code"] = code
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        
        return self._paginated_request("/fins/announcement", params, "announcement")
    
    # ===================
    # 市場データ
    # ===================
    
    def get_trading_calendar(self, from_date: Optional[str] = None,
                              to_date: Optional[str] = None) -> List[Dict]:
        """
        取引カレンダーを取得
        
        Args:
            from_date: 開始日
            to_date: 終了日
        
        Returns:
            取引日情報のリスト
        """
        params = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        
        return self._paginated_request("/markets/trading_calendar", params, "trading_calendar")
    
    def get_topix_daily(self, from_date: Optional[str] = None,
                        to_date: Optional[str] = None) -> List[Dict]:
        """
        TOPIX日次データを取得
        """
        params = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        
        return self._paginated_request("/indices/topix", params, "topix")
    
    def get_indices_daily(self, code: Optional[str] = None,
                          from_date: Optional[str] = None,
                          to_date: Optional[str] = None) -> List[Dict]:
        """
        指数日次データを取得
        """
        params = {}
        if code:
            params["code"] = code
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        
        return self._paginated_request("/indices", params, "indices")
    
    # ===================
    # 投資部門別・信用取引
    # ===================
    
    def get_investor_trades(self, from_date: Optional[str] = None,
                            to_date: Optional[str] = None) -> List[Dict]:
        """
        投資部門別売買状況を取得
        """
        params = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        
        return self._paginated_request("/markets/trades_spec", params, "trades_spec")
    
    def get_margin_trades(self, code: Optional[str] = None,
                          from_date: Optional[str] = None,
                          to_date: Optional[str] = None) -> List[Dict]:
        """
        信用取引週末残高を取得
        """
        params = {}
        if code:
            params["code"] = code
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        
        return self._paginated_request("/markets/weekly_margin_interest", params, "weekly_margin_interest")
    
    def get_short_selling(self, sector_33: Optional[str] = None,
                          from_date: Optional[str] = None,
                          to_date: Optional[str] = None) -> List[Dict]:
        """
        業種別空売り比率を取得
        """
        params = {}
        if sector_33:
            params["sector33code"] = sector_33
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        
        return self._paginated_request("/markets/short_selling", params, "short_selling")
    
    def get_breakdown_trading(self, code: Optional[str] = None,
                               from_date: Optional[str] = None,
                               to_date: Optional[str] = None) -> List[Dict]:
        """
        日々公表信用取引残高を取得
        """
        params = {}
        if code:
            params["code"] = code
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        
        return self._paginated_request("/markets/breakdown", params, "breakdown")
