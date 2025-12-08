"""
設定管理モジュール

使用前に環境変数または.envファイルで以下を設定:
- JQUANTS_MAIL_ADDRESS: J-Quants登録メールアドレス
- JQUANTS_PASSWORD: J-Quantsパスワード
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# .envファイルを読み込み
load_dotenv()

# プロジェクトルート
PROJECT_ROOT = Path(__file__).parent.parent

# データディレクトリ
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
DB_PATH = DATA_DIR / "jp_stock.db"

# ディレクトリ作成
DATA_DIR.mkdir(exist_ok=True)
RAW_DATA_DIR.mkdir(exist_ok=True)

# J-Quants API設定
JQUANTS_CONFIG = {
    "mail_address": os.getenv("JQUANTS_MAIL_ADDRESS", ""),
    "password": os.getenv("JQUANTS_PASSWORD", ""),
    "base_url": "https://api.jquants.com/v1",
}

# データ収集設定
COLLECTION_CONFIG = {
    # 過去データ取得期間（年）
    "historical_years": 10,
    # API呼び出し間隔（秒）- レートリミット対策
    "api_interval": 0.5,
    # リトライ回数
    "max_retries": 3,
}

# 対象市場区分
TARGET_MARKETS = [
    "プライム",
    "スタンダード", 
    "グロース",
]

# 除外銘柄（ETF、REITなど）
# 必要に応じて追加
EXCLUDE_CODES = []


def validate_config():
    """設定の検証"""
    errors = []
    
    if not JQUANTS_CONFIG["mail_address"]:
        errors.append("JQUANTS_MAIL_ADDRESS が設定されていません")
    if not JQUANTS_CONFIG["password"]:
        errors.append("JQUANTS_PASSWORD が設定されていません")
    
    if errors:
        raise ValueError("\n".join(errors))
    
    return True
