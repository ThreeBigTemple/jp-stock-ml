"""
日付関連ユーティリティ
"""
from datetime import datetime, date, timedelta
from typing import List, Optional
import pandas as pd


def get_trading_days(start_date: date, end_date: date, 
                     calendar_df: Optional[pd.DataFrame] = None) -> List[date]:
    """
    取引日リストを取得
    
    Args:
        start_date: 開始日
        end_date: 終了日
        calendar_df: 取引カレンダーDataFrame（date, is_trading_day列）
    
    Returns:
        取引日のリスト
    """
    if calendar_df is not None:
        mask = (
            (calendar_df['date'] >= start_date) & 
            (calendar_df['date'] <= end_date) &
            (calendar_df['is_trading_day'] == True)
        )
        return calendar_df.loc[mask, 'date'].tolist()
    
    # カレンダーがない場合は平日を返す（簡易版）
    days = []
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:  # 月-金
            days.append(current)
        current += timedelta(days=1)
    return days


def get_previous_trading_day(target_date: date, 
                             calendar_df: Optional[pd.DataFrame] = None,
                             n: int = 1) -> date:
    """
    N営業日前の日付を取得
    
    Args:
        target_date: 基準日
        calendar_df: 取引カレンダーDataFrame
        n: 何営業日前か
    
    Returns:
        N営業日前の日付
    """
    if calendar_df is not None:
        trading_days = calendar_df[
            (calendar_df['date'] < target_date) &
            (calendar_df['is_trading_day'] == True)
        ]['date'].sort_values(ascending=False)
        
        if len(trading_days) >= n:
            return trading_days.iloc[n-1]
    
    # 簡易版
    current = target_date - timedelta(days=1)
    count = 0
    while count < n:
        if current.weekday() < 5:
            count += 1
            if count == n:
                return current
        current -= timedelta(days=1)
    return current


def parse_date(date_str: str) -> date:
    """
    日付文字列をパース
    
    サポートするフォーマット:
    - YYYY-MM-DD
    - YYYYMMDD
    - YYYY/MM/DD
    """
    formats = ["%Y-%m-%d", "%Y%m%d", "%Y/%m/%d"]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    
    raise ValueError(f"日付をパースできません: {date_str}")


def date_range(start_date: date, end_date: date) -> List[date]:
    """日付範囲を生成"""
    days = []
    current = start_date
    while current <= end_date:
        days.append(current)
        current += timedelta(days=1)
    return days
