"""
評価指標モジュール

IC、ICIR、シャープレシオ等のファクター投資評価指標を実装
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    情報係数（IC）- スピアマン順位相関

    Args:
        y_true: 実際のリターン
        y_pred: 予測スコア

    Returns:
        スピアマン順位相関係数
    """
    # NaNを除去
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 3:
        return np.nan

    corr, _ = stats.spearmanr(y_true[mask], y_pred[mask])
    return corr


def pearson_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    ピアソン相関による情報係数

    Args:
        y_true: 実際のリターン
        y_pred: 予測スコア

    Returns:
        ピアソン相関係数
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 3:
        return np.nan

    corr, _ = stats.pearsonr(y_true[mask], y_pred[mask])
    return corr


def ic_by_date(results_df: pd.DataFrame,
               y_true_col: str = 'y_true',
               y_pred_col: str = 'y_pred') -> pd.Series:
    """
    日次IC

    Args:
        results_df: 予測結果DataFrame
        y_true_col: 真値カラム名
        y_pred_col: 予測値カラム名

    Returns:
        日次ICのSeries
    """
    def calc_ic(group):
        return information_coefficient(
            group[y_true_col].values,
            group[y_pred_col].values
        )

    return results_df.groupby('date').apply(calc_ic)


def icir(ic_series: pd.Series) -> float:
    """
    ICIR（IC Information Ratio）

    IC平均 / IC標準偏差

    Args:
        ic_series: 日次ICのSeries

    Returns:
        ICIR
    """
    ic_clean = ic_series.dropna()
    if len(ic_clean) < 2 or ic_clean.std() == 0:
        return 0.0
    return ic_clean.mean() / ic_clean.std()


def annualized_icir(ic_series: pd.Series, periods_per_year: int = 252) -> float:
    """
    年率換算ICIR

    Args:
        ic_series: 日次ICのSeries
        periods_per_year: 年間期間数

    Returns:
        年率換算ICIR
    """
    return icir(ic_series) * np.sqrt(periods_per_year)


def hit_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    ヒット率（方向正解率）

    Args:
        y_true: 実際のリターン
        y_pred: 予測スコア

    Returns:
        正解率
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return np.nan

    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    return np.mean(np.sign(y_true_clean) == np.sign(y_pred_clean))


def top_bottom_return(results_df: pd.DataFrame,
                      n_quantiles: int = 5,
                      y_true_col: str = 'y_true',
                      y_pred_col: str = 'y_pred') -> pd.DataFrame:
    """
    ロング・ショートリターン分析（分位別）

    Args:
        results_df: 予測結果DataFrame
        n_quantiles: 分位数
        y_true_col: 真値カラム名
        y_pred_col: 予測値カラム名

    Returns:
        分位別の統計量DataFrame
    """
    results = results_df.copy()

    # 日次で分位を割り当て
    def assign_quantile(group):
        try:
            group['quantile'] = pd.qcut(
                group[y_pred_col],
                n_quantiles,
                labels=range(n_quantiles),
                duplicates='drop'
            )
        except ValueError:
            group['quantile'] = np.nan
        return group

    results = results.groupby('date', group_keys=False).apply(assign_quantile)

    # 分位別統計
    stats = results.groupby('quantile')[y_true_col].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('count', 'count'),
        ('median', 'median'),
    ])

    # シャープ比
    stats['sharpe'] = stats['mean'] / stats['std'] * np.sqrt(252)

    return stats


def long_short_return(results_df: pd.DataFrame,
                      top_pct: float = 0.2,
                      bottom_pct: float = 0.2,
                      y_true_col: str = 'y_true',
                      y_pred_col: str = 'y_pred') -> pd.Series:
    """
    ロングショートリターン（日次）

    Args:
        results_df: 予測結果DataFrame
        top_pct: ロング比率
        bottom_pct: ショート比率
        y_true_col: 真値カラム名
        y_pred_col: 予測値カラム名

    Returns:
        日次ロングショートリターンのSeries
    """
    def calc_ls(group):
        n = len(group)
        if n < 5:  # サンプル数が少なすぎる場合
            return np.nan

        top_n = max(1, int(n * top_pct))
        bottom_n = max(1, int(n * bottom_pct))

        sorted_group = group.sort_values(y_pred_col, ascending=False)
        long_return = sorted_group.head(top_n)[y_true_col].mean()
        short_return = sorted_group.tail(bottom_n)[y_true_col].mean()

        return long_return - short_return

    return results_df.groupby('date').apply(calc_ls)


def long_only_return(results_df: pd.DataFrame,
                     top_n: int = 50,
                     y_true_col: str = 'y_true',
                     y_pred_col: str = 'y_pred') -> pd.Series:
    """
    ロングオンリーリターン（日次）

    Args:
        results_df: 予測結果DataFrame
        top_n: 選択銘柄数
        y_true_col: 真値カラム名
        y_pred_col: 予測値カラム名

    Returns:
        日次ロングリターンのSeries
    """
    def calc_long(group):
        if len(group) < top_n:
            return np.nan

        sorted_group = group.sort_values(y_pred_col, ascending=False)
        return sorted_group.head(top_n)[y_true_col].mean()

    return results_df.groupby('date').apply(calc_long)


def simulated_portfolio_performance(results_df: pd.DataFrame,
                                     top_n: int = 50,
                                     holding_days: int = 20,
                                     transaction_cost: float = 0.001,
                                     y_true_col: str = 'y_true',
                                     y_pred_col: str = 'y_pred') -> Dict[str, float]:
    """
    シミュレーションポートフォリオパフォーマンス

    Args:
        results_df: 予測結果DataFrame
        top_n: 保有銘柄数
        holding_days: 保有期間
        transaction_cost: 取引コスト（往復）
        y_true_col: 真値カラム名
        y_pred_col: 予測値カラム名

    Returns:
        パフォーマンス指標の辞書
    """
    # 日次リターンを取得（リバランス日のみ）
    dates = sorted(results_df['date'].unique())

    portfolio_returns = []
    prev_holdings = set()

    for i, rebalance_date in enumerate(dates[::holding_days]):
        # その日のデータ
        day_data = results_df[results_df['date'] == rebalance_date]

        if len(day_data) < top_n:
            continue

        # 上位銘柄を選択
        top_stocks = day_data.nlargest(top_n, y_pred_col)
        current_holdings = set(top_stocks['code'].values)

        # リターン計算
        period_return = top_stocks[y_true_col].mean()

        # ターンオーバーコスト
        if prev_holdings:
            turnover = len(current_holdings - prev_holdings) / len(current_holdings)
            cost = transaction_cost * turnover
        else:
            cost = transaction_cost  # 初回は全額

        net_return = period_return - cost
        portfolio_returns.append(net_return)

        prev_holdings = current_holdings

    if len(portfolio_returns) < 2:
        return {
            'total_return': np.nan,
            'annual_return': np.nan,
            'annual_volatility': np.nan,
            'sharpe': np.nan,
            'max_dd': np.nan,
            'calmar': np.nan,
            'n_periods': 0,
        }

    returns = pd.Series(portfolio_returns)
    periods_per_year = 252 / holding_days

    # 累積リターン
    total_return = (1 + returns).prod() - 1

    # 年率リターン
    n_periods = len(returns)
    annual_return = (1 + total_return) ** (periods_per_year / n_periods) - 1

    # 年率ボラティリティ
    annual_volatility = returns.std() * np.sqrt(periods_per_year)

    # シャープレシオ
    sharpe = annual_return / annual_volatility if annual_volatility > 0 else 0

    # 最大ドローダウン
    mdd = max_drawdown(returns)

    # カルマーレシオ
    calmar = annual_return / abs(mdd) if abs(mdd) > 0 else 0

    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe': sharpe,
        'max_dd': mdd,
        'calmar': calmar,
        'n_periods': n_periods,
        'avg_turnover': np.nan,  # 別途計算が必要
    }


def max_drawdown(returns: pd.Series) -> float:
    """
    最大ドローダウン

    Args:
        returns: リターンのSeries

    Returns:
        最大ドローダウン（負の値）
    """
    if len(returns) == 0:
        return 0.0

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    カルマーレシオ（年率リターン/最大DD）

    Args:
        returns: リターンのSeries
        periods_per_year: 年間期間数

    Returns:
        カルマーレシオ
    """
    if len(returns) == 0:
        return 0.0

    annual_return = returns.mean() * periods_per_year
    mdd = abs(max_drawdown(returns))
    return annual_return / mdd if mdd > 0 else 0


def turnover_rate(results_df: pd.DataFrame,
                  top_n: int = 50,
                  y_pred_col: str = 'y_pred') -> float:
    """
    ターンオーバー率（リバランス時の入れ替え率）

    Args:
        results_df: 予測結果DataFrame
        top_n: ポートフォリオ銘柄数
        y_pred_col: 予測値カラム名

    Returns:
        平均ターンオーバー率
    """
    dates = sorted(results_df['date'].unique())

    turnovers = []
    prev_holdings = None

    for date in dates:
        day_data = results_df[results_df['date'] == date]

        if len(day_data) < top_n:
            continue

        top_stocks = set(day_data.nlargest(top_n, y_pred_col)['code'].values)

        if prev_holdings is not None:
            # 入れ替わった銘柄の割合
            changed = len(top_stocks - prev_holdings)
            turnover = changed / len(top_stocks)
            turnovers.append(turnover)

        prev_holdings = top_stocks

    return np.mean(turnovers) if turnovers else 0.0


def evaluate_all(results_df: pd.DataFrame,
                 y_true_col: str = 'y_true',
                 y_pred_col: str = 'y_pred',
                 top_n: int = 50,
                 holding_days: int = 20) -> Dict[str, float]:
    """
    全評価指標を計算

    Args:
        results_df: 予測結果DataFrame（code, date, y_true, y_pred）
        y_true_col: 真値カラム名
        y_pred_col: 予測値カラム名
        top_n: ポートフォリオ銘柄数
        holding_days: 保有期間

    Returns:
        評価指標の辞書
    """
    # 基本指標
    ic_series = ic_by_date(results_df, y_true_col, y_pred_col)
    ls_returns = long_short_return(results_df, y_true_col=y_true_col, y_pred_col=y_pred_col)

    # 日次ヒット率
    daily_hit_rates = results_df.groupby('date').apply(
        lambda x: hit_rate(x[y_true_col].values, x[y_pred_col].values)
    )

    # ポートフォリオシミュレーション
    portfolio_perf = simulated_portfolio_performance(
        results_df,
        top_n=top_n,
        holding_days=holding_days,
        y_true_col=y_true_col,
        y_pred_col=y_pred_col
    )

    return {
        # IC関連
        'ic_mean': ic_series.mean(),
        'ic_std': ic_series.std(),
        'icir': icir(ic_series),
        'annual_icir': annualized_icir(ic_series),
        'ic_positive_rate': (ic_series > 0).mean(),

        # ヒット率
        'hit_rate': daily_hit_rates.mean(),

        # ロングショートリターン
        'ls_return_mean': ls_returns.mean(),
        'ls_return_std': ls_returns.std(),
        'ls_sharpe': ls_returns.mean() / ls_returns.std() * np.sqrt(252) if ls_returns.std() > 0 else 0,
        'ls_max_drawdown': max_drawdown(ls_returns),

        # ポートフォリオパフォーマンス
        'portfolio_total_return': portfolio_perf['total_return'],
        'portfolio_annual_return': portfolio_perf['annual_return'],
        'portfolio_sharpe': portfolio_perf['sharpe'],
        'portfolio_max_dd': portfolio_perf['max_dd'],
        'portfolio_calmar': portfolio_perf['calmar'],

        # その他
        'n_samples': len(results_df),
        'n_dates': results_df['date'].nunique(),
        'n_stocks': results_df['code'].nunique(),
    }


def print_evaluation_report(metrics: Dict[str, float], title: str = "評価レポート"):
    """
    評価結果をフォーマットして表示

    Args:
        metrics: evaluate_all()の戻り値
        title: レポートタイトル
    """
    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")

    print("\n[IC関連]")
    print(f"  IC平均:         {metrics.get('ic_mean', np.nan):.4f}")
    print(f"  IC標準偏差:     {metrics.get('ic_std', np.nan):.4f}")
    print(f"  ICIR:           {metrics.get('icir', np.nan):.4f}")
    print(f"  年率ICIR:       {metrics.get('annual_icir', np.nan):.4f}")
    print(f"  IC正率:         {metrics.get('ic_positive_rate', np.nan):.1%}")

    print("\n[ヒット率]")
    print(f"  方向正解率:     {metrics.get('hit_rate', np.nan):.1%}")

    print("\n[ロングショート]")
    print(f"  平均リターン:   {metrics.get('ls_return_mean', np.nan):.4f}")
    print(f"  シャープレシオ: {metrics.get('ls_sharpe', np.nan):.2f}")
    print(f"  最大DD:         {metrics.get('ls_max_drawdown', np.nan):.1%}")

    print("\n[ポートフォリオ]")
    print(f"  累積リターン:   {metrics.get('portfolio_total_return', np.nan):.1%}")
    print(f"  年率リターン:   {metrics.get('portfolio_annual_return', np.nan):.1%}")
    print(f"  シャープレシオ: {metrics.get('portfolio_sharpe', np.nan):.2f}")
    print(f"  最大DD:         {metrics.get('portfolio_max_dd', np.nan):.1%}")
    print(f"  カルマーレシオ: {metrics.get('portfolio_calmar', np.nan):.2f}")

    print("\n[データ概要]")
    print(f"  サンプル数:     {metrics.get('n_samples', 0):,}")
    print(f"  日数:           {metrics.get('n_dates', 0):,}")
    print(f"  銘柄数:         {metrics.get('n_stocks', 0):,}")
    print(f"{'='*50}\n")
