"""
ウォークフォワード検証モジュール

時系列データのための適切なクロスバリデーション手法を実装
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import date
from typing import Generator, Callable, Dict, List, Optional, Any
import logging

from .model import LightGBMModel
from .metrics import evaluate_all

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardSplit:
    """ウォークフォワード分割情報"""
    fold_id: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date

    def __str__(self) -> str:
        return (f"Fold {self.fold_id}: "
                f"Train[{self.train_start} - {self.train_end}] -> "
                f"Test[{self.test_start} - {self.test_end}]")


class WalkForwardCV:
    """
    ウォークフォワードクロスバリデーション

    時系列分割:
    |----Train----|--Gap--|--Test--|
                  |----Train----|--Gap--|--Test--|
                                |----Train----|--Gap--|--Test--|
    """

    def __init__(self,
                 train_period_days: int = 756,
                 test_period_days: int = 63,
                 step_days: int = 21,
                 embargo_days: int = 20,
                 expanding: bool = False):
        """
        Args:
            train_period_days: 学習期間（営業日）- デフォルト約3年
            test_period_days: テスト期間（営業日）- デフォルト約3ヶ月
            step_days: スライドステップ（営業日）- デフォルト約1ヶ月
            embargo_days: 学習・テスト間のギャップ（ラベルリーク防止）
            expanding: Trueの場合、学習期間を拡張していく（固定長でなく）
        """
        self.train_period = train_period_days
        self.test_period = test_period_days
        self.step = step_days
        self.embargo = embargo_days
        self.expanding = expanding

    def split(self,
              start_date: date,
              end_date: date,
              trading_calendar: Optional[pd.DataFrame] = None) -> Generator[WalkForwardSplit, None, None]:
        """
        分割を生成

        Args:
            start_date: データ開始日
            end_date: データ終了日
            trading_calendar: 取引カレンダー（省略時は全日を取引日とみなす）

        Yields:
            WalkForwardSplit
        """
        if trading_calendar is not None:
            # 取引日のリストを取得
            calendar = trading_calendar.copy()
            calendar['date'] = pd.to_datetime(calendar['date'])
            trading_days = calendar[
                (calendar['date'] >= pd.to_datetime(start_date)) &
                (calendar['date'] <= pd.to_datetime(end_date)) &
                (calendar['is_trading_day'])
            ]['date'].dt.date.tolist()
        else:
            # カレンダーがない場合は単純に日数で計算
            trading_days = pd.date_range(start_date, end_date, freq='B').date.tolist()

        if len(trading_days) == 0:
            logger.warning("取引日がありません")
            return

        n_days = len(trading_days)
        fold_id = 0

        # 最初の学習期間開始位置
        if self.expanding:
            train_start_idx = 0
        else:
            train_start_idx = 0

        while True:
            # 学習期間
            if self.expanding:
                # 拡張モード: 常に最初から
                train_start_idx = 0
            train_end_idx = train_start_idx + self.train_period - 1

            # テスト期間
            test_start_idx = train_end_idx + self.embargo + 1
            test_end_idx = test_start_idx + self.test_period - 1

            # 範囲チェック
            if test_end_idx >= n_days:
                break

            split = WalkForwardSplit(
                fold_id=fold_id,
                train_start=trading_days[train_start_idx],
                train_end=trading_days[train_end_idx],
                test_start=trading_days[test_start_idx],
                test_end=trading_days[test_end_idx],
            )

            yield split

            fold_id += 1

            # 次のフォールドへ
            if self.expanding:
                # 拡張モード: 学習期間を延長
                train_end_idx += self.step
            else:
                # 固定長モード: スライド
                train_start_idx += self.step

    def get_n_splits(self, start_date: date, end_date: date,
                     trading_calendar: Optional[pd.DataFrame] = None) -> int:
        """分割数を取得"""
        return sum(1 for _ in self.split(start_date, end_date, trading_calendar))


class WalkForwardValidator:
    """ウォークフォワード検証実行"""

    def __init__(self,
                 cv: WalkForwardCV,
                 model_factory: Callable[[], LightGBMModel],
                 feature_df: pd.DataFrame,
                 label_df: pd.DataFrame,
                 feature_columns: List[str]):
        """
        Args:
            cv: WalkForwardCVインスタンス
            model_factory: モデル生成関数
            feature_df: 特徴量DataFrame
            label_df: ラベルDataFrame（code, date, target列を含む）
            feature_columns: 使用する特徴量カラムリスト
        """
        self.cv = cv
        self.model_factory = model_factory
        self.feature_df = feature_df.copy()
        self.feature_df['date'] = pd.to_datetime(self.feature_df['date'])

        self.label_df = label_df.copy()
        self.label_df['date'] = pd.to_datetime(self.label_df['date'])

        self.feature_columns = feature_columns

        self.fold_models: List[LightGBMModel] = []
        self.fold_results: List[pd.DataFrame] = []
        self.fold_metrics: List[Dict[str, float]] = []

    def _prepare_data(self, split: WalkForwardSplit) -> tuple:
        """フォールド用のデータを準備"""
        # 特徴量とラベルをマージ
        df = self.feature_df.merge(
            self.label_df[['code', 'date', 'target']],
            on=['code', 'date'],
            how='inner'
        )

        # 日付でフィルタ
        train_mask = (
            (df['date'] >= pd.to_datetime(split.train_start)) &
            (df['date'] <= pd.to_datetime(split.train_end))
        )
        test_mask = (
            (df['date'] >= pd.to_datetime(split.test_start)) &
            (df['date'] <= pd.to_datetime(split.test_end))
        )

        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()

        # 欠損値処理（中央値埋め）
        for col in self.feature_columns:
            if col in train_df.columns:
                median_val = train_df[col].median()
                train_df[col] = train_df[col].fillna(median_val)
                test_df[col] = test_df[col].fillna(median_val)

        # 無限値処理
        train_df = train_df.replace([np.inf, -np.inf], np.nan)
        test_df = test_df.replace([np.inf, -np.inf], np.nan)

        # 欠損が残っている場合は0埋め
        train_df = train_df.fillna(0)
        test_df = test_df.fillna(0)

        return train_df, test_df

    def run(self,
            start_date: date,
            end_date: date,
            trading_calendar: Optional[pd.DataFrame] = None,
            validation_ratio: float = 0.2,
            verbose: bool = True) -> pd.DataFrame:
        """
        ウォークフォワード検証実行

        Args:
            start_date: 検証開始日
            end_date: 検証終了日
            trading_calendar: 取引カレンダー
            validation_ratio: 学習データ内の検証データ比率
            verbose: 詳細出力

        Returns:
            各フォールドの予測結果を結合したDataFrame
            columns: [code, date, y_true, y_pred, fold_id]
        """
        self.fold_models = []
        self.fold_results = []
        self.fold_metrics = []

        n_splits = self.cv.get_n_splits(start_date, end_date, trading_calendar)

        if verbose:
            logger.info(f"ウォークフォワード検証開始: {n_splits}フォールド")

        for split in self.cv.split(start_date, end_date, trading_calendar):
            if verbose:
                logger.info(f"\n{split}")

            # データ準備
            train_df, test_df = self._prepare_data(split)

            if len(train_df) == 0 or len(test_df) == 0:
                logger.warning(f"Fold {split.fold_id}: データなし、スキップ")
                continue

            # 学習データを学習/検証に分割
            train_dates = sorted(train_df['date'].unique())
            val_split_idx = int(len(train_dates) * (1 - validation_ratio))
            val_start_date = train_dates[val_split_idx]

            val_mask = train_df['date'] >= val_start_date
            X_train = train_df[~val_mask][self.feature_columns]
            y_train = train_df[~val_mask]['target']
            X_valid = train_df[val_mask][self.feature_columns]
            y_valid = train_df[val_mask]['target']

            X_test = test_df[self.feature_columns]
            y_test = test_df['target']

            # モデル学習
            model = self.model_factory()
            model.fit(
                X_train, y_train,
                X_valid=X_valid if len(X_valid) > 0 else None,
                y_valid=y_valid if len(y_valid) > 0 else None,
            )

            # 予測
            y_pred = model.predict(X_test)

            # 結果保存
            result_df = pd.DataFrame({
                'code': test_df['code'].values,
                'date': test_df['date'].values,
                'y_true': y_test.values,
                'y_pred': y_pred,
                'fold_id': split.fold_id,
            })

            self.fold_models.append(model)
            self.fold_results.append(result_df)

            # フォールド評価
            fold_metrics = evaluate_all(result_df)
            self.fold_metrics.append(fold_metrics)

            if verbose:
                logger.info(f"  Train: {len(X_train)}, Test: {len(X_test)}")
                logger.info(f"  IC: {fold_metrics['ic_mean']:.4f}, "
                           f"ICIR: {fold_metrics['icir']:.4f}, "
                           f"Hit: {fold_metrics['hit_rate']:.1%}")

        if len(self.fold_results) == 0:
            logger.error("有効なフォールドがありませんでした")
            return pd.DataFrame()

        # 結果を結合
        all_results = pd.concat(self.fold_results, ignore_index=True)

        if verbose:
            logger.info(f"\n検証完了: 合計{len(all_results)}サンプル")

        return all_results

    def evaluate(self, results_df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        全体評価

        Args:
            results_df: 予測結果DataFrame（省略時は内部結果を使用）

        Returns:
            評価指標の辞書
        """
        if results_df is None:
            if len(self.fold_results) == 0:
                raise ValueError("検証が実行されていません")
            results_df = pd.concat(self.fold_results, ignore_index=True)

        return evaluate_all(results_df)

    def evaluate_by_period(self, results_df: Optional[pd.DataFrame] = None,
                           freq: str = 'M') -> pd.DataFrame:
        """
        期間別評価

        Args:
            results_df: 予測結果DataFrame
            freq: 集計頻度（'M': 月次, 'Q': 四半期, 'Y': 年次）

        Returns:
            期間別評価指標DataFrame
        """
        if results_df is None:
            if len(self.fold_results) == 0:
                raise ValueError("検証が実行されていません")
            results_df = pd.concat(self.fold_results, ignore_index=True)

        results_df = results_df.copy()
        results_df['period'] = pd.to_datetime(results_df['date']).dt.to_period(freq)

        period_metrics = []
        for period, group in results_df.groupby('period'):
            metrics = evaluate_all(group)
            metrics['period'] = str(period)
            period_metrics.append(metrics)

        return pd.DataFrame(period_metrics)

    def evaluate_by_fold(self) -> pd.DataFrame:
        """
        フォールド別評価

        Returns:
            フォールド別評価指標DataFrame
        """
        if len(self.fold_metrics) == 0:
            raise ValueError("検証が実行されていません")

        df = pd.DataFrame(self.fold_metrics)
        df.insert(0, 'fold_id', range(len(df)))
        return df

    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        全フォールドの平均特徴量重要度

        Args:
            importance_type: 'gain' or 'split'

        Returns:
            特徴量重要度DataFrame
        """
        if len(self.fold_models) == 0:
            raise ValueError("検証が実行されていません")

        # 各フォールドの重要度を取得
        importance_dfs = []
        for model in self.fold_models:
            fi = model.feature_importance(importance_type)
            importance_dfs.append(fi)

        # 平均を計算
        merged = importance_dfs[0].copy()
        for df in importance_dfs[1:]:
            merged = merged.merge(df, on='feature', suffixes=('', '_tmp'))
            merged['importance'] = merged['importance'] + merged['importance_tmp']
            merged = merged.drop(columns=['importance_tmp'])

        merged['importance'] = merged['importance'] / len(importance_dfs)

        return merged.sort_values('importance', ascending=False).reset_index(drop=True)

    def get_feature_importance_by_group(self,
                                         feature_groups: Dict[str, List[str]],
                                         importance_type: str = 'gain') -> pd.DataFrame:
        """
        特徴量グループ別の重要度分析

        Args:
            feature_groups: {グループ名: [特徴量名リスト]}
            importance_type: 'gain' or 'split'

        Returns:
            グループ別重要度DataFrame
        """
        fi = self.get_feature_importance(importance_type)
        total_importance = fi['importance'].sum()

        group_data = []
        for group_name, features in feature_groups.items():
            mask = fi['feature'].isin(features)
            group_importance = fi.loc[mask, 'importance'].sum()

            group_data.append({
                'group': group_name,
                'importance': group_importance,
                'importance_ratio': group_importance / total_importance if total_importance > 0 else 0,
                'feature_count': len(features),
                'used_features': mask.sum(),
            })

        return pd.DataFrame(group_data).sort_values('importance', ascending=False).reset_index(drop=True)

    def get_best_model(self, metric: str = 'ic_mean') -> LightGBMModel:
        """
        最良のフォールドのモデルを取得

        Args:
            metric: 評価指標名

        Returns:
            最良モデル
        """
        if len(self.fold_models) == 0:
            raise ValueError("検証が実行されていません")

        fold_df = self.evaluate_by_fold()
        best_idx = fold_df[metric].idxmax()

        return self.fold_models[best_idx]


def create_simple_validator(
    feature_df: pd.DataFrame,
    label_df: pd.DataFrame,
    feature_columns: List[str],
    train_years: int = 3,
    test_months: int = 3,
    step_months: int = 1,
    embargo_days: int = 20,
    task_type: str = 'regression',
    params: Optional[Dict[str, Any]] = None
) -> WalkForwardValidator:
    """
    簡易バリデーター作成

    Args:
        feature_df: 特徴量DataFrame
        label_df: ラベルDataFrame
        feature_columns: 特徴量カラムリスト
        train_years: 学習期間（年）
        test_months: テスト期間（月）
        step_months: スライドステップ（月）
        embargo_days: エンバーゴ期間
        task_type: タスクタイプ
        params: LightGBMパラメータ

    Returns:
        WalkForwardValidatorインスタンス
    """
    cv = WalkForwardCV(
        train_period_days=train_years * 252,
        test_period_days=test_months * 21,
        step_days=step_months * 21,
        embargo_days=embargo_days,
    )

    def model_factory():
        return LightGBMModel(params=params, task_type=task_type)

    return WalkForwardValidator(
        cv=cv,
        model_factory=model_factory,
        feature_df=feature_df,
        label_df=label_df,
        feature_columns=feature_columns,
    )
