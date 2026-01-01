"""
Optunaハイパーパラメータ最適化モジュール

LightGBMパラメータと特徴量選択の最適化を実装
"""
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import pandas as pd
from datetime import date
from typing import Dict, Any, List, Optional
import logging
import json
from pathlib import Path

from .model import LightGBMModel
from .walk_forward import WalkForwardCV, WalkForwardValidator
from .metrics import evaluate_all

logger = logging.getLogger(__name__)

# Optunaのログレベルを調整
optuna.logging.set_verbosity(optuna.logging.WARNING)


class HyperparameterOptimizer:
    """Optunaによるハイパーパラメータ最適化"""

    # パラメータ探索範囲
    PARAM_SPACE = {
        'num_leaves': (15, 255),
        'learning_rate': (0.01, 0.3),
        'feature_fraction': (0.4, 1.0),
        'bagging_fraction': (0.4, 1.0),
        'min_child_samples': (5, 100),
        'reg_alpha': (1e-8, 10.0),
        'reg_lambda': (1e-8, 10.0),
        'max_depth': (3, 12),
    }

    def __init__(self,
                 feature_df: pd.DataFrame,
                 label_df: pd.DataFrame,
                 feature_columns: List[str],
                 cv: Optional[WalkForwardCV] = None,
                 n_trials: int = 100,
                 timeout: Optional[int] = 3600,
                 metric: str = 'icir',
                 task_type: str = 'regression',
                 n_cv_folds: int = 3):
        """
        Args:
            feature_df: 特徴量DataFrame
            label_df: ラベルDataFrame
            feature_columns: 特徴量カラムリスト
            cv: WalkForwardCV（省略時はデフォルト作成）
            n_trials: 試行回数
            timeout: タイムアウト（秒）
            metric: 最適化指標 ('ic', 'icir', 'sharpe', 'hit_rate')
            task_type: タスクタイプ
            n_cv_folds: 最適化時に使用するCVフォールド数
        """
        self.feature_df = feature_df.copy()
        self.feature_df['date'] = pd.to_datetime(self.feature_df['date'])

        self.label_df = label_df.copy()
        self.label_df['date'] = pd.to_datetime(self.label_df['date'])

        self.feature_columns = feature_columns
        self.n_trials = n_trials
        self.timeout = timeout
        self.metric = metric
        self.task_type = task_type
        self.n_cv_folds = n_cv_folds

        # CVの設定（最適化用に短い期間を使用）
        if cv is None:
            self.cv = WalkForwardCV(
                train_period_days=504,  # 約2年
                test_period_days=63,    # 約3ヶ月
                step_days=63,           # 約3ヶ月
                embargo_days=20,
            )
        else:
            self.cv = cv

        self.study: Optional[optuna.Study] = None
        self.best_params: Optional[Dict[str, Any]] = None

    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """パラメータをサジェスト"""
        params = {
            'num_leaves': trial.suggest_int(
                'num_leaves',
                self.PARAM_SPACE['num_leaves'][0],
                self.PARAM_SPACE['num_leaves'][1]
            ),
            'learning_rate': trial.suggest_float(
                'learning_rate',
                self.PARAM_SPACE['learning_rate'][0],
                self.PARAM_SPACE['learning_rate'][1],
                log=True
            ),
            'feature_fraction': trial.suggest_float(
                'feature_fraction',
                self.PARAM_SPACE['feature_fraction'][0],
                self.PARAM_SPACE['feature_fraction'][1]
            ),
            'bagging_fraction': trial.suggest_float(
                'bagging_fraction',
                self.PARAM_SPACE['bagging_fraction'][0],
                self.PARAM_SPACE['bagging_fraction'][1]
            ),
            'min_child_samples': trial.suggest_int(
                'min_child_samples',
                self.PARAM_SPACE['min_child_samples'][0],
                self.PARAM_SPACE['min_child_samples'][1]
            ),
            'reg_alpha': trial.suggest_float(
                'reg_alpha',
                self.PARAM_SPACE['reg_alpha'][0],
                self.PARAM_SPACE['reg_alpha'][1],
                log=True
            ),
            'reg_lambda': trial.suggest_float(
                'reg_lambda',
                self.PARAM_SPACE['reg_lambda'][0],
                self.PARAM_SPACE['reg_lambda'][1],
                log=True
            ),
            'max_depth': trial.suggest_int(
                'max_depth',
                self.PARAM_SPACE['max_depth'][0],
                self.PARAM_SPACE['max_depth'][1]
            ),
        }
        return params

    def _get_metric_value(self, results_df: pd.DataFrame) -> float:
        """評価指標の値を取得"""
        metrics = evaluate_all(results_df)

        metric_map = {
            'ic': 'ic_mean',
            'icir': 'icir',
            'sharpe': 'ls_sharpe',
            'hit_rate': 'hit_rate',
            'portfolio_sharpe': 'portfolio_sharpe',
        }

        metric_key = metric_map.get(self.metric, self.metric)
        return metrics.get(metric_key, 0.0)

    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna目的関数

        Args:
            trial: Optunaトライアル

        Returns:
            最適化指標の値
        """
        params = self._suggest_params(trial)

        # 期間を決定
        dates = sorted(self.feature_df['date'].unique())
        start_date = dates[0].date()
        end_date = dates[-1].date()

        # バリデーター作成
        def model_factory():
            return LightGBMModel(params=params, task_type=self.task_type)

        validator = WalkForwardValidator(
            cv=self.cv,
            model_factory=model_factory,
            feature_df=self.feature_df,
            label_df=self.label_df,
            feature_columns=self.feature_columns,
        )

        # 検証実行
        results_df = validator.run(
            start_date=start_date,
            end_date=end_date,
            verbose=False,
        )

        if len(results_df) == 0:
            return 0.0

        # 指標計算
        score = self._get_metric_value(results_df)

        # プルーニング用に中間結果を報告
        trial.report(score, step=0)

        if trial.should_prune():
            raise optuna.TrialPruned()

        return score

    def optimize(self,
                 train_start: Optional[date] = None,
                 train_end: Optional[date] = None,
                 study_name: str = 'lightgbm_optimization',
                 storage: Optional[str] = None,
                 load_if_exists: bool = True) -> Dict[str, Any]:
        """
        最適化実行

        Args:
            train_start: 学習開始日（省略時はデータ開始日）
            train_end: 学習終了日（省略時はデータ終了日）
            study_name: Optuna study名
            storage: Optuna storage URL（省略時はインメモリ）
            load_if_exists: 既存studyを読み込むか

        Returns:
            最適パラメータ
        """
        # データ期間でフィルタ
        if train_start is not None:
            self.feature_df = self.feature_df[
                self.feature_df['date'] >= pd.to_datetime(train_start)
            ]
            self.label_df = self.label_df[
                self.label_df['date'] >= pd.to_datetime(train_start)
            ]

        if train_end is not None:
            self.feature_df = self.feature_df[
                self.feature_df['date'] <= pd.to_datetime(train_end)
            ]
            self.label_df = self.label_df[
                self.label_df['date'] <= pd.to_datetime(train_end)
            ]

        logger.info(f"最適化開始: {self.n_trials}試行, タイムアウト={self.timeout}秒")
        logger.info(f"最適化指標: {self.metric}")

        # Optuna study作成
        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction='maximize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_warmup_steps=5),
            load_if_exists=load_if_exists,
        )

        # 最適化実行
        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True,
            gc_after_trial=True,
        )

        self.best_params = self.study.best_params

        logger.info("\n最適化完了!")
        logger.info(f"最良スコア: {self.study.best_value:.4f}")
        logger.info(f"最良パラメータ: {self.best_params}")

        return self.best_params

    def get_optimization_history(self) -> pd.DataFrame:
        """最適化履歴を取得"""
        if self.study is None:
            raise ValueError("最適化が実行されていません")

        trials_data = []
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                data = {
                    'trial_number': trial.number,
                    'value': trial.value,
                    **trial.params
                }
                trials_data.append(data)

        return pd.DataFrame(trials_data).sort_values('value', ascending=False)

    def get_param_importance(self) -> pd.DataFrame:
        """パラメータ重要度を取得"""
        if self.study is None:
            raise ValueError("最適化が実行されていません")

        importance = optuna.importance.get_param_importances(self.study)

        return pd.DataFrame([
            {'parameter': k, 'importance': v}
            for k, v in importance.items()
        ]).sort_values('importance', ascending=False)

    def save_results(self, path: str):
        """最適化結果を保存"""
        if self.study is None or self.best_params is None:
            raise ValueError("最適化が実行されていません")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        results = {
            'best_params': self.best_params,
            'best_value': self.study.best_value,
            'metric': self.metric,
            'n_trials': len(self.study.trials),
            'param_importance': self.get_param_importance().to_dict('records'),
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"最適化結果保存: {path}")

    @classmethod
    def load_results(cls, path: str) -> Dict[str, Any]:
        """最適化結果を読み込み"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)


class FeatureSelector:
    """特徴量選択最適化"""

    def __init__(self,
                 feature_df: pd.DataFrame,
                 label_df: pd.DataFrame,
                 feature_groups: Dict[str, List[str]],
                 cv: Optional[WalkForwardCV] = None,
                 base_params: Optional[Dict[str, Any]] = None,
                 n_trials: int = 50,
                 metric: str = 'icir'):
        """
        Args:
            feature_df: 特徴量DataFrame
            label_df: ラベルDataFrame
            feature_groups: {グループ名: [特徴量リスト]}
            cv: WalkForwardCV
            base_params: LightGBMベースパラメータ
            n_trials: 試行回数
            metric: 最適化指標
        """
        self.feature_df = feature_df.copy()
        self.label_df = label_df.copy()
        self.feature_groups = feature_groups
        self.base_params = base_params or {}
        self.n_trials = n_trials
        self.metric = metric

        if cv is None:
            self.cv = WalkForwardCV(
                train_period_days=504,
                test_period_days=63,
                step_days=63,
                embargo_days=20,
            )
        else:
            self.cv = cv

        self.study: Optional[optuna.Study] = None
        self.best_features: Optional[List[str]] = None

    def objective(self, trial: optuna.Trial) -> float:
        """特徴量選択の目的関数"""
        # グループごとに使用/不使用を選択
        selected_features = []

        for group_name, features in self.feature_groups.items():
            use_group = trial.suggest_categorical(f'use_{group_name}', [True, False])
            if use_group:
                selected_features.extend(features)

        if len(selected_features) == 0:
            return 0.0

        # バリデーター作成
        def model_factory():
            return LightGBMModel(params=self.base_params, task_type='regression')

        validator = WalkForwardValidator(
            cv=self.cv,
            model_factory=model_factory,
            feature_df=self.feature_df,
            label_df=self.label_df,
            feature_columns=selected_features,
        )

        # 期間を決定
        dates = sorted(self.feature_df['date'].unique())
        start_date = dates[0].date()
        end_date = dates[-1].date()

        # 検証実行
        results_df = validator.run(
            start_date=start_date,
            end_date=end_date,
            verbose=False,
        )

        if len(results_df) == 0:
            return 0.0

        # スコア計算
        metrics = evaluate_all(results_df)
        metric_map = {
            'ic': 'ic_mean',
            'icir': 'icir',
            'sharpe': 'ls_sharpe',
        }
        return metrics.get(metric_map.get(self.metric, self.metric), 0.0)

    def optimize(self) -> List[str]:
        """特徴量選択最適化を実行"""
        logger.info(f"特徴量選択最適化開始: {self.n_trials}試行")

        self.study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
        )

        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            show_progress_bar=True,
        )

        # 最良の特徴量グループを取得
        best_groups = []
        for group_name in self.feature_groups.keys():
            if self.study.best_params.get(f'use_{group_name}', False):
                best_groups.append(group_name)

        self.best_features = []
        for group in best_groups:
            self.best_features.extend(self.feature_groups[group])

        logger.info("\n最適化完了!")
        logger.info(f"最良スコア: {self.study.best_value:.4f}")
        logger.info(f"選択グループ: {best_groups}")
        logger.info(f"特徴量数: {len(self.best_features)}")

        return self.best_features

    def get_group_importance(self) -> pd.DataFrame:
        """グループ別の選択頻度を取得"""
        if self.study is None:
            raise ValueError("最適化が実行されていません")

        group_counts = {group: 0 for group in self.feature_groups}

        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                for group_name in self.feature_groups:
                    if trial.params.get(f'use_{group_name}', False):
                        group_counts[group_name] += 1

        n_trials = len([t for t in self.study.trials
                       if t.state == optuna.trial.TrialState.COMPLETE])

        return pd.DataFrame([
            {
                'group': k,
                'selection_count': v,
                'selection_rate': v / n_trials if n_trials > 0 else 0,
                'feature_count': len(self.feature_groups[k]),
            }
            for k, v in group_counts.items()
        ]).sort_values('selection_rate', ascending=False)
