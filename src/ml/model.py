"""
LightGBMモデルラッパー

モデルの学習、予測、保存/読み込み、特徴量重要度分析を担当
"""
import json
import lightgbm as lgb
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class LightGBMModel:
    """LightGBMモデルラッパー"""

    DEFAULT_PARAMS = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42,
        'bagging_seed': 42,
        'feature_fraction_seed': 42,
        'n_jobs': -1,
    }

    TASK_TYPE_PARAMS = {
        'regression': {
            'objective': 'regression',
            'metric': 'rmse',
        },
        'binary': {
            'objective': 'binary',
            'metric': 'binary_logloss',
        },
        'ranking': {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [5, 10, 20],
        },
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None,
                 task_type: str = 'regression'):
        """
        Args:
            params: LightGBMパラメータ（Noneならデフォルト使用）
            task_type: 'regression', 'binary', 'ranking'
        """
        self.task_type = task_type

        # デフォルトパラメータをベースに構築
        self.params = {**self.DEFAULT_PARAMS}

        # タスクタイプ固有のパラメータを追加
        if task_type in self.TASK_TYPE_PARAMS:
            self.params.update(self.TASK_TYPE_PARAMS[task_type])

        # ユーザー指定パラメータで上書き
        if params:
            self.params.update(params)

        self.model: Optional[lgb.Booster] = None
        self.feature_names: Optional[List[str]] = None
        self.best_iteration: Optional[int] = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_valid: Optional[pd.DataFrame] = None,
            y_valid: Optional[pd.Series] = None,
            num_boost_round: int = 1000,
            early_stopping_rounds: int = 50,
            categorical_features: Optional[List[str]] = None,
            group_train: Optional[np.ndarray] = None,
            group_valid: Optional[np.ndarray] = None) -> 'LightGBMModel':
        """
        モデル学習

        Args:
            X_train: 学習特徴量
            y_train: 学習ラベル
            X_valid: 検証特徴量（Early Stopping用）
            y_valid: 検証ラベル
            num_boost_round: 最大イテレーション数
            early_stopping_rounds: Early Stoppingラウンド数
            categorical_features: カテゴリカル特徴量名リスト
            group_train: ランキング用グループサイズ（学習）
            group_valid: ランキング用グループサイズ（検証）

        Returns:
            自身（メソッドチェーン用）
        """
        self.feature_names = list(X_train.columns)

        # データセット作成
        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            categorical_feature=categorical_features,
            group=group_train
        )

        valid_sets = [train_data]
        valid_names = ['train']

        if X_valid is not None and y_valid is not None:
            valid_data = lgb.Dataset(
                X_valid,
                label=y_valid,
                reference=train_data,
                group=group_valid
            )
            valid_sets.append(valid_data)
            valid_names.append('valid')

        # コールバック設定
        callbacks = []
        if X_valid is not None:
            callbacks.append(lgb.early_stopping(early_stopping_rounds))
        callbacks.append(lgb.log_evaluation(period=100))

        # 学習
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        self.best_iteration = self.model.best_iteration

        logger.info(f"学習完了: best_iteration={self.best_iteration}")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        予測

        Args:
            X: 特徴量DataFrame

        Returns:
            予測値の配列
        """
        if self.model is None:
            raise ValueError("Model not trained")

        # 特徴量の順序を揃える
        X_aligned = X[self.feature_names]

        return self.model.predict(X_aligned, num_iteration=self.best_iteration)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        確率予測（binary分類用）

        Args:
            X: 特徴量DataFrame

        Returns:
            確率値の配列
        """
        if self.task_type != 'binary':
            logger.warning("predict_proba is designed for binary classification")

        return self.predict(X)

    def feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        特徴量重要度

        Args:
            importance_type: 'gain' or 'split'

        Returns:
            特徴量重要度DataFrame
        """
        if self.model is None:
            raise ValueError("Model not trained")

        importance = self.model.feature_importance(importance_type=importance_type)

        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).reset_index(drop=True)

    def feature_importance_by_group(self,
                                     feature_groups: Dict[str, List[str]],
                                     importance_type: str = 'gain') -> pd.DataFrame:
        """
        カテゴリ別特徴量重要度

        Args:
            feature_groups: {グループ名: [特徴量名リスト]}
            importance_type: 'gain' or 'split'

        Returns:
            グループ別重要度DataFrame
        """
        fi = self.feature_importance(importance_type)
        total_importance = fi['importance'].sum()

        group_importance = {}
        for group_name, features in feature_groups.items():
            mask = fi['feature'].isin(features)
            group_importance[group_name] = {
                'importance': fi.loc[mask, 'importance'].sum(),
                'feature_count': len(features),
                'used_features': mask.sum(),
            }

        result = pd.DataFrame([
            {
                'group': k,
                'importance': v['importance'],
                'importance_ratio': v['importance'] / total_importance if total_importance > 0 else 0,
                'feature_count': v['feature_count'],
                'used_features': v['used_features'],
            }
            for k, v in group_importance.items()
        ]).sort_values('importance', ascending=False).reset_index(drop=True)

        return result

    def top_features(self, n: int = 20, importance_type: str = 'gain') -> List[str]:
        """
        上位N特徴量を取得

        Args:
            n: 取得する特徴量数
            importance_type: 'gain' or 'split'

        Returns:
            特徴量名リスト
        """
        fi = self.feature_importance(importance_type)
        return fi.head(n)['feature'].tolist()

    def save(self, path: str, save_metadata: bool = True):
        """
        モデル保存

        Args:
            path: 保存先パス（.lgb）
            save_metadata: メタデータも保存するか
        """
        if self.model is None:
            raise ValueError("Model not trained")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # モデル保存
        self.model.save_model(str(path))
        logger.info(f"モデル保存: {path}")

        # メタデータ保存
        if save_metadata:
            metadata = {
                'task_type': self.task_type,
                'params': self.params,
                'feature_names': self.feature_names,
                'best_iteration': self.best_iteration,
            }
            metadata_path = path.with_suffix('.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"メタデータ保存: {metadata_path}")

    @classmethod
    def load(cls, path: str) -> 'LightGBMModel':
        """
        モデル読み込み

        Args:
            path: モデルパス（.lgb）

        Returns:
            読み込んだLightGBMModelインスタンス
        """
        path = Path(path)

        # メタデータ読み込み
        metadata_path = path.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            instance = cls(
                params=metadata.get('params'),
                task_type=metadata.get('task_type', 'regression')
            )
            instance.feature_names = metadata.get('feature_names')
            instance.best_iteration = metadata.get('best_iteration')
        else:
            instance = cls()

        # モデル読み込み
        instance.model = lgb.Booster(model_file=str(path))

        # 特徴量名がない場合はモデルから取得
        if instance.feature_names is None:
            instance.feature_names = instance.model.feature_name()

        logger.info(f"モデル読み込み: {path}")
        return instance

    def get_params(self) -> Dict[str, Any]:
        """パラメータを取得"""
        return self.params.copy()

    def set_params(self, **params):
        """パラメータを設定"""
        self.params.update(params)

    def clone(self) -> 'LightGBMModel':
        """パラメータを引き継いだ新しいインスタンスを作成"""
        return LightGBMModel(params=self.params.copy(), task_type=self.task_type)
