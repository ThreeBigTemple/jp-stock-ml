# JP Stock ML - Phase 3: MLパイプライン実装プロンプト

## 前提条件

Phase 1（データ基盤）とPhase 2（特徴量エンジニアリング）が完了済み。

### 利用可能なデータ

```
data/jp_stock.db  # SQLiteデータベース
├── stocks        # 銘柄マスタ（約4,000銘柄）
├── prices        # 日次株価（過去10年分）
├── financials    # 財務データ（Point-in-Time管理）
├── topix         # TOPIX日次データ
├── margin_balance    # 信用取引残高
└── short_selling     # 業種別空売り比率
```

### 利用可能な特徴量（Phase 2で実装済み想定）

```python
# src/features/ 配下
from src.features import FeatureBuilder

builder = FeatureBuilder(session)
feature_df = builder.build(start_date, end_date)
# columns: [code, date, feature_1, ..., feature_N]
```

## Phase 3 タスク: MLパイプライン構築

### 実装すべきファイル構造

```
src/ml/
├── __init__.py
├── dataset.py        # データセット作成（ラベル生成、train/test分割）
├── model.py          # LightGBMモデルラッパー
├── trainer.py        # 学習・評価ロジック
├── walk_forward.py   # ウォークフォワード検証
├── optimizer.py      # Optunaハイパーパラメータ最適化
└── metrics.py        # 評価指標（IC、シャープレシオ等）

scripts/
├── train_model.py    # モデル学習スクリプト
├── optimize_params.py # ハイパラ最適化スクリプト
└── predict.py        # 予測実行スクリプト
```

---

## 1. dataset.py - データセット作成

### ラベル（ターゲット）定義

```python
class LabelGenerator:
    """予測ターゲット生成"""
    
    def __init__(self, prices_df: pd.DataFrame):
        self.prices = prices_df
    
    def forward_return(self, days: int = 20) -> pd.Series:
        """
        N営業日後リターン（回帰用）
        
        Args:
            days: 保有期間（営業日）
        
        Returns:
            N日後リターン（%）
        """
        pass
    
    def forward_return_binary(self, days: int = 20, 
                               threshold: float = 0.0) -> pd.Series:
        """
        N営業日後リターンの2値分類
        
        Args:
            days: 保有期間
            threshold: 閾値（例: 0.05 = 5%以上で1）
        
        Returns:
            0/1ラベル
        """
        pass
    
    def forward_return_quintile(self, days: int = 20) -> pd.Series:
        """
        N営業日後リターンの5分位ラベル（ランキング用）
        
        各日付でクロスセクショナルに5分位に分類
        
        Returns:
            0-4のラベル（4が最上位）
        """
        pass
```

### データセット分割

```python
class DatasetBuilder:
    """学習用データセット構築"""
    
    def __init__(self, feature_df: pd.DataFrame, label_series: pd.Series):
        """
        Args:
            feature_df: 特徴量DataFrame [code, date, features...]
            label_series: ラベルSeries（indexはfeature_dfと対応）
        """
        pass
    
    def create_dataset(self, 
                       train_start: date,
                       train_end: date,
                       test_start: date,
                       test_end: date,
                       min_samples: int = 100) -> Tuple[Dataset, Dataset]:
        """
        学習・テストデータセット作成
        
        Args:
            train_start/end: 学習期間
            test_start/end: テスト期間
            min_samples: 銘柄あたり最小サンプル数
        
        Returns:
            (train_dataset, test_dataset)
        """
        pass
    
    def filter_stocks(self, 
                      min_price: float = 100,
                      min_volume: float = 10000,
                      exclude_sectors: List[str] = None) -> 'DatasetBuilder':
        """
        銘柄フィルタリング
        
        - 低位株除外
        - 流動性フィルタ
        - セクター除外（金融等）
        """
        pass
```

---

## 2. model.py - LightGBMモデル

```python
import lightgbm as lgb
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


class LightGBMModel:
    """LightGBMモデルラッパー"""
    
    DEFAULT_PARAMS = {
        'objective': 'regression',  # or 'binary', 'lambdarank'
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42,
    }
    
    def __init__(self, params: Optional[Dict[str, Any]] = None,
                 task_type: str = 'regression'):
        """
        Args:
            params: LightGBMパラメータ
            task_type: 'regression', 'binary', 'ranking'
        """
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.task_type = task_type
        self.model: Optional[lgb.Booster] = None
        self.feature_names: Optional[List[str]] = None
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_valid: Optional[pd.DataFrame] = None,
            y_valid: Optional[pd.Series] = None,
            num_boost_round: int = 1000,
            early_stopping_rounds: int = 50) -> 'LightGBMModel':
        """モデル学習"""
        pass
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """予測"""
        pass
    
    def feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """特徴量重要度"""
        pass
    
    def save(self, path: str):
        """モデル保存"""
        pass
    
    @classmethod
    def load(cls, path: str) -> 'LightGBMModel':
        """モデル読み込み"""
        pass
```

---

## 3. walk_forward.py - ウォークフォワード検証

### 検証方式

```
時間軸 →
|----Train----|--Test--|
              |----Train----|--Test--|
                            |----Train----|--Test--|

パラメータ例:
- train_period: 3年（756営業日）
- test_period: 3ヶ月（63営業日）
- step: 1ヶ月（21営業日）
- embargo: 5営業日（ラベルリーク防止）
```

```python
from dataclasses import dataclass
from typing import Generator, Tuple
from datetime import date


@dataclass
class WalkForwardSplit:
    """ウォークフォワード分割情報"""
    fold_id: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date


class WalkForwardCV:
    """ウォークフォワードクロスバリデーション"""
    
    def __init__(self,
                 train_period_days: int = 756,    # 約3年
                 test_period_days: int = 63,      # 約3ヶ月
                 step_days: int = 21,             # 約1ヶ月
                 embargo_days: int = 5):          # ギャップ期間
        """
        Args:
            train_period_days: 学習期間（営業日）
            test_period_days: テスト期間（営業日）
            step_days: スライドステップ（営業日）
            embargo_days: 学習・テスト間のギャップ（ラベルリーク防止）
        """
        self.train_period = train_period_days
        self.test_period = test_period_days
        self.step = step_days
        self.embargo = embargo_days
    
    def split(self, 
              start_date: date, 
              end_date: date,
              trading_calendar: pd.DataFrame) -> Generator[WalkForwardSplit, None, None]:
        """
        分割を生成
        
        Args:
            start_date: 全体の開始日
            end_date: 全体の終了日
            trading_calendar: 取引カレンダー
        
        Yields:
            WalkForwardSplit
        """
        pass
    
    def get_n_splits(self, start_date: date, end_date: date,
                     trading_calendar: pd.DataFrame) -> int:
        """分割数を取得"""
        pass


class WalkForwardValidator:
    """ウォークフォワード検証実行"""
    
    def __init__(self, 
                 cv: WalkForwardCV,
                 model_factory: Callable[[], LightGBMModel],
                 dataset_builder: DatasetBuilder):
        self.cv = cv
        self.model_factory = model_factory
        self.dataset_builder = dataset_builder
    
    def run(self, 
            start_date: date,
            end_date: date) -> pd.DataFrame:
        """
        ウォークフォワード検証実行
        
        Returns:
            各フォールドの予測結果を結合したDataFrame
            columns: [code, date, y_true, y_pred, fold_id]
        """
        pass
    
    def evaluate(self, results_df: pd.DataFrame) -> Dict[str, float]:
        """
        全体評価
        
        Returns:
            {'ic': 0.05, 'icir': 0.8, 'sharpe': 1.2, ...}
        """
        pass
```

---

## 4. optimizer.py - Optunaハイパーパラメータ最適化

```python
import optuna
from optuna.samplers import TPESampler


class HyperparameterOptimizer:
    """Optunaによるハイパーパラメータ最適化"""
    
    def __init__(self,
                 dataset_builder: DatasetBuilder,
                 cv: WalkForwardCV,
                 n_trials: int = 100,
                 timeout: int = 3600,
                 metric: str = 'ic'):
        """
        Args:
            dataset_builder: データセットビルダー
            cv: クロスバリデーション
            n_trials: 試行回数
            timeout: タイムアウト（秒）
            metric: 最適化指標 ('ic', 'icir', 'sharpe')
        """
        self.dataset_builder = dataset_builder
        self.cv = cv
        self.n_trials = n_trials
        self.timeout = timeout
        self.metric = metric
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna目的関数
        
        探索するパラメータ:
        - num_leaves: [15, 255]
        - learning_rate: [0.01, 0.3]
        - feature_fraction: [0.5, 1.0]
        - bagging_fraction: [0.5, 1.0]
        - min_child_samples: [5, 100]
        - reg_alpha: [0, 10]
        - reg_lambda: [0, 10]
        """
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 15, 255),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        }
        
        # ウォークフォワードで評価
        # ...
        
        return score
    
    def optimize(self, 
                 train_start: date,
                 train_end: date) -> Dict[str, Any]:
        """
        最適化実行
        
        Returns:
            最適パラメータ
        """
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout
        )
        
        return study.best_params
```

---

## 5. metrics.py - 評価指標

```python
import numpy as np
import pandas as pd
from scipy import stats


def information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    情報係数（IC）- スピアマン順位相関
    
    ファクター投資の標準的な評価指標
    """
    return stats.spearmanr(y_true, y_pred)[0]


def ic_by_date(results_df: pd.DataFrame) -> pd.Series:
    """
    日次IC
    
    Args:
        results_df: [date, y_true, y_pred]を含むDataFrame
    
    Returns:
        日付をインデックスとしたIC Series
    """
    return results_df.groupby('date').apply(
        lambda x: information_coefficient(x['y_true'], x['y_pred'])
    )


def icir(ic_series: pd.Series) -> float:
    """
    ICIR（IC Information Ratio）
    
    IC_mean / IC_std
    目安: 0.5以上で良好
    """
    return ic_series.mean() / ic_series.std()


def hit_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    ヒット率（方向正解率）
    
    予測方向が正しかった割合
    """
    return np.mean(np.sign(y_true) == np.sign(y_pred))


def top_bottom_return(results_df: pd.DataFrame, 
                      n_quantiles: int = 5) -> pd.DataFrame:
    """
    ロング・ショートリターン分析
    
    予測スコア上位/下位のリターンを比較
    
    Returns:
        quantile別の平均リターン
    """
    results_df['quantile'] = results_df.groupby('date')['y_pred'].transform(
        lambda x: pd.qcut(x, n_quantiles, labels=False, duplicates='drop')
    )
    return results_df.groupby('quantile')['y_true'].mean()


def simulated_sharpe(results_df: pd.DataFrame,
                     top_n: int = 50,
                     holding_days: int = 20,
                     transaction_cost: float = 0.001) -> float:
    """
    シミュレーションシャープレシオ
    
    上位N銘柄を等金額で保有した場合のシャープレシオ
    
    Args:
        results_df: 予測結果
        top_n: 保有銘柄数
        holding_days: 保有期間
        transaction_cost: 取引コスト（往復）
    
    Returns:
        年率シャープレシオ
    """
    pass


def max_drawdown(returns: pd.Series) -> float:
    """最大ドローダウン"""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()
```

---

## 6. スクリプト

### scripts/train_model.py

```python
#!/usr/bin/env python
"""
モデル学習スクリプト

使用方法:
    uv run python scripts/train_model.py --config config/model_config.yaml
"""
import argparse
from datetime import date
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-start', type=str, default='2015-01-01')
    parser.add_argument('--train-end', type=str, default='2023-12-31')
    parser.add_argument('--output-dir', type=str, default='models/')
    parser.add_argument('--task', type=str, choices=['regression', 'binary', 'ranking'],
                        default='regression')
    args = parser.parse_args()
    
    # 1. データロード
    # 2. 特徴量構築
    # 3. モデル学習
    # 4. 評価
    # 5. モデル保存


if __name__ == '__main__':
    main()
```

### scripts/predict.py

```python
#!/usr/bin/env python
"""
予測実行スクリプト

使用方法:
    uv run python scripts/predict.py --date 2024-12-06 --top-n 50
"""
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, required=True, help='予測基準日')
    parser.add_argument('--model-path', type=str, default='models/latest.lgb')
    parser.add_argument('--top-n', type=int, default=50, help='出力銘柄数')
    parser.add_argument('--output', type=str, default='predictions/')
    args = parser.parse_args()
    
    # 1. モデルロード
    # 2. 最新特徴量生成
    # 3. 予測実行
    # 4. 上位銘柄出力（CSV）


if __name__ == '__main__':
    main()
```

---

## 実装上の注意点

### 1. メモリ管理
```python
# 大量データを扱うため、チャンク処理を検討
# 特徴量DataFrameは数GB規模になりうる
```

### 2. 再現性
```python
# 乱数シード固定
import numpy as np
np.random.seed(42)

# LightGBMのseed
params['seed'] = 42
params['bagging_seed'] = 42
params['feature_fraction_seed'] = 42
```

### 3. 欠損値処理
```python
# LightGBMは欠損値を扱えるが、
# 極端な欠損（80%以上）は特徴量から除外を検討
```

### 4. クラス不均衡（分類タスク）
```python
# 成長銘柄は少数派なので、
# scale_pos_weight や class_weight を調整
```

---

## 評価目標

| 指標 | 目標値 | 説明 |
|------|--------|------|
| IC | > 0.03 | 日次情報係数 |
| ICIR | > 0.5 | IC安定性 |
| Hit Rate | > 52% | 方向正解率 |
| Top-Bottom Spread | > 2%/月 | 上位下位リターン差 |
| Sharpe Ratio | > 1.0 | 年率（コスト控除後） |

---

## 実行コマンド

```bash
# 環境確認
uv sync

# モデル学習（ウォークフォワード検証付き）
uv run python scripts/train_model.py \
    --train-start 2015-01-01 \
    --train-end 2023-12-31 \
    --task regression

# ハイパーパラメータ最適化
uv run python scripts/optimize_params.py \
    --n-trials 100 \
    --timeout 7200

# 予測実行
uv run python scripts/predict.py \
    --date 2024-12-06 \
    --top-n 50
```
