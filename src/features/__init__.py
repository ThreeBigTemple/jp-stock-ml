# Features Package - Phase 2 特徴量エンジニアリング
# EDINET、TDnet、海外指数、Google Trends対応

from .technical import TechnicalFeatures, calculate_sector_rank
from .fundamental import (
    FundamentalFeatures,
    calculate_sector_relative_valuation,
    calculate_growth_cagr
)
from .market import MarketFeatures, merge_market_features
from .edinet_features import EdinetFeatures, calculate_rd_sector_relative
from .disclosure_features import (
    DisclosureFeatures,
    calculate_disclosure_momentum,
    aggregate_sector_disclosures
)
from .global_index_features import (
    GlobalIndexFeatures,
    merge_global_features,
    calculate_market_regime
)
from .trends_features import (
    TrendsFeatures,
    merge_trends_features,
    calculate_sector_trends,
    calculate_relative_search_interest
)
from .builder import (
    FeatureBuilder,
    get_feature_columns,
    clean_features,
    split_by_date
)

__all__ = [
    # Technical
    'TechnicalFeatures',
    'calculate_sector_rank',
    # Fundamental (J-Quants)
    'FundamentalFeatures',
    'calculate_sector_relative_valuation',
    'calculate_growth_cagr',
    # Market & Supply-Demand
    'MarketFeatures',
    'merge_market_features',
    # EDINET Detailed Financials
    'EdinetFeatures',
    'calculate_rd_sector_relative',
    # TDnet Disclosures
    'DisclosureFeatures',
    'calculate_disclosure_momentum',
    'aggregate_sector_disclosures',
    # Global Indices
    'GlobalIndexFeatures',
    'merge_global_features',
    'calculate_market_regime',
    # Google Trends
    'TrendsFeatures',
    'merge_trends_features',
    'calculate_sector_trends',
    'calculate_relative_search_interest',
    # Builder
    'FeatureBuilder',
    'get_feature_columns',
    'clean_features',
    'split_by_date',
]
