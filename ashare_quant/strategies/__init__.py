"""策略包：导入即完成全部策略注册。"""

from .base import (REGISTRY, Strategy, build_strategy, cross_rank, cross_zscore,
                   get_strategy, inverse_vol_weights, list_strategies,
                   normalize_weights, rebalance_mask, register, stateful_signal,
                   top_k_weights)
from . import reversion, rotation, trend  # noqa: F401  触发注册
from .reversion import BollingerReversionStrategy, GridTradingStrategy, RSIReversionStrategy
from .rotation import (EnsembleStrategy, IndustryRotationStrategy,
                       LowVolatilityStrategy, MomentumRotationStrategy,
                       MultiFactorStrategy)
from .trend import (DualMAStrategy, IndexTimingStrategy, MACDStrategy,
                    TurtleBreakoutStrategy)

__all__ = [
    "REGISTRY", "Strategy", "register", "get_strategy", "build_strategy", "list_strategies",
    "normalize_weights", "top_k_weights", "cross_rank", "cross_zscore", "rebalance_mask",
    "stateful_signal", "inverse_vol_weights",
    "DualMAStrategy", "MACDStrategy", "TurtleBreakoutStrategy", "IndexTimingStrategy",
    "RSIReversionStrategy", "BollingerReversionStrategy", "GridTradingStrategy",
    "MomentumRotationStrategy", "IndustryRotationStrategy", "MultiFactorStrategy",
    "LowVolatilityStrategy", "EnsembleStrategy",
]