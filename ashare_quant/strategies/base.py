"""策略基类与通用工具。

设计约定（重要）
----------------
策略只负责输出**目标权重矩阵** ``target_weights``：

* 行 = 交易日（信号使用当日收盘数据计算，引擎在**次日开盘**执行，杜绝未来函数）
* 列 = 股票代码
* 值 = 该股票占组合总资产的目标权重（0~1 之间的多头权重，A 股不支持裸卖空）

下单、整手取整、涨跌停、停牌、T+1、手续费、滑点全部交给 ``backtest`` 引擎，
策略本身保持"信号纯度"，便于复用与横向比较。
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Type

import numpy as np
import pandas as pd

from ..config import TRADE_DAYS_PER_YEAR
from ..data import DataBundle

_FREQ_MAP = {"W": "W", "M": "M", "Q": "Q", "Y": "Y"}


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
def rebalance_mask(index: pd.DatetimeIndex, freq: str | int = "M") -> pd.Series:
    """生成调仓日掩码（每个周期最后一个交易日为 True）。"""
    idx = pd.DatetimeIndex(index)
    if isinstance(freq, int) or str(freq).isdigit():
        n = max(1, int(freq))
        arr = np.zeros(len(idx), dtype=bool)
        arr[n - 1::n] = True
        return pd.Series(arr, index=idx)
    f = _FREQ_MAP.get(str(freq).upper(), "M")
    periods = idx.to_period(f)
    last = pd.Series(idx, index=periods).groupby(level=0).last()
    return pd.Series(idx.isin(set(last.values)), index=idx)


def cross_rank(df: pd.DataFrame, ascending: bool = False) -> pd.DataFrame:
    """横截面百分位排名（0~1），用于多因子打分与排序。"""
    return df.rank(axis=1, ascending=ascending, pct=True)


def cross_zscore(df: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    """横截面 Z-Score 标准化（对极端值做截断）。"""
    m = df.mean(axis=1)
    sd = df.std(axis=1, ddof=0).replace(0.0, np.nan)
    z = df.sub(m, axis=0).div(sd, axis=0)
    return z.clip(-clip, clip) if clip else z


def top_k_weights(score: pd.DataFrame, k: int = 5, weight_mode: str = "equal",
                  max_weight: float = 0.30, min_score: Optional[float] = None,
                  total_exposure: float = 1.0) -> pd.DataFrame:
    """按打分选前 K 名，输出目标权重矩阵。

    Parameters
    ----------
    score : DataFrame
        日期 × 股票的因子/信号打分，越大越优先；NaN 表示不可选。
    k : int
        持仓数量上限。
    weight_mode : {'equal', 'score'}
        等权，或按打分（截断到正值）加权。
    max_weight : float
        单一标的最大权重。
    min_score : float, optional
        低于该阈值不入选。
    total_exposure : float
        总仓位上限（1.0 = 满仓，0.5 = 半仓）。
    """
    if score.empty:
        return score.copy()
    s = score.copy()
    if min_score is not None:
        s = s.where(s >= min_score)
    k = max(1, int(k))
    ranks = s.rank(axis=1, ascending=False, method="first")
    picked = ranks.le(k) & s.notna()

    if weight_mode == "score":
        raw = s.where(picked).clip(lower=0.0)
        raw = raw.where(raw.notna(), np.where(picked, 1.0, np.nan))
        row_sum = raw.sum(axis=1).replace(0.0, np.nan)
        w = raw.div(row_sum, axis=0)
    else:
        w = picked.astype(float)
        n = w.sum(axis=1).replace(0.0, np.nan)
        w = w.div(n, axis=0)
    w = w.clip(upper=max_weight)
    return normalize_weights(w, total_exposure=total_exposure)


def normalize_weights(w: pd.DataFrame, total_exposure: float = 1.0) -> pd.DataFrame:
    """权重归一化：确保每行之和不超过 ``total_exposure``，并清理 NaN/负值。"""
    out = w.copy().fillna(0.0)
    out[out < 0] = 0.0
    s = out.sum(axis=1)
    scale = pd.Series(1.0, index=out.index)
    over = s > total_exposure
    scale[over] = total_exposure / s[over].replace(0.0, np.nan)
    return out.mul(scale, axis=0).fillna(0.0)


def stateful_signal(entry: pd.DataFrame, exit_: pd.DataFrame, init: float = 0.0) -> pd.DataFrame:
    """由"入场/出场"两个布尔矩阵生成持仓状态矩阵（持有到出场信号出现为止）。

    逐列逐行推进，属于事件型策略的标准写法；与纯向量化相比不易引入未来函数。
    """
    out = pd.DataFrame(init, index=entry.index, columns=entry.columns, dtype=float)
    ent = entry.fillna(False).to_numpy()
    ext = exit_.reindex_like(entry).fillna(False).to_numpy()
    res = out.to_numpy(copy=True)
    for j in range(ent.shape[1]):
        state = init
        for i in range(ent.shape[0]):
            if state == 0.0 and ent[i, j]:
                state = 1.0
            elif state == 1.0 and ext[i, j]:
                state = 0.0
            res[i, j] = state
    return pd.DataFrame(res, index=entry.index, columns=entry.columns)


def apply_market_timing(weights: pd.DataFrame, timing: pd.Series,
                        mode: str = "scale") -> pd.DataFrame:
    """用市场择时信号处理权重。

    ``mode='scale'`` 按择时强度缩放仓位；``mode='switch'`` 择时为 0 时清仓。
    """
    t = timing.reindex(weights.index).ffill().fillna(0.0).clip(0.0, 1.0)
    if mode == "switch":
        t = (t > 0.5).astype(float)
    return weights.mul(t, axis=0)


def inverse_vol_weights(base: pd.DataFrame, returns: pd.DataFrame, lookback: int = 20,
                        target_vol: Optional[float] = None,
                        max_leverage: float = 1.0) -> pd.DataFrame:
    """风险平价简化版：按目标标的的滚动波动率倒数分配权重，可选波动率目标化。"""
    vol = returns.rolling(int(lookback), min_periods=max(3, int(lookback) // 2)).std(ddof=0)
    iv = 1.0 / vol.replace(0.0, np.nan)
    w = base.mul(iv.reindex_like(base))
    w = normalize_weights(w, total_exposure=1.0)
    if target_vol:
        port_vol = (w.shift(1) * returns).sum(axis=1, min_count=1) \
            .rolling(int(lookback), min_periods=5).std(ddof=0) * np.sqrt(TRADE_DAYS_PER_YEAR)
        scale = (float(target_vol) / port_vol.replace(0.0, np.nan)).clip(upper=max_leverage).fillna(1.0)
        w = w.mul(scale, axis=0)
    return normalize_weights(w, total_exposure=max_leverage)


# --------------------------------------------------------------------------- #
# 策略注册表
# --------------------------------------------------------------------------- #
@dataclass
class Strategy(abc.ABC):
    """策略抽象基类。子类需定义 ``key/display_name/category/default_params/param_space``。"""

    key: str = "base"
    display_name: str = "基础策略"
    category: str = "通用"
    description: str = ""
    default_params: Dict[str, Any] = field(default_factory=dict)
    param_space: Dict[str, Sequence[Any]] = field(default_factory=dict)

    def __init__(self, **params: Any) -> None:
        merged = dict(self.default_params)
        merged.update({k: v for k, v in params.items() if v is not None})
        self.params = merged

    # ---- 参数访问 ----
    def p(self, name: str, default: Any = None) -> Any:
        if name in self.params:
            return self.params[name]
        if name in self.default_params:
            return self.default_params[name]
        return default

    @property
    def label(self) -> str:
        return f"{self.display_name}({self.param_text})"

    @property
    def param_text(self) -> str:
        items = [f"{k}={v}" for k, v in self.params.items()]
        return ",".join(items) if items else "默认"

    # ---- 核心接口 ----
    @abc.abstractmethod
    def generate(self, data: DataBundle) -> pd.DataFrame:
        """返回目标权重矩阵（索引=交易日，列=股票代码）。"""

    def weight_matrix(self, data: DataBundle) -> pd.DataFrame:
        w = self.generate(data)
        w = w.reindex(index=data.calendar, columns=data.symbols).fillna(0.0)
        return normalize_weights(w)

    @classmethod
    def space(cls) -> Dict[str, Sequence[Any]]:
        return dict(cls.param_space)

    def info(self) -> Dict[str, Any]:
        return {"key": self.key, "name": self.display_name, "category": self.category,
                "params": dict(self.params), "description": self.description}


REGISTRY: Dict[str, Type[Strategy]] = {}


def register(cls: Type[Strategy]) -> Type[Strategy]:
    """策略注册装饰器。"""
    REGISTRY[cls.key] = cls
    return cls


def get_strategy(key: str) -> Type[Strategy]:
    if key not in REGISTRY:
        raise KeyError(f"未注册的策略：{key}；可用：{sorted(REGISTRY)}")
    return REGISTRY[key]


def build_strategy(key: str, **params: Any) -> Strategy:
    return get_strategy(key)(**params)


def list_strategies() -> pd.DataFrame:
    rows = [{"key": c.key, "name": c.display_name, "category": c.category,
             "description": c.description, "默认参数": c.default_params,
             "可优化参数": list(c.param_space.keys())}
            for c in REGISTRY.values()]
    return pd.DataFrame(rows)