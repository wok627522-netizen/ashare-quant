"""趋势类策略：双均线、MACD、唐奇安突破、指数择时。"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd

from .. import indicators as ind
from ..data import DataBundle
from .base import (Strategy, apply_market_timing, normalize_weights,
                   rebalance_mask, stateful_signal, top_k_weights, register)

__all__ = ["DualMAStrategy", "MACDStrategy", "TurtleBreakoutStrategy",
           "IndexTimingStrategy", "hold_between_rebalance"]


def hold_between_rebalance(w: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """调仓日之间的权重保持不变（用 ffill 实现，避免天天调仓带来的高换手）。"""
    if w.empty:
        return w
    m = mask.reindex(w.index).fillna(False).astype(bool)
    tmp = w.where(m, np.nan)
    return tmp.ffill().fillna(0.0)


@register
class DualMAStrategy(Strategy):
    key = "dual_ma"
    display_name = "双均线趋势"
    category = "趋势"
    description = ("经典趋势跟随：快线上穿慢线且价格站上趋势均线时建仓，"
                   "按趋势强度排名选前 K 只等权，周期内持有不动。")
    default_params = {"fast": 5, "slow": 20, "trend_ma": 60, "top_k": 5,
                      "rebalance": 5, "max_weight": 0.25, "vol_filter": 0.0,
                      "momentum_confirm": 0}
    param_space = {"fast": [3, 5, 8, 10], "slow": [15, 20, 30, 40, 60],
                   "trend_ma": [0, 60, 120], "top_k": [3, 5, 8, 10],
                   "rebalance": [5, 10, 20]}

    def generate(self, data: DataBundle) -> pd.DataFrame:
        close = data.close_matrix()
        fast, slow = int(self.p("fast")), int(self.p("slow"))
        trend = int(self.p("trend_ma") or 0)
        ma_f, ma_s = ind.sma(close, fast), ind.sma(close, slow)
        signal = (ma_f > ma_s) & (close > ma_s)
        if trend > 0:
            signal = signal & (close > ind.sma(close, trend))
        mom_confirm = int(self.p("momentum_confirm") or 0)
        if mom_confirm > 0:
            signal = signal & (ind.momentum(close, mom_confirm) > 0)
        vol_filter = float(self.p("vol_filter") or 0)
        if vol_filter > 0:
            signal = signal & (ind.volatility(close, 20) < vol_filter)

        score = (ma_f / ma_s - 1.0).replace([np.inf, -np.inf], np.nan) \
            + 0.5 * ind.roc(close, max(fast * 4, 20)) / 100.0
        score = score.where(signal)

        mask = rebalance_mask(close.index, self.p("rebalance"))
        w = top_k_weights(score, k=int(self.p("top_k")), max_weight=float(self.p("max_weight")),
                          total_exposure=1.0)
        return hold_between_rebalance(w, mask)


@register
class MACDStrategy(Strategy):
    key = "macd"
    display_name = "MACD 动量"
    category = "趋势"
    description = ("MACD 金叉且柱状线翻红、价格位于中期均线上方时持有，"
                   "按柱线强度选股；适合中期波段。")
    default_params = {"fast": 12, "slow": 26, "signal": 9, "trend_ma": 20,
                      "top_k": 5, "rebalance": 3, "max_weight": 0.25}
    param_space = {"fast": [8, 12, 16], "slow": [20, 26, 34], "signal": [7, 9, 12],
                   "trend_ma": [0, 20, 60], "top_k": [3, 5, 8]}

    def generate(self, data: DataBundle) -> pd.DataFrame:
        close = data.close_matrix()
        dif, dea, hist = ind.macd(close, int(self.p("fast")), int(self.p("slow")), int(self.p("signal")))
        trend = int(self.p("trend_ma") or 0)
        signal = (dif > dea) & (hist > 0)
        if trend > 0:
            signal = signal & (close > ind.sma(close, trend))
        score = (hist / close) + (dif - dea) / close
        score = score.where(signal)
        mask = rebalance_mask(close.index, self.p("rebalance"))
        w = top_k_weights(score, k=int(self.p("top_k")), max_weight=float(self.p("max_weight")))
        return hold_between_rebalance(w, mask)


@register
class TurtleBreakoutStrategy(Strategy):
    key = "turtle"
    display_name = "唐奇安突破(海龟)"
    category = "趋势"
    description = ("海龟交易法简化版：价格突破 N 日最高价买入，跌破 M 日最低价卖出，"
                   "持有到出场信号为止；止损由风控模块的 ATR 跟踪止损负责。")
    default_params = {"entry": 20, "exit": 10, "top_k": 5, "max_weight": 0.25,
                      "rebalance": 2}
    param_space = {"entry": [10, 20, 30, 55], "exit": [5, 10, 20], "top_k": [3, 5, 8]}

    def generate(self, data: DataBundle) -> pd.DataFrame:
        high, low, close = data.high_matrix(), data.low_matrix(), data.close_matrix()
        up, dn, _ = ind.donchian(high, low, int(self.p("entry")), include_today=False)
        _, exit_dn, _ = ind.donchian(high, low, int(self.p("exit")), include_today=False)
        entry = close > up
        exit_ = close < exit_dn
        holds = stateful_signal(entry, exit_)
        strength = (close / up - 1.0).replace([np.inf, -np.inf], np.nan)
        score = strength.where(holds > 0)
        mask = rebalance_mask(close.index, self.p("rebalance"))
        w = top_k_weights(score, k=int(self.p("top_k")), max_weight=float(self.p("max_weight")))
        return hold_between_rebalance(w, mask)


@register
class IndexTimingStrategy(Strategy):
    key = "index_timing"
    display_name = "指数择时(MA)"
    category = "择时"
    description = ("以基准指数（或单只 ETF）均线多头排列作为总仓位开关："
                   "快线在慢线上方满仓，否则空仓；常用于 ETF 轮动与大盘择时叠加。")
    default_params = {"fast": 20, "slow": 60, "rebalance": 1, "mode": "switch",
                      "per_symbol": 0.5}
    param_space = {"fast": [5, 10, 20, 30], "slow": [30, 60, 120, 200],
                   "mode": ["switch", "scale"]}

    def generate(self, data: DataBundle) -> pd.DataFrame:
        close = data.close_matrix()
        if data.benchmark is not None and len(data.benchmark.dropna()) > 30:
            ref = data.benchmark.dropna()
        else:
            ref = close.mean(axis=1)
        fast, slow = int(self.p("fast")), int(self.p("slow"))
        ma_f, ma_s = ind.sma(ref, fast), ind.sma(ref, slow)
        timing = ((ma_f > ma_s) & (ref > ma_s)).astype(float)
        # 温和过渡：快慢线距离越大，仓位越接近满仓
        strength = ((ma_f / ma_s - 1.0) / 0.05).clip(0.0, 1.0).fillna(0.0)
        timing = pd.concat([timing, strength], axis=1).max(axis=1)

        tradable = close.notna()
        base = tradable.astype(float)
        n = base.sum(axis=1).replace(0.0, np.nan)
        base = base.div(n, axis=0).fillna(0.0)
        per_symbol = float(self.p("per_symbol") or 0)
        if per_symbol > 0:
            base = base.where(base <= 0, per_symbol)
        return apply_market_timing(base, timing, mode=str(self.p("mode")))