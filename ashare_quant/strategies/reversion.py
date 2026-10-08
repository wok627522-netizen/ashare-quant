"""均值回归 / 震荡类策略：RSI 超卖、布林带回归、网格交易。

A 股注意点：个股 T+1，日内"低买高卖"不可行；本模块的信号都是**隔日**级别的
超跌反弹与区间震荡逻辑，出场后资金可再次使用。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import indicators as ind
from ..data import DataBundle
from .base import (Strategy, normalize_weights, rebalance_mask, register,
                   stateful_signal, top_k_weights)
from .trend import hold_between_rebalance

__all__ = ["RSIReversionStrategy", "BollingerReversionStrategy", "GridTradingStrategy"]


@register
class RSIReversionStrategy(Strategy):
    key = "rsi_reversion"
    display_name = "RSI 超跌反弹"
    category = "均值回归"
    description = ("RSI 低于超卖阈值买入，回到中性区离场；"
                   "叠加长期均线过滤，避免在下跌趋势中不断抄底。")
    default_params = {"rsi_n": 14, "oversold": 30, "exit_level": 55, "trend_ma": 120,
                      "top_k": 5, "max_weight": 0.25, "rebalance": 1}
    param_space = {"rsi_n": [6, 14, 21], "oversold": [20, 25, 30, 35],
                   "exit_level": [50, 55, 60, 70], "trend_ma": [0, 60, 120, 250],
                   "top_k": [3, 5, 8]}

    def generate(self, data: DataBundle) -> pd.DataFrame:
        close = data.close_matrix()
        n = int(self.p("rsi_n"))
        r = ind.rsi(close, n)
        filter_ok = pd.DataFrame(True, index=close.index, columns=close.columns)
        trend_ma = int(self.p("trend_ma") or 0)
        if trend_ma > 0:
            filter_ok = close > ind.sma(close, trend_ma)
        entry = (r < float(self.p("oversold"))) & filter_ok
        exit_ = r > float(self.p("exit_level"))
        holds = stateful_signal(entry, exit_)
        score = (100.0 - r).where(holds > 0)     # RSI 越低分越高
        mask = rebalance_mask(close.index, self.p("rebalance"))
        w = top_k_weights(score, k=int(self.p("top_k")), max_weight=float(self.p("max_weight")))
        return hold_between_rebalance(w, mask)


@register
class BollingerReversionStrategy(Strategy):
    key = "boll_reversion"
    display_name = "布林带回归"
    category = "均值回归"
    description = ("价格跌破布林下轨买入，回到中轨上方离场；"
                   "带宽过滤掉极端波动区间，适合震荡市。")
    default_params = {"n": 20, "k": 2.0, "entry_pctb": 0.05, "exit_pctb": 0.55,
                      "max_width": 0.0, "top_k": 5, "max_weight": 0.25, "rebalance": 1}
    param_space = {"n": [10, 20, 30], "k": [1.5, 2.0, 2.5, 3.0],
                   "entry_pctb": [0.0, 0.05, 0.15], "exit_pctb": [0.5, 0.6, 0.8],
                   "top_k": [3, 5, 8]}

    def generate(self, data: DataBundle) -> pd.DataFrame:
        close = data.close_matrix()
        mid, up, dn, pctb, width = ind.boll(close, int(self.p("n")), float(self.p("k")))
        entry = pctb < float(self.p("entry_pctb"))
        max_width = float(self.p("max_width") or 0)
        if max_width > 0:
            entry = entry & (width < max_width)
        exit_ = pctb > float(self.p("exit_pctb"))
        holds = stateful_signal(entry, exit_)
        score = (-pctb).where(holds > 0)
        mask = rebalance_mask(close.index, self.p("rebalance"))
        w = top_k_weights(score, k=int(self.p("top_k")), max_weight=float(self.p("max_weight")))
        return hold_between_rebalance(w, mask)


@register
class GridTradingStrategy(Strategy):
    key = "grid"
    display_name = "网格交易"
    category = "震荡"
    description = ("以均线为基准价，价格每下跌一格加仓、回升一格减仓；"
                   "叠加长期趋势过滤，只在多头趋势中做网格。")
    default_params = {"ref_window": 20, "grid_step": 0.03, "max_grids": 4,
                      "per_grid": 0.15, "trend_ma": 120, "rebalance": 1,
                      "max_weight": 0.35}
    param_space = {"ref_window": [10, 20, 60], "grid_step": [0.02, 0.03, 0.05, 0.08],
                   "max_grids": [2, 3, 4, 6], "per_grid": [0.1, 0.15, 0.25],
                   "trend_ma": [0, 60, 120]}

    def generate(self, data: DataBundle) -> pd.DataFrame:
        close = data.close_matrix()
        ref = ind.sma(close, int(self.p("ref_window")))
        step = float(self.p("grid_step"))
        grids = np.floor(((ref - close) / ref.replace(0.0, np.nan)) / step)
        grids = grids.clip(lower=0, upper=int(self.p("max_grids")))
        w = grids * float(self.p("per_grid"))
        trend_ma = int(self.p("trend_ma") or 0)
        if trend_ma > 0:
            w = w.where(close > ind.sma(close, trend_ma), 0.0)
        w = w.clip(upper=float(self.p("max_weight")))
        mask = rebalance_mask(close.index, self.p("rebalance"))
        return hold_between_rebalance(w, mask).fillna(0.0)