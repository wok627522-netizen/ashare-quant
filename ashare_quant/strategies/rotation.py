"""轮动与多因子类策略：动量轮动、行业轮动、多因子打分、低波动、组合集成。"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from .. import indicators as ind
from ..data import DataBundle
from .base import (Strategy, apply_market_timing, build_strategy, cross_zscore,
                   normalize_weights, rebalance_mask, register, top_k_weights)
from .trend import hold_between_rebalance

__all__ = ["MomentumRotationStrategy", "IndustryRotationStrategy",
           "MultiFactorStrategy", "LowVolatilityStrategy", "EnsembleStrategy"]


def _market_timing(data: DataBundle, ma_len: int) -> pd.Series:
    """基准指数均线择时：指数在均线上方才满仓，否则空仓。"""
    ref = data.benchmark if data.benchmark is not None else data.close_matrix().mean(axis=1)
    ref = pd.Series(ref).dropna()
    ma = ind.sma(ref, int(ma_len))
    return ((ref > ma) & (ind.sma(ref, max(5, int(ma_len) // 4)) > ma)).astype(float)


@register
class MomentumRotationStrategy(Strategy):
    key = "momentum_rotation"
    display_name = "动量轮动"
    category = "轮动"
    description = ("经典截面动量：按 N 日涨幅（跳过最近几日以规避短期反转）排名，"
                   "定期调仓持有最强的 K 只，并以大盘均线控制总仓位。")
    default_params = {"lookback": 60, "skip": 5, "top_k": 3, "rebalance": "M",
                      "trend_ma": 60, "use_timing": 1, "bench_ma": 60,
                      "max_weight": 0.50, "weight_mode": "equal"}
    param_space = {"lookback": [20, 40, 60, 120], "skip": [0, 5, 10],
                   "top_k": [2, 3, 5], "rebalance": ["W", "M", 10, 20],
                   "trend_ma": [0, 20, 60, 120], "use_timing": [0, 1],
                   "bench_ma": [20, 60, 120]}

    def generate(self, data: DataBundle) -> pd.DataFrame:
        close = data.close_matrix()
        lb, skip = int(self.p("lookback")), int(self.p("skip"))
        score = close.shift(skip) / close.shift(skip + lb) - 1.0
        trend_ma = int(self.p("trend_ma") or 0)
        if trend_ma > 0:
            score = score.where(close > ind.sma(close, trend_ma))
        w = top_k_weights(score, k=int(self.p("top_k")), max_weight=float(self.p("max_weight")),
                          weight_mode=str(self.p("weight_mode")))
        mask = rebalance_mask(close.index, self.p("rebalance"))
        w = hold_between_rebalance(w, mask)
        if int(self.p("use_timing") or 0):
            w = apply_market_timing(w, _market_timing(data, int(self.p("bench_ma"))), mode="switch")
        return w


@register
class IndustryRotationStrategy(Strategy):
    key = "industry_rotation"
    display_name = "行业轮动"
    category = "轮动"
    description = ("先在行业维度做动量排名（行业内个股动量均值），"
                   "买入最强行业的龙头个股，实现「行业 Beta + 个股 Alpha」的双层筛选。")
    default_params = {"lookback": 40, "n_industry": 3, "per_industry": 2,
                      "rebalance": "M", "trend_ma": 60, "max_weight": 0.30,
                      "use_timing": 1, "bench_ma": 60}
    param_space = {"lookback": [20, 40, 60, 120], "n_industry": [1, 2, 3, 4],
                   "per_industry": [1, 2, 3], "rebalance": ["W", "M", 20],
                   "trend_ma": [0, 60, 120], "use_timing": [0, 1]}

    def generate(self, data: DataBundle) -> pd.DataFrame:
        close = data.close_matrix()
        lb = int(self.p("lookback"))
        raw = close / close.shift(lb) - 1.0
        trend_ma = int(self.p("trend_ma") or 0)
        if trend_ma > 0:
            raw = raw.where(close > ind.sma(close, trend_ma))
        industries = {s: data.industry_of(s) for s in close.columns}
        ind_df = pd.DataFrame({s: industries[s] for s in close.columns}, index=["industry"]).T
        score = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        ind_score = pd.DataFrame(index=close.index, columns=sorted(set(industries.values())),
                                 dtype=float)
        for name in ind_score.columns:
            cols = [s for s in close.columns if industries[s] == name]
            ind_score[name] = raw[cols].mean(axis=1)
        top_ind = ind_score.rank(axis=1, ascending=False) <= int(self.p("n_industry"))
        for name in ind_score.columns:
            cols = [s for s in close.columns if industries[s] == name]
            sel = raw[cols].where(top_ind[name], np.nan)
            keep = sel.rank(axis=1, ascending=False) <= int(self.p("per_industry"))
            score[cols] = sel.where(keep)
        w = top_k_weights(score, k=int(self.p("n_industry")) * int(self.p("per_industry")),
                          max_weight=float(self.p("max_weight")))
        mask = rebalance_mask(close.index, self.p("rebalance"))
        w = hold_between_rebalance(w, mask)
        if int(self.p("use_timing") or 0):
            w = apply_market_timing(w, _market_timing(data, int(self.p("bench_ma"))), mode="switch")
        return w


@register
class MultiFactorStrategy(Strategy):
    key = "multi_factor"
    display_name = "多因子选股"
    category = "多因子"
    description = ("动量、短期反转、低波动、趋势强度、流动性五个维度横截面标准化后加权打分，"
                   "买入综合分最高的 K 只，是目前公募/私募最常用的框架之一。")
    default_params = {
        "w_momentum": 0.30, "w_reversal": 0.10, "w_lowvol": 0.20,
        "w_trend": 0.25, "w_liquidity": 0.15, "mom_window": 60,
        "rev_window": 5, "vol_window": 20, "top_k": 5, "rebalance": "M",
        "max_weight": 0.25, "min_amount": 0.0, "use_timing": 0,
        "bench_ma": 60, "weight_mode": "equal",
    }
    param_space = {"w_momentum": [0.2, 0.3, 0.4], "w_lowvol": [0.0, 0.2, 0.3],
                   "w_trend": [0.1, 0.25, 0.4], "mom_window": [20, 40, 60, 120],
                   "top_k": [3, 5, 8, 10], "rebalance": ["M", 10, 20],
                   "use_timing": [0, 1]}

    def generate(self, data: DataBundle) -> pd.DataFrame:
        close, amount = data.close_matrix(), data.amount_matrix()
        ret = close.pct_change()
        f_mom = ind.momentum(close, int(self.p("mom_window")))
        f_rev = -ind.momentum(close, int(self.p("rev_window")))
        f_vol = -ind.volatility(close, int(self.p("vol_window")))
        f_trend = ind.sma(close, 20) / ind.sma(close, 60) - 1.0
        liq = np.log1p(amount.rolling(20, min_periods=5).mean())
        f_liq = -liq   # 流动性因子：中小市值/低成交额往往有溢价，故取负号

        z = lambda f: cross_zscore(f.replace([np.inf, -np.inf], np.nan))  # noqa: E731
        score = (float(self.p("w_momentum")) * z(f_mom)
                 + float(self.p("w_reversal")) * z(f_rev)
                 + float(self.p("w_lowvol")) * z(f_vol)
                 + float(self.p("w_trend")) * z(f_trend)
                 + float(self.p("w_liquidity")) * z(f_liq))

        min_amount = float(self.p("min_amount") or 0)
        if min_amount > 0:
            score = score.where(amount.rolling(20, min_periods=5).mean() >= min_amount)
        # 剔除 ST 与停牌标的
        st_cols = [s for s in score.columns if data.is_st(s)]
        if st_cols:
            score[st_cols] = np.nan
        score = score.where(~data.suspended_matrix())

        w = top_k_weights(score, k=int(self.p("top_k")), max_weight=float(self.p("max_weight")),
                          weight_mode=str(self.p("weight_mode")))
        mask = rebalance_mask(close.index, self.p("rebalance"))
        w = hold_between_rebalance(w, mask)
        if int(self.p("use_timing") or 0):
            w = apply_market_timing(w, _market_timing(data, int(self.p("bench_ma"))), mode="switch")
        return w


@register
class LowVolatilityStrategy(Strategy):
    key = "low_vol"
    display_name = "低波动防御"
    category = "多因子"
    description = ("买入过去一段时间波动率最低、回撤最小的股票，"
                   "熊市与震荡市中防御性突出，A 股「低波动异象」长期有效。")
    default_params = {"vol_window": 60, "top_k": 6, "rebalance": "M",
                      "trend_ma": 120, "max_weight": 0.25, "use_timing": 1,
                      "bench_ma": 60}
    param_space = {"vol_window": [20, 60, 120], "top_k": [3, 5, 8, 10],
                   "rebalance": ["M", 20], "trend_ma": [0, 120, 250], "use_timing": [0, 1]}

    def generate(self, data: DataBundle) -> pd.DataFrame:
        close = data.close_matrix()
        ret = close.pct_change()
        vol = ind.volatility(close, int(self.p("vol_window")))
        # 最大回撤（负的，越大越好）
        roll_max = close.rolling(int(self.p("vol_window")), min_periods=5).max()
        dd = close / roll_max - 1.0
        score = cross_zscore(-vol) + 0.5 * cross_zscore(dd)
        trend_ma = int(self.p("trend_ma") or 0)
        if trend_ma > 0:
            score = score.where(close > ind.sma(close, trend_ma))
        score = score.where(~data.suspended_matrix())
        w = top_k_weights(score, k=int(self.p("top_k")), max_weight=float(self.p("max_weight")))
        mask = rebalance_mask(close.index, self.p("rebalance"))
        w = hold_between_rebalance(w, mask)
        if int(self.p("use_timing") or 0):
            w = apply_market_timing(w, _market_timing(data, int(self.p("bench_ma"))), mode="switch")
        return w


@register
class EnsembleStrategy(Strategy):
    key = "ensemble"
    display_name = "多策略集成"
    category = "组合"
    description = ("把多个子策略的权重等权（或按历史波动率倒数加权）合成，"
                   "通过低相关性叠加提升夏普、降低单策略失效风险。")
    default_params = {"members": ["dual_ma", "momentum_rotation", "low_vol"],
                      "mode": "equal", "max_weight": 0.30}
    param_space = {"members": [
        ["dual_ma", "momentum_rotation", "low_vol"],
        ["dual_ma", "macd", "rsi_reversion"],
        ["momentum_rotation", "multi_factor", "low_vol"],
        ["multi_factor", "grid", "turtle"],
    ], "mode": ["equal", "inv_vol"]}

    def generate(self, data: DataBundle) -> pd.DataFrame:
        members: List[str] = list(self.p("members") or [])
        mats = []
        for key in members:
            try:
                mats.append(build_strategy(key).weight_matrix(data))
            except Exception:
                continue
        if not mats:
            return pd.DataFrame(0.0, index=data.calendar, columns=data.symbols)
        out = pd.DataFrame(0.0, index=data.calendar, columns=data.symbols)
        mode = str(self.p("mode"))
        if mode == "inv_vol":
            ret = data.close_matrix().pct_change()
            weights = []
            for key, m in zip(members, mats):
                r = (m.shift(1) * ret).sum(axis=1)
                vol = r.rolling(60, min_periods=10).std(ddof=0)
                weights.append(1.0 / vol.replace(0.0, np.nan))
            wdf = pd.concat(weights, axis=1).replace([np.inf], np.nan).fillna(0.0)
            wdf = wdf.div(wdf.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(1.0 / len(mats))
            for i, m in enumerate(mats):
                out = out.add(m.mul(wdf.iloc[:, i], axis=0), fill_value=0.0)
        else:
            for m in mats:
                out = out.add(m / len(mats), fill_value=0.0)
        out = out.clip(upper=float(self.p("max_weight")))
        return normalize_weights(out)