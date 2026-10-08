"""因子研究与选股：因子面板、IC 分析、分层回测、综合打分。

因子体系（A 股常用）
--------------------
* 动量：20/60/120 日收益（跳过近 5 日）
* 反转：5 日收益取负（A 股短期反转效应显著）
* 波动：20/60 日波动率取负（低波动异象）
* 换手：换手率/成交额（流动性溢价与情绪）
* 趋势：均线多头排列强度、价格相对 52 周高点位置
* 质量代理：波动调整后收益（夏普代理）、回撤修复能力
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from . import indicators as ind
from .data import DataBundle
from .strategies.base import cross_zscore, top_k_weights

__all__ = ["FACTOR_LABELS", "factor_panel", "composite_score", "forward_returns",
           "factor_ic", "ic_summary", "quantile_returns", "score_to_weights",
           "factor_table"]

FACTOR_LABELS: Dict[str, str] = {
    "mom20": "20日动量", "mom60": "60日动量", "mom120": "120日动量",
    "rev5": "5日反转", "vol20": "20日波动(低)", "vol60": "60日波动(低)",
    "trend": "趋势强度", "amount20": "流动性(小)", "high52": "距52周高点",
    "sharpe60": "风险调整收益", "maxdd60": "60日回撤(浅)",
    # ---- 因子研究模块新增因子（IC 分析页签用）----
    "atr_pct": "低波幅", "ma_slope20": "均线斜率", "liquidity": "流动性(小市值)",
    "illiq": "非流动性", "updays20": "上涨天数占比", "skew20": "收益偏度",
    "rsi_rev14": "RSI反向", "boll_pos": "布林低位", "turnover20": "换手活跃度",
    "boll_break": "跌破布林下轨", "rebound_combo": "超跌反弹组合(严格)", "rebound_soft": "超跌反弹组合(宽松)",
}


def _default_bundle():
    return {}


def factor_panel(data: DataBundle, factors: Optional[Sequence[str]] = None,
                 forward_days: int = 1) -> Dict[str, pd.DataFrame]:
    """计算全部（或指定）因子的横截面面板：``{因子名: 日期 × 股票}``。

    所有因子都经过"越大越好"的方向调整，方便直接加权合成。
    """
    close = data.close_matrix()
    amount = data.amount_matrix()
    volume = data.volume_matrix()
    high, low = data.high_matrix(), data.low_matrix()
    ret = close.pct_change()

    panel: Dict[str, pd.DataFrame] = {}
    panel["mom20"] = close.shift(5) / close.shift(25) - 1.0
    panel["mom60"] = close.shift(5) / close.shift(65) - 1.0
    panel["mom120"] = close.shift(5) / close.shift(125) - 1.0
    panel["rev5"] = -(close / close.shift(5) - 1.0)
    panel["vol20"] = -ind.volatility(close, 20)
    panel["vol60"] = -ind.volatility(close, 60)
    panel["trend"] = ind.sma(close, 20) / ind.sma(close, 60) - 1.0
    panel["amount20"] = -np.log1p(amount.rolling(20, min_periods=5).mean())
    panel["high52"] = close / close.rolling(250, min_periods=60).max()
    mean20 = ret.rolling(20, min_periods=10).mean()
    sd20 = ret.rolling(20, min_periods=10).std(ddof=0)
    panel["sharpe60"] = (mean20 / sd20.replace(0.0, np.nan)).rolling(3, min_periods=1).mean()
    dd = close / close.rolling(60, min_periods=20).max() - 1.0
    panel["maxdd60"] = dd  # 回撤越浅（越接近 0）越好
    for k in list(panel):
        panel[k] = panel[k].replace([np.inf, -np.inf], np.nan)
    if factors:
        panel = {k: v for k, v in panel.items() if k in factors}
    return panel


def composite_score(panel: Dict[str, pd.DataFrame],
                    weights: Optional[Dict[str, float]] = None,
                    standardize: bool = True) -> pd.DataFrame:
    """多因子合成打分（默认等权，横截面 Z-Score 标准化后加权）。"""
    if not panel:
        return pd.DataFrame()
    keys = list(panel.keys())
    w = dict(weights or {k: 1.0 for k in keys})
    total = sum(abs(v) for k, v in w.items() if k in panel) or 1.0
    out = None
    for k in keys:
        f = panel[k]
        z = cross_zscore(f) if standardize else f
        term = z * (w.get(k, 0.0) / total)
        out = term if out is None else out.add(term, fill_value=0.0)
    return out


def forward_returns(data: DataBundle, horizon: int = 20) -> pd.DataFrame:
    """未来 N 日收益率（用于 IC 分析；只在研究阶段使用，绝不可进入策略信号）。"""
    close = data.close_matrix()
    return close.shift(-int(horizon)) / close - 1.0


def factor_ic(panel: Dict[str, pd.DataFrame], fwd: pd.DataFrame,
              method: str = "spearman") -> pd.DataFrame:
    """逐日计算因子与未来收益的截面相关系数（IC）。"""
    out = {}
    for k, f in panel.items():
        a = f.reindex_like(fwd)
        if method == "spearman":
            a = a.rank(axis=1)
            b = fwd.reindex_like(a).rank(axis=1)
        else:
            b = fwd.reindex_like(a)
        out[k] = a.corrwith(b, axis=1)
    return pd.DataFrame(out)


def ic_summary(ic: pd.DataFrame, periods: int = 252) -> pd.DataFrame:
    """IC 统计：均值、标准差、IR、胜率、t 值。"""
    rows = []
    for k in ic.columns:
        s = ic[k].dropna()
        if len(s) < 5:
            rows.append({"因子": k, "名称": FACTOR_LABELS.get(k, k), "IC均值": np.nan})
            continue
        mean, sd = float(s.mean()), float(s.std(ddof=0))
        rows.append({
            "因子": k, "名称": FACTOR_LABELS.get(k, k), "IC均值": mean, "IC标准差": sd,
            "ICIR": mean / sd * np.sqrt(periods / max(len(s), 1)) if sd > 0 else 0.0,
            "IC>0占比": float((s > 0).mean()), "t值": mean / sd * np.sqrt(len(s)) if sd > 0 else 0.0,
            "样本数": len(s),
        })
    return pd.DataFrame(rows).sort_values("IC均值", ascending=False, na_position="last")


def quantile_returns(factor: pd.DataFrame, fwd: pd.DataFrame, q: int = 5) -> pd.DataFrame:
    """分层（分位数）收益：检验因子的单调性与多空组合表现。"""
    f = factor.reindex_like(fwd)
    ranks = f.rank(axis=1, pct=True)
    labels = np.ceil(ranks * q).clip(1, q)
    out = {}
    for g in range(1, q + 1):
        mask = labels == g
        out[f"Q{g}"] = (fwd.where(mask)).mean(axis=1)
    res = pd.DataFrame(out)
    res["多空(Q{}-Q1)".format(q)] = res[f"Q{q}"] - res["Q1"]
    return res


def score_to_weights(score: pd.DataFrame, top_k: int = 5, max_weight: float = 0.25,
                     weight_mode: str = "equal", total_exposure: float = 1.0) -> pd.DataFrame:
    """综合打分 → 目标权重（供回测或实盘信号使用）。"""
    return top_k_weights(score, k=top_k, max_weight=max_weight, weight_mode=weight_mode,
                         total_exposure=total_exposure)


def factor_table(panel: Dict[str, pd.DataFrame], date, symbols: Optional[Iterable[str]] = None,
                 top: Optional[int] = None) -> pd.DataFrame:
    """某一日的因子明细表（含标准化得分与排名），用于界面展示。"""
    if not panel:
        return pd.DataFrame()
    d = pd.Timestamp(date)
    rows = {}
    for k, f in panel.items():
        if d in f.index:
            rows[FACTOR_LABELS.get(k, k)] = f.loc[d]
    df = pd.DataFrame(rows)
    if symbols is not None:
        df = df.reindex(list(symbols))
    z = df.apply(lambda s: (s - s.mean()) / (s.std(ddof=0) or np.nan))
    for c in list(df.columns):
        df[f"{c}（标准化）"] = z[c]
    df["综合分"] = z.mean(axis=1)
    df["排名"] = df["综合分"].rank(ascending=False)
    df = df.sort_values("综合分", ascending=False)
    return df.head(top) if top else df




