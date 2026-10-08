"""风险管理：仓位计算、权重约束、组合风险度量与压力测试。"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .config import TRADE_DAYS_PER_YEAR
from .data import DataBundle

__all__ = ["kelly_fraction", "vol_target_scale", "atr_position_size", "risk_parity_weights",
           "apply_position_caps", "industry_exposure", "portfolio_risk",
           "stress_test", "risk_report", "correlation_summary"]


# --------------------------------------------------------------------------- #
# 仓位计算
# --------------------------------------------------------------------------- #
def kelly_fraction(win_rate: float, payoff_ratio: float, cap: float = 0.5) -> float:
    """凯利公式仓位：f* = p - (1-p)/b，并做上限截断（半凯利更稳健）。"""
    p = float(np.clip(win_rate, 0.0, 1.0))
    b = float(payoff_ratio)
    if b <= 0:
        return 0.0
    f = p - (1 - p) / b
    return float(np.clip(f, 0.0, cap))


def vol_target_scale(returns: pd.Series, target_vol: float = 0.15, lookback: int = 60,
                     max_leverage: float = 1.0) -> pd.Series:
    """波动率目标化系数：实际波动越高仓位越低。"""
    r = pd.Series(returns).dropna()
    realized = r.rolling(int(lookback), min_periods=max(5, int(lookback) // 4)).std(ddof=0) * np.sqrt(TRADE_DAYS_PER_YEAR)
    return (float(target_vol) / realized.replace(0.0, np.nan)).clip(upper=max_leverage).fillna(1.0)


def atr_position_size(equity: float, price: float, atr: float, risk_per_trade: float = 0.01,
                      atr_mult: float = 2.0, lot: int = 100, max_weight: float = 0.25) -> int:
    """按 ATR 风险预算法计算可买股数。

    例：账户 100 万，单笔风险 1%（1 万元），价格 20 元、ATR 0.8、2 倍 ATR 止损
    → 每股风险 1.6 元 → 6250 股，再受单票权重上限约束。
    """
    if price <= 0 or atr <= 0 or equity <= 0:
        return 0
    risk_amount = float(equity) * float(risk_per_trade)
    per_share_risk = float(atr) * float(atr_mult)
    shares = risk_amount / per_share_risk
    shares = min(shares, float(equity) * float(max_weight) / float(price))
    return int(shares // lot) * lot


def risk_parity_weights(cov: pd.DataFrame) -> pd.Series:
    """简化风险平价：按波动率倒数分配（对角协方差近似），迭代两次改进。"""
    vol = np.sqrt(np.diag(cov.to_numpy(dtype=float)))
    vol = np.where(vol <= 0, np.nan, vol)
    w = 1.0 / vol
    w = np.nan_to_num(w, nan=0.0)
    if w.sum() <= 0:
        return pd.Series(0.0, index=cov.index)
    w = w / w.sum()
    for _ in range(2):
        port_vol = float(np.sqrt(w @ cov.to_numpy(dtype=float) @ w))
        if port_vol <= 0:
            break
        mrc = cov.to_numpy(dtype=float) @ w / port_vol
        target = port_vol / max(len(w), 1)
        w = w * (target / np.where(mrc <= 0, np.nan, mrc))
        w = np.nan_to_num(w, nan=0.0)
        w = w / w.sum() if w.sum() > 0 else w
    return pd.Series(w, index=cov.index)


# --------------------------------------------------------------------------- #
# 权重约束
# --------------------------------------------------------------------------- #
def apply_position_caps(weights: pd.DataFrame, meta: Optional[pd.DataFrame] = None,
                        max_stock: float = 0.25, max_industry: float = 0.40,
                        max_total: float = 1.0) -> pd.DataFrame:
    """逐行执行"单票上限 + 行业上限 + 总仓位上限"，超出部分按比例削减。"""
    w = weights.fillna(0.0).clip(lower=0.0)
    w = w.clip(upper=max_stock)
    if meta is not None and "industry" in meta.columns and max_industry:
        industries = meta["industry"].reindex(w.columns).fillna("未知")
        for _, cols in industries.groupby(industries).groups.items():
            cols = [c for c in cols if c in w.columns]
            if not cols:
                continue
            s = w[cols].sum(axis=1)
            over = s > max_industry
            if over.any():
                w.loc[over, cols] = w.loc[over, cols].mul((max_industry / s[over]).clip(upper=1.0), axis=0)
    total = w.sum(axis=1)
    over = total > max_total
    if over.any():
        w.loc[over] = w.loc[over].mul((max_total / total[over]), axis=0)
    return w


def industry_exposure(weights: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """行业权重暴露时间序列。"""
    ind = meta["industry"].reindex(weights.columns).fillna("未知")
    return weights.T.groupby(ind).sum().T


# --------------------------------------------------------------------------- #
# 组合风险度量
# --------------------------------------------------------------------------- #
def portfolio_risk(weights: pd.DataFrame, returns: pd.DataFrame, periods: int = TRADE_DAYS_PER_YEAR,
                   conf: float = 0.95) -> Dict[str, float]:
    """基于历史持仓与收益估计组合风险（波动、VaR、CVaR、集中度）。"""
    w = weights.reindex(columns=returns.columns).fillna(0.0)
    r = returns.reindex(index=w.index, columns=w.columns).fillna(0.0)
    port_ret = (w.shift(1).fillna(0.0) * r).sum(axis=1)
    vol = float(port_ret.std(ddof=0) * np.sqrt(periods))
    q = float(np.nanpercentile(port_ret.to_numpy(dtype=float), (1 - conf) * 100)) if len(port_ret) > 5 else 0.0
    tail = port_ret[port_ret <= q]
    hhi = float((w ** 2).sum(axis=1).mean())
    return {
        "年化波动": vol,
        "日波动": float(port_ret.std(ddof=0)),
        "VaR(95%,日)": abs(q),
        "CVaR(95%,日)": abs(float(tail.mean())) if len(tail) else 0.0,
        "最大单日亏损": float(port_ret.min()) if len(port_ret) else 0.0,
        "集中度HHI": hhi,
        "有效持仓数": float(1.0 / hhi) if hhi > 0 else 0.0,
        "平均仓位": float(w.sum(axis=1).mean()),
    }


def correlation_summary(returns: pd.DataFrame) -> pd.DataFrame:
    """相关性矩阵（用于检查分散化程度）。"""
    return returns.corr().round(3)


def stress_test(weights: pd.Series, returns: pd.DataFrame,
                scenarios: Optional[Dict[str, Dict[str, float]]] = None,
                beta: Optional[pd.Series] = None) -> pd.DataFrame:
    """压力测试：历史极端日 + 自定义情景（如大盘 -5%、行业 -8%）。

    scenarios 示例：``{"大盘暴跌": {"beta_adjust": -0.05}}``
    """
    w = pd.Series(weights).fillna(0.0)
    rows = []
    r = returns.reindex(columns=w.index).dropna(how="all")
    if len(r):
        worst = r.sum(axis=1).nsmallest(5)
        for d, _ in worst.items():
            day_ret = float((w * r.loc[d].fillna(0.0)).sum())
            rows.append({"情景": f"历史极端日 {pd.Timestamp(d).date()}", "组合冲击": day_ret,
                         "说明": f"当日市场平均 {r.loc[d].mean():.2%}"})
    for name, cfg in (scenarios or {}).items():
        if "beta_adjust" in cfg:
            b = beta.reindex(w.index).fillna(1.0) if beta is not None else pd.Series(1.0, index=w.index)
            shock = float((w * b * cfg["beta_adjust"]).sum())
            rows.append({"情景": name, "组合冲击": shock,
                         "说明": f"指数 {cfg['beta_adjust']:.1%}"})
        elif "uniform" in cfg:
            rows.append({"情景": name, "组合冲击": float(w.sum() * cfg["uniform"]),
                         "说明": f"全仓标的 {cfg['uniform']:.1%}"})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("组合冲击")
    return out


def risk_report(result, data: DataBundle, recent_days: int = 120) -> pd.DataFrame:
    """把回测结果整理成一份风控体检表。"""
    w = result.weights.tail(recent_days)
    ret = data.returns_matrix().reindex(w.index).fillna(0.0)
    pr = portfolio_risk(w, ret)
    rows = [{"项目": k, "数值": f"{v:.2%}" if abs(v) < 10 else f"{v:.2f}"}
            for k, v in pr.items() if k not in ("有效持仓数",)]
    rows.append({"项目": "有效持仓数", "数值": f"{pr['有效持仓数']:.1f} 只"})
    rows.append({"项目": "单票最大权重", "数值": f"{w.max().max():.2%}"})
    rows.append({"项目": "最大回撤(全样本)", "数值": f"{abs(result.metrics.get('max_drawdown', 0)):.2%}"})
    rows.append({"项目": "止损触发次数", "数值": f"{len(result.events[result.events['event'] == '止损']) if len(result.events) else 0}"})
    rows.append({"项目": "熔断次数", "数值": f"{len(result.events[result.events['event'] == '回撤熔断']) if len(result.events) else 0}"})
    return pd.DataFrame(rows)