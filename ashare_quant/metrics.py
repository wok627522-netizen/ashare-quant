"""绩效与风险指标。

输入统一为**净值序列**（equity curve）与**成交明细**，输出可直接展示的指标字典、
月度收益表、滚动指标与回合交易配对结果。
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import RISK_FREE_RATE, TRADE_DAYS_PER_YEAR

__all__ = [
    "to_returns", "drawdown_series", "max_drawdown_detail", "annualized_return",
    "annualized_vol", "sharpe_ratio", "sortino_ratio", "calmar_ratio",
    "alpha_beta", "information_ratio", "monthly_return_table", "round_trip_trades",
    "rolling_metrics", "performance_summary", "METRIC_LABELS",
]

METRIC_LABELS: Dict[str, str] = {
    "total_return": "累计收益", "annual_return": "年化收益", "annual_vol": "年化波动",
    "sharpe": "夏普比率", "sortino": "索提诺比率", "calmar": "卡玛比率",
    "max_drawdown": "最大回撤", "max_dd_duration": "最长回撤天数",
    "win_rate": "胜率", "profit_factor": "盈亏比", "trade_count": "交易次数",
    "avg_win": "平均盈利", "avg_loss": "平均亏损", "avg_holding_days": "平均持仓天数",
    "turnover": "年化换手率", "exposure": "平均仓位", "cash_ratio": "平均现金比例",
    "alpha": "年化 Alpha", "beta": "Beta", "excess_return": "超额收益",
    "information_ratio": "信息比率", "tracking_error": "跟踪误差",
    "var95": "日 VaR(95%)", "cvar95": "日 CVaR(95%)", "skew": "偏度",
    "kurtosis": "峰度", "best_month": "最佳月度", "worst_month": "最差月度",
    "positive_month_ratio": "月胜率", "benchmark_return": "基准收益",
    "monthly_win_rate": "月度跑赢比例",
}

PCT_KEYS = {"total_return", "annual_return", "annual_vol", "max_drawdown", "avg_win",
            "avg_loss", "exposure", "cash_ratio", "alpha", "excess_return",
            "tracking_error", "var95", "cvar95", "best_month", "worst_month",
            "win_rate", "positive_month_ratio", "benchmark_return", "monthly_win_rate",
            "turnover"}


# --------------------------------------------------------------------------- #
# 基础计算
# --------------------------------------------------------------------------- #
def to_returns(equity: pd.Series) -> pd.Series:
    """净值 → 日收益率。"""
    return pd.Series(equity).dropna().pct_change().dropna()


def drawdown_series(equity: pd.Series) -> pd.Series:
    """回撤序列（负值，-0.15 表示从高点回撤 15%）。"""
    e = pd.Series(equity).dropna()
    return e / e.cummax() - 1.0


def max_drawdown_detail(equity: pd.Series) -> Dict[str, object]:
    """最大回撤及发生区间、修复天数。"""
    e = pd.Series(equity).dropna()
    if len(e) < 2:
        return {"max_drawdown": 0.0, "peak_date": None, "trough_date": None,
                "recover_date": None, "duration": 0, "recover_days": 0}
    dd = e / e.cummax() - 1.0
    trough = dd.idxmin()
    peak = e.loc[:trough].idxmax()
    mdd = float(dd.min())
    after = e.loc[trough:]
    rec = after[after >= float(e.loc[peak])]
    recover = rec.index[0] if len(rec) else None
    duration = int(len(e.loc[peak:trough]))
    recover_days = int(len(e.loc[peak:recover])) if recover is not None else int(len(e.loc[peak:]))
    return {"max_drawdown": mdd, "peak_date": peak, "trough_date": trough,
            "recover_date": recover, "duration": duration, "recover_days": recover_days}


def annualized_return(equity: pd.Series, periods: int = TRADE_DAYS_PER_YEAR) -> float:
    e = pd.Series(equity).dropna()
    if len(e) < 2 or e.iloc[0] <= 0:
        return 0.0
    total = float(e.iloc[-1] / e.iloc[0])
    years = len(e) / float(periods)
    if years <= 0 or total <= 0:
        return 0.0
    return float(total ** (1.0 / years) - 1.0)


def annualized_vol(returns: pd.Series, periods: int = TRADE_DAYS_PER_YEAR) -> float:
    r = pd.Series(returns).dropna()
    return float(r.std(ddof=0) * np.sqrt(periods)) if len(r) > 1 else 0.0


def sharpe_ratio(returns: pd.Series, rf: float = RISK_FREE_RATE,
                 periods: int = TRADE_DAYS_PER_YEAR) -> float:
    r = pd.Series(returns).dropna()
    if len(r) < 2:
        return 0.0
    excess = r - rf / periods
    sd = float(excess.std(ddof=0))
    return float(excess.mean() / sd * np.sqrt(periods)) if sd > 1e-12 else 0.0


def sortino_ratio(returns: pd.Series, rf: float = RISK_FREE_RATE,
                  periods: int = TRADE_DAYS_PER_YEAR) -> float:
    r = pd.Series(returns).dropna()
    if len(r) < 2:
        return 0.0
    excess = r - rf / periods
    downside = excess[excess < 0]
    dd = float(downside.std(ddof=0))
    return float(excess.mean() / dd * np.sqrt(periods)) if len(downside) > 1 and dd > 1e-12 else 0.0


def calmar_ratio(equity: pd.Series, periods: int = TRADE_DAYS_PER_YEAR) -> float:
    mdd = abs(float(max_drawdown_detail(equity)["max_drawdown"]))
    return float(annualized_return(equity, periods) / mdd) if mdd > 1e-9 else 0.0


def alpha_beta(returns: pd.Series, benchmark_returns: pd.Series,
               rf: float = RISK_FREE_RATE, periods: int = TRADE_DAYS_PER_YEAR):
    """CAPM 年化 Alpha 与 Beta。"""
    df = pd.concat([pd.Series(returns), pd.Series(benchmark_returns)], axis=1).dropna()
    if len(df) < 20:
        return 0.0, 0.0
    r, b = df.iloc[:, 0], df.iloc[:, 1]
    var = float(b.var(ddof=0))
    beta = float(r.cov(b) / var) if var > 1e-12 else 0.0
    rf_d = rf / periods
    alpha_d = float(r.mean()) - (rf_d + beta * (float(b.mean()) - rf_d))
    return float(alpha_d * periods), beta


def information_ratio(returns: pd.Series, benchmark_returns: pd.Series,
                      periods: int = TRADE_DAYS_PER_YEAR):
    df = pd.concat([pd.Series(returns), pd.Series(benchmark_returns)], axis=1).dropna()
    if len(df) < 20:
        return 0.0, 0.0
    active = df.iloc[:, 0] - df.iloc[:, 1]
    te = float(active.std(ddof=0))
    ir = float(active.mean() / te * np.sqrt(periods)) if te > 1e-12 else 0.0
    return ir, float(te * np.sqrt(periods))


def _monthly_returns(series: pd.Series) -> pd.Series:
    """任意价格/净值序列 → 月度收益率（版本无关实现）。"""
    s = pd.Series(series).dropna()
    if s.empty:
        return pd.Series(dtype=float)
    per = s.index.to_period("M")
    last = s.groupby(per).last()
    out = last.pct_change()
    if len(last):
        out.iloc[0] = last.iloc[0] / s.iloc[0] - 1.0
    out.index = last.index
    return out.dropna()


def monthly_return_table(equity: pd.Series) -> pd.DataFrame:
    """年 × 月 收益率矩阵（用于热力图）。"""
    mret = _monthly_returns(equity)
    if mret.empty:
        return pd.DataFrame()
    df = pd.DataFrame({"year": mret.index.year, "month": mret.index.month, "ret": mret.values})
    return df.pivot(index="year", columns="month", values="ret")


def rolling_metrics(equity: pd.Series, window: int = 60,
                    periods: int = TRADE_DAYS_PER_YEAR) -> pd.DataFrame:
    """滚动年化收益 / 波动 / 夏普 / 回撤。"""
    e = pd.Series(equity).dropna()
    r = e.pct_change()
    ann_ret = (1 + r.rolling(window).mean()) ** periods - 1
    ann_vol = r.rolling(window).std(ddof=0) * np.sqrt(periods)
    sharpe = ((r.rolling(window).mean() - RISK_FREE_RATE / periods)
              / r.rolling(window).std(ddof=0).replace(0.0, np.nan) * np.sqrt(periods))
    dd = e / e.rolling(window, min_periods=1).max() - 1.0
    return pd.DataFrame({"年化收益": ann_ret, "年化波动": ann_vol,
                         "夏普": sharpe, "滚动回撤": dd})


# --------------------------------------------------------------------------- #
# 交易明细 → 回合交易
# --------------------------------------------------------------------------- #
def round_trip_trades(trades: Optional[pd.DataFrame]) -> pd.DataFrame:
    """把逐笔成交配对成回合交易（FIFO），用于计算胜率、盈亏比与持仓天数。"""
    cols = ["symbol", "entry_date", "exit_date", "shares", "entry_price",
            "exit_price", "pnl", "ret", "holding_days", "fee"]
    if trades is None or len(trades) == 0:
        return pd.DataFrame(columns=cols)
    df = trades.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    rows = []
    for sym, g in df.groupby("symbol"):
        queue = []                       # [date, price, shares, fee_per_share]
        for _, t in g.iterrows():
            qty = float(t["shares"])
            price = float(t["price"])
            fee_ps = float(t.get("total_fee", 0.0)) / qty if qty else 0.0
            if t["side"] == "buy":
                queue.append([t["date"], price, qty, fee_ps])
                continue
            remain = qty
            while remain > 1e-9 and queue:
                lot = queue[0]
                matched = min(lot[2], remain)
                cost = lot[1] * matched + lot[3] * matched
                proceeds = price * matched - fee_ps * matched
                pnl = proceeds - cost
                rows.append({
                    "symbol": sym, "entry_date": lot[0], "exit_date": t["date"],
                    "shares": matched, "entry_price": lot[1], "exit_price": price,
                    "pnl": pnl, "ret": pnl / cost if cost > 0 else 0.0,
                    "holding_days": int((t["date"] - lot[0]).days),
                    "fee": (lot[3] + fee_ps) * matched,
                })
                lot[2] -= matched
                remain -= matched
                if lot[2] <= 1e-9:
                    queue.pop(0)
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values("exit_date").reset_index(drop=True) if not out.empty else out


# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #
def performance_summary(equity: pd.Series, benchmark: Optional[pd.Series] = None,
                        trades: Optional[pd.DataFrame] = None,
                        weights: Optional[pd.DataFrame] = None,
                        periods: int = TRADE_DAYS_PER_YEAR,
                        rf: float = RISK_FREE_RATE,
                        round_trips: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    """一次性计算全部核心绩效指标。"""
    e = pd.Series(equity).dropna()
    if len(e) < 2:
        return {k: 0.0 for k in METRIC_LABELS}
    r = e.pct_change().dropna()
    dd = max_drawdown_detail(e)
    res: Dict[str, float] = {
        "total_return": float(e.iloc[-1] / e.iloc[0] - 1.0),
        "annual_return": annualized_return(e, periods),
        "annual_vol": annualized_vol(r, periods),
        "sharpe": sharpe_ratio(r, rf, periods),
        "sortino": sortino_ratio(r, rf, periods),
        "calmar": calmar_ratio(e, periods),
        "max_drawdown": float(dd["max_drawdown"]),
        "max_dd_duration": float(dd["recover_days"]),
        "var95": float(np.nanpercentile(r.to_numpy(dtype=float), 5)) if len(r) > 5 else 0.0,
        "cvar95": float(r[r <= np.nanpercentile(r.to_numpy(dtype=float), 5)].mean()) if len(r) > 5 else 0.0,
        "skew": float(r.skew()) if len(r) > 3 else 0.0,
        "kurtosis": float(r.kurtosis()) if len(r) > 3 else 0.0,
    }
    # ---- 交易层面 ----
    rt = round_trips if round_trips is not None else round_trip_trades(trades)
    if rt is not None and not rt.empty:
        wins = rt[rt["pnl"] > 0]
        losses = rt[rt["pnl"] <= 0]
        gross_win = float(wins["pnl"].sum())
        gross_loss = float(abs(losses["pnl"].sum()))
        res["win_rate"] = float(len(wins) / len(rt))
        res["profit_factor"] = float(gross_win / gross_loss) if gross_loss > 1e-9 else float("inf")
        res["avg_win"] = float(wins["pnl"].mean()) if len(wins) else 0.0
        res["avg_loss"] = float(losses["pnl"].mean()) if len(losses) else 0.0
        res["avg_holding_days"] = float(rt["holding_days"].mean())
        res["trade_count"] = float(len(rt))
    else:
        res.update({"win_rate": 0.0, "profit_factor": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
                    "avg_holding_days": 0.0, "trade_count": float(len(trades)) if trades is not None else 0.0})
    # ---- 仓位与换手 ----
    if weights is not None and not weights.empty:
        w = weights.reindex(e.index).fillna(0.0)
        exposure = w.sum(axis=1)
        res["exposure"] = float(exposure.mean())
        res["cash_ratio"] = float(1.0 - exposure.clip(upper=1.0).mean())
        turn = float(w.diff().abs().sum(axis=1).fillna(0.0).sum())
        res["turnover"] = float(turn / max(len(w) / periods, 1e-9) / 2.0)
    else:
        res.update({"exposure": 0.0, "cash_ratio": 1.0, "turnover": 0.0})
    # ---- 月度 ----
    mt = monthly_return_table(e)
    if not mt.empty:
        vals = mt.stack().astype(float)
        res["best_month"] = float(vals.max())
        res["worst_month"] = float(vals.min())
        res["positive_month_ratio"] = float((vals > 0).mean())
    # ---- 基准 ----
    if benchmark is not None and len(pd.Series(benchmark).dropna()) > 5:
        b = pd.Series(benchmark).dropna().reindex(e.index).ffill().dropna()
        br = b.pct_change().dropna()
        common = r.index.intersection(br.index)
        a, beta = alpha_beta(r.loc[common], br.loc[common], rf, periods)
        ir, te = information_ratio(r.loc[common], br.loc[common], periods)
        res["alpha"], res["beta"] = a, beta
        res["information_ratio"], res["tracking_error"] = ir, te
        res["benchmark_return"] = float(b.iloc[-1] / b.iloc[0] - 1.0) if len(b) > 1 else 0.0
        res["excess_return"] = res["total_return"] - res["benchmark_return"]
        m_b = _monthly_returns(b)
        m_s = _monthly_returns(e)
        common_m = m_s.index.intersection(m_b.index)
        res["monthly_win_rate"] = float((m_s.reindex(common_m) > m_b.reindex(common_m)).mean()) \
            if len(common_m) else 0.0
    else:
        res.update({"alpha": 0.0, "beta": 0.0, "information_ratio": 0.0,
                    "tracking_error": 0.0, "benchmark_return": 0.0,
                    "excess_return": res["total_return"], "monthly_win_rate": 0.0})
    for k in METRIC_LABELS:
        res.setdefault(k, 0.0)
    return res


def format_metric(key: str, value: float) -> str:
    """按指标类型格式化（百分比 / 倍数 / 天数）。"""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "--"
    if key in PCT_KEYS:
        return f"{value * 100:.2f}%"
    if key in {"sharpe", "sortino", "calmar", "profit_factor", "information_ratio", "beta"}:
        return f"{value:.2f}"
    if key in {"trade_count", "max_dd_duration", "avg_holding_days"}:
        return f"{value:.0f}"
    return f"{value:,.2f}"