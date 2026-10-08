"""Plotly 图表库：A 股习惯配色（红涨绿跌），供 Streamlit 界面调用。"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

try:  # plotly 为界面依赖，无 plotly 时不影响回测功能
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as exc:  # pragma: no cover
    raise ImportError("需要安装 plotly：pip install plotly") from exc

UP, DOWN = "#e5484d", "#12a06a"       # 红涨绿跌
GRID = "rgba(140,140,140,0.18)"
PALETTE = ["#2f6feb", "#e5484d", "#12a06a", "#f2a33c", "#8b5cf6", "#0ea5e9",
           "#f43f5e", "#84cc16", "#f97316", "#14b8a6", "#a855f7", "#64748b"]

__all__ = ["price_chart", "equity_chart", "drawdown_chart", "monthly_heatmap",
           "weights_area", "round_trip_scatter", "rolling_chart", "return_distribution",
           "ic_bar", "correlation_heatmap", "yearly_bar", "factor_exposure_bar",
           "empty_figure", "PLOTLY_CONFIG"]


def _layout(fig: go.Figure, height: int = 420, title: str = "") -> go.Figure:
    fig.update_layout(
        height=height, title=title, template="plotly_white",
        margin=dict(l=40, r=20, t=50 if title else 24, b=30),
        hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                           xanchor="right", x=1, font=dict(size=11)),
        font=dict(family="Microsoft YaHei, PingFang SC, sans-serif", size=12),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, rangeslider_visible=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID)
    return fig


PLOTLY_CONFIG = {"displaylogo": False, "scrollZoom": True,
                 "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


def empty_figure(text: str = "暂无数据", height: int = 300) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=text, showarrow=False, font=dict(size=15, color="#888"))
    fig.update_layout(height=height, template="plotly_white",
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def price_chart(df: pd.DataFrame, trades: Optional[pd.DataFrame] = None,
                title: str = "", ma_cols: Sequence[str] = ("ma5", "ma20", "ma60"),
                height: int = 520) -> go.Figure:
    """K 线 + 均线 + 成交量 + 买卖点标记。"""
    if df is None or df.empty:
        return empty_figure("该标的暂无行情数据")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[0.74, 0.26])
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="K线", increasing_line_color=UP, decreasing_line_color=DOWN,
        increasing_fillcolor=UP, decreasing_fillcolor=DOWN, line=dict(width=1)), row=1, col=1)
    ma_names = {"ma5": "5日均线", "ma10": "10日均线", "ma20": "20日均线",
                "ma30": "30日均线", "ma60": "60日均线", "ma120": "120日均线"}
    for i, col in enumerate(ma_cols):
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col],
                                     name=ma_names.get(str(col).lower(), str(col)),
                                     line=dict(width=1.2, color=PALETTE[i % len(PALETTE)])),
                          row=1, col=1)
    colors = np.where(df["close"].diff().fillna(0) >= 0, UP, DOWN)
    fig.add_trace(go.Bar(x=df.index, y=df.get("volume", pd.Series(index=df.index, dtype=float)),
                         name="成交量", marker_color=colors, opacity=0.6), row=2, col=1)
    if trades is not None and not trades.empty:
        t = trades.copy()
        t["date"] = pd.to_datetime(t["date"])
        for side, color, symbol, label in (("buy", UP, "triangle-up", "买入"),
                                           ("sell", DOWN, "triangle-down", "卖出")):
            sub = t[t["side"] == side]
            if sub.empty:
                continue
            y = sub["price"].astype(float) * (0.985 if side == "buy" else 1.015)
            fig.add_trace(go.Scatter(x=sub["date"], y=y, mode="markers", name=label,
                                     marker=dict(color=color, size=10, symbol=symbol,
                                                 line=dict(width=1, color="white")),
                                     text=[f"{label} {int(s)} 股 @ {p:.2f}"
                                           for s, p in zip(sub["shares"], sub["price"])],
                                     hoverinfo="text+x"), row=1, col=1)
    _layout(fig, height, title)
    fig.update_yaxes(title_text="价格(元)", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    return fig


def equity_chart(equity: pd.Series, benchmark: Optional[pd.Series] = None,
                 normalize: bool = True, title: str = "净值曲线", height: int = 460) -> go.Figure:
    """策略净值 vs 基准净值（归一化到 1.0）。"""
    e = pd.Series(equity).dropna()
    if e.empty:
        return empty_figure()
    if normalize and e.iloc[0] != 0:
        e = e / e.iloc[0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=e.index, y=e, name="策略", line=dict(color="#2f6feb", width=2),
                             fill="tozeroy", fillcolor="rgba(47,111,235,0.08)"))
    if benchmark is not None and len(pd.Series(benchmark).dropna()) > 1:
        b = pd.Series(benchmark).dropna()
        if normalize and b.iloc[0] != 0:
            b = b / b.iloc[0]
        fig.add_trace(go.Scatter(x=b.index, y=b, name="基准", line=dict(color="#8a8f98", width=1.6,
                                                                        dash="dot")))
    _layout(fig, height, title)
    fig.update_yaxes(title_text="净值")
    return fig


def drawdown_chart(equity: pd.Series, height: int = 300, title: str = "回撤曲线") -> go.Figure:
    e = pd.Series(equity).dropna()
    if e.empty:
        return empty_figure()
    dd = e / e.cummax() - 1.0
    fig = go.Figure(go.Scatter(x=dd.index, y=dd, name="回撤", fill="tozeroy",
                               line=dict(color=DOWN, width=1.2),
                               fillcolor="rgba(18,160,106,0.18)"))
    _layout(fig, height, title)
    fig.update_yaxes(title_text="回撤", tickformat=".1%")
    return fig


def monthly_heatmap(table: pd.DataFrame, height: int = 380, title: str = "月度收益热力图") -> go.Figure:
    if table is None or table.empty:
        return empty_figure("样本不足，无法生成月度收益")
    z = table.reindex(columns=range(1, 13)).astype(float)
    text = np.where(np.isnan(z.to_numpy()), "",
                    np.vectorize(lambda v: f"{v:.1%}")(np.nan_to_num(z.to_numpy())))
    fig = go.Figure(go.Heatmap(z=z.to_numpy(), x=[f"{m}月" for m in range(1, 13)],
                               y=[str(y) for y in z.index], text=text, texttemplate="%{text}",
                               colorscale=[[0, DOWN], [0.5, "#f5f5f5"], [1, UP]],
                               zmid=0, colorbar=dict(title="收益", tickformat=".0%")))
    _layout(fig, height, title)
    return fig


def weights_area(weights: pd.DataFrame, top_n: int = 10, height: int = 380,
                 title: str = "持仓权重变化") -> go.Figure:
    if weights is None or weights.empty:
        return empty_figure("暂无持仓")
    w = weights.copy()
    keep = w.mean().sort_values(ascending=False).head(top_n).index.tolist()
    other = [c for c in w.columns if c not in keep]
    if other:
        w["其他"] = w[other].sum(axis=1)
    w = w[keep + (["其他"] if other else [])]
    fig = go.Figure()
    for i, c in enumerate(w.columns):
        fig.add_trace(go.Scatter(x=w.index, y=w[c], name=c, stackgroup="one", mode="lines",
                                 line=dict(width=0.5, color=PALETTE[i % len(PALETTE)])))
    _layout(fig, height, title)
    fig.update_yaxes(title_text="权重", tickformat=".0%")
    return fig


def round_trip_scatter(rt: pd.DataFrame, height: int = 360,
                       title: str = "回合交易盈亏") -> go.Figure:
    if rt is None or rt.empty:
        return empty_figure("暂无完整的回合交易")
    df = rt.copy()
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    colors = np.where(df["pnl"] >= 0, UP, DOWN)
    fig = go.Figure(go.Scatter(
        x=df["exit_date"], y=df["ret"], mode="markers",
        marker=dict(color=colors, size=np.clip(df["shares"] / df["shares"].max() * 22 + 6, 6, 26),
                    line=dict(width=0.8, color="white")),
        text=[f"{s} {d1:%Y-%m-%d}→{d2:%Y-%m-%d}<br>收益 {r:.2%} / 盈亏 {p:,.0f} 元"
              for s, d1, d2, r, p in zip(df["symbol"], df["entry_date"], df["exit_date"],
                                         df["ret"], df["pnl"])],
        hoverinfo="text"))
    fig.add_hline(y=0, line=dict(color="#999", width=1, dash="dot"))
    _layout(fig, height, title)
    fig.update_yaxes(title_text="单笔收益率", tickformat=".0%")
    return fig


def rolling_chart(rolling_df: pd.DataFrame, height: int = 360,
                  title: str = "滚动绩效") -> go.Figure:
    if rolling_df is None or rolling_df.empty:
        return empty_figure()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for i, c in enumerate(rolling_df.columns):
        fig.add_trace(go.Scatter(x=rolling_df.index, y=rolling_df[c], name=c,
                                 line=dict(width=1.4, color=PALETTE[i % len(PALETTE)])),
                      secondary_y=(c == "滚动回撤"))
    _layout(fig, height, title)
    return fig


def return_distribution(returns: pd.Series, height: int = 320,
                        title: str = "日收益分布") -> go.Figure:
    r = pd.Series(returns).dropna()
    if r.empty:
        return empty_figure()
    fig = go.Figure(go.Histogram(x=r, nbinsx=60, marker_color="#2f6feb", opacity=0.75, name="日收益"))
    fig.add_vline(x=float(r.mean()), line=dict(color=UP, dash="dash"), annotation_text="均值")
    _layout(fig, height, title)
    fig.update_xaxes(tickformat=".1%")
    return fig


def ic_bar(ic_summary: pd.DataFrame, height: int = 340, title: str = "因子 IC 分析") -> go.Figure:
    if ic_summary is None or ic_summary.empty:
        return empty_figure("暂无因子 IC 结果")
    df = ic_summary.dropna(subset=["IC均值"]).copy()
    colors = np.where(df["IC均值"] >= 0, UP, DOWN)
    fig = go.Figure(go.Bar(x=df["名称"], y=df["IC均值"], marker_color=colors,
                           text=[f"{v:.3f}" for v in df["IC均值"]], textposition="outside"))
    _layout(fig, height, title)
    fig.update_yaxes(title_text="IC 均值")
    return fig


def correlation_heatmap(corr: pd.DataFrame, height: int = 460,
                        title: str = "收益相关性") -> go.Figure:
    if corr is None or corr.empty:
        return empty_figure()
    fig = go.Figure(go.Heatmap(z=corr.to_numpy(), x=list(corr.columns), y=list(corr.index),
                               colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
                               colorbar=dict(title="相关系数")))
    _layout(fig, height, title)
    return fig


def yearly_bar(yearly: pd.DataFrame, height: int = 340, title: str = "年度收益对比") -> go.Figure:
    if yearly is None or yearly.empty:
        return empty_figure()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=yearly["年份"].astype(str), y=yearly["策略收益"], name="策略",
                         marker_color="#2f6feb", text=[f"{v:.1%}" for v in yearly["策略收益"]],
                         textposition="outside"))
    if "基准收益" in yearly.columns:
        fig.add_trace(go.Bar(x=yearly["年份"].astype(str), y=yearly["基准收益"], name="基准",
                             marker_color="#b8bcc4", text=[f"{v:.1%}" for v in yearly["基准收益"]],
                             textposition="outside"))
    _layout(fig, height, title)
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(barmode="group")
    return fig


def ic_series_chart(ic: pd.Series, name: str = "", height: int = 380,
                    title: str = "IC 时序与累计 IC") -> go.Figure:
    """单因子 IC 时序（柱）+ 累计 IC（线，右轴）。"""
    s = pd.Series(ic).dropna()
    if s.empty:
        return empty_figure("暂无 IC 数据")
    cum = s.cumsum()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    colors = np.where(s >= 0, UP, DOWN)
    fig.add_trace(go.Bar(x=s.index, y=s, name="每日IC", marker_color=colors, opacity=0.65),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=cum.index, y=cum, name="累计IC", mode="lines",
                             line=dict(color="#2f6feb", width=2)), secondary_y=True)
    fig.add_hline(y=0, line=dict(color="#999", width=1, dash="dot"))
    _layout(fig, height, title or f"{name} IC 时序")
    fig.update_yaxes(title_text="IC", secondary_y=False)
    fig.update_yaxes(title_text="累计 IC", secondary_y=True)
    return fig


FACTOR_CN: Dict[str, str] = {
    "mom20": "20日动量", "mom60": "60日动量", "mom120": "120日动量",
    "rev5": "5日反转", "rev20": "20日反转", "vol20": "20日低波动", "vol60": "60日低波动",
    "atr_pct": "低波幅", "trend": "趋势强度", "ma_slope20": "均线斜率", "high52": "距52周高点",
    "sharpe60": "风险调整收益", "maxdd60": "回撤浅", "liquidity": "流动性(小市值)",
    "illiq": "非流动性", "updays20": "上涨天数占比", "skew20": "收益偏度",
    "rsi_rev14": "RSI反向", "boll_pos": "布林低位", "turnover20": "换手活跃度",
    "boll_break": "跌破布林下轨", "rebound_combo": "超跌反弹组合(严格)", "rebound_soft": "超跌反弹组合(宽松)", "amount20": "成交额",
}


def ic_decay_heatmap(decay: pd.DataFrame, height: int = 480,
                     title: str = "IC 衰减（不同持有期）") -> go.Figure:
    """因子 × 持有期 的 IC 均值热力图。"""
    if decay is None or decay.empty:
        return empty_figure("暂无 IC 衰减数据")
    df = decay.copy()
    df.index = [FACTOR_CN.get(str(i), str(i)) for i in df.index]
    z = df.to_numpy(dtype=float)
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"{c}日" for c in df.columns], y=list(df.index),
        colorscale=[[0, DOWN], [0.5, "#f5f5f5"], [1, UP]], zmid=0,
        text=np.round(z, 3), texttemplate="%{text}", colorbar=dict(title="IC均值")))
    _layout(fig, height, title)
    return fig


def quantile_curve_chart(qr: pd.DataFrame, height: int = 400,
                         title: str = "因子分层累计收益（Q1 最差 → Q5 最好）") -> go.Figure:
    """分层组合的累计净值曲线。"""
    if qr is None or qr.empty:
        return empty_figure("暂无分层收益数据")
    cum = (1.0 + qr.fillna(0.0)).cumprod()
    fig = go.Figure()
    for i, c in enumerate(cum.columns):
        is_ls = c == "多空"
        fig.add_trace(go.Scatter(x=cum.index, y=cum[c], name=c,
                                 line=dict(width=2.4 if is_ls else 1.4,
                                           dash="dash" if is_ls else "solid",
                                           color="#111111" if is_ls else PALETTE[i % len(PALETTE)])))
    _layout(fig, height, title)
    fig.update_yaxes(title_text="累计净值")
    return fig


def factor_exposure_bar(exposure: pd.Series, height: int = 420,
                        title: str = "行业暴露") -> go.Figure:
    if exposure is None or len(exposure) == 0:
        return empty_figure()
    s = pd.Series(exposure).sort_values(ascending=True)
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color="#2f6feb",
                           text=[f"{v:.1%}" for v in s.values], textposition="outside"))
    _layout(fig, height, title)
    fig.update_xaxes(tickformat=".0%")
    return fig




