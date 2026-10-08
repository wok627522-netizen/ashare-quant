"""生成自包含的 HTML 回测报告（离线可打开，适合交付与研究归档）。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence

import pandas as pd

from . import charts as ch
from . import indicators as ind
from .data import DataBundle
from .metrics import METRIC_LABELS, format_metric

__all__ = ["html_report"]

CSS = """
body {font-family: "Microsoft YaHei", "PingFang SC", -apple-system, sans-serif;
      margin: 0; background: #f6f7f9; color: #1f2430;}
.wrap {max-width: 1320px; margin: 0 auto; padding: 28px 22px 60px;}
h1 {font-size: 26px; margin: 0 0 6px;}
h2 {font-size: 19px; margin: 34px 0 10px; padding-left: 10px; border-left: 4px solid #2f6feb;}
.sub {color: #667085; font-size: 13px; margin-bottom: 18px;}
.card {background: #fff; border: 1px solid #e6e9ef; border-radius: 14px; padding: 16px 18px;
       box-shadow: 0 1px 3px rgba(16,24,40,.05); margin-bottom: 18px;}
.kpis {display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 8px;}
.kpi {flex: 1 1 150px; background: #fff; border: 1px solid #e6e9ef; border-radius: 12px;
      padding: 12px 14px;}
.kpi .k {font-size: 12px; color: #667085;}
.kpi .v {font-size: 20px; font-weight: 650; margin-top: 4px;}
table {border-collapse: collapse; width: 100%; font-size: 13px;}
th, td {border-bottom: 1px solid #eef1f5; padding: 7px 10px; text-align: right;}
th:first-child, td:first-child {text-align: left;}
th {background: #f8fafc; color: #475467; font-weight: 600;}
.disclaimer {background: #fff7ed; border-left: 4px solid #f59e0b; padding: 12px 16px;
             border-radius: 8px; font-size: 13px; color: #7c2d12; margin-top: 26px;}
"""


def _kpi_html(metrics: dict, keys: Sequence[str]) -> str:
    items = []
    for k in keys:
        items.append(f'<div class="kpi"><div class="k">{METRIC_LABELS.get(k, k)}</div>'
                     f'<div class="v">{format_metric(k, metrics.get(k, 0.0))}</div></div>')
    return '<div class="kpis">' + "".join(items) + "</div>"


def html_report(results: Dict[str, object], bundle: DataBundle, path,
                title: str = "A 股量化策略回测报告",
                main_key: Optional[str] = None) -> Path:
    """把多个回测结果渲染成一个自包含 HTML 报告。

    Parameters
    ----------
    results : dict
        ``{策略名: BacktestResult}``，第一个（或 main_key）用于展示明细图表。
    """
    if not results:
        raise ValueError("results 不能为空")
    main_key = main_key or list(results)[0]
    main = results[main_key]

    figs = []
    # 1. 净值对比
    import plotly.graph_objects as go
    eq_fig = go.Figure()
    for i, (name, r) in enumerate(results.items()):
        e = r.equity / r.equity.iloc[0]
        eq_fig.add_trace(go.Scatter(x=e.index, y=e, name=name, line=dict(width=1.8, color=ch.PALETTE[i % len(ch.PALETTE)])))
    if main.benchmark is not None:
        b = main.benchmark.dropna()
        eq_fig.add_trace(go.Scatter(x=b.index, y=b / b.iloc[0], name="基准",
                                    line=dict(color="#9aa0a6", width=1.3, dash="dot")))
    eq_fig.update_layout(title="净值对比（归一化）", template="plotly_white", height=460,
                         hovermode="x unified", yaxis_title="净值", font=dict(family="Microsoft YaHei"))
    figs.append(eq_fig)

    # 2. 回撤
    dd_fig = go.Figure()
    for i, (name, r) in enumerate(results.items()):
        dd = r.equity / r.equity.cummax() - 1.0
        dd_fig.add_trace(go.Scatter(x=dd.index, y=dd, name=name, fill="tozeroy",
                                    line=dict(width=1.1, color=ch.PALETTE[i % len(ch.PALETTE)])))
    dd_fig.update_layout(title="回撤曲线", template="plotly_white", height=320,
                         hovermode="x unified", yaxis_tickformat=".1%", font=dict(family="Microsoft YaHei"))
    figs.append(dd_fig)

    # 3. 月度收益热力图 + 年度收益
    figs.append(ch.monthly_heatmap(main.monthly_table()))
    figs.append(ch.yearly_bar(main.yearly_returns()))

    # 4. 主策略持仓权重
    figs.append(ch.weights_area(main.weights))

    # 5. 主策略 K 线买卖点（取成交额最大的标的）
    if len(main.trades):
        top_sym = main.trades.groupby("symbol")["amount"].sum().idxmax()
        kdf = ind.add_all(bundle.get(top_sym))
        figs.append(ch.price_chart(kdf, main.trades[main.trades["symbol"] == top_sym],
                                   title=f"{top_sym} {bundle.name_of(top_sym)} 买卖点"))
        figs.append(ch.round_trip_scatter(main.round_trips))

    # 指标对比表
    rows = []
    for name, r in results.items():
        m = r.metrics
        rows.append({"策略": name, "累计收益": m.get("total_return", 0), "年化收益": m.get("annual_return", 0),
                     "年化波动": m.get("annual_vol", 0), "夏普": m.get("sharpe", 0),
                     "索提诺": m.get("sortino", 0), "卡玛": m.get("calmar", 0),
                     "最大回撤": m.get("max_drawdown", 0), "胜率": m.get("win_rate", 0),
                     "盈亏比": m.get("profit_factor", 0), "换手率": m.get("turnover", 0),
                     "交易次数": m.get("trade_count", 0), "超额收益": m.get("excess_return", 0)})
    cmp_df = pd.DataFrame(rows)
    fmt = {"累计收益": "{:.2%}", "年化收益": "{:.2%}", "年化波动": "{:.2%}", "最大回撤": "{:.2%}",
           "胜率": "{:.2%}", "换手率": "{:.2f}", "夏普": "{:.2f}", "索提诺": "{:.2f}",
           "卡玛": "{:.2f}", "盈亏比": "{:.2f}", "交易次数": "{:.0f}", "超额收益": "{:.2%}"}
    cmp_html = cmp_df.to_html(index=False, classes="tbl", border=0,
                              formatters={k: (lambda v, f=f: f.format(v)) for k, f in fmt.items()},
                              escape=False)

    # 主策略指标 & 明细
    kpi_keys = ["total_return", "annual_return", "annual_vol", "sharpe", "max_drawdown",
                "calmar", "win_rate", "profit_factor", "turnover", "exposure",
                "excess_return", "information_ratio"]
    kpi_html = _kpi_html(main.metrics, kpi_keys)
    trades_html = main.trades.head(200).to_html(index=False, classes="tbl", border=0) if len(main.trades) else "<p>无成交记录</p>"
    rt_html = main.round_trips.head(200).to_html(index=False, classes="tbl", border=0) if len(main.round_trips) else "<p>无完整回合交易</p>"

    parts = []
    for i, fig in enumerate(figs):
        parts.append("<div class='card'>" + fig.to_html(full_html=False, include_plotlyjs=(i == 0),
                                                        config=ch.PLOTLY_CONFIG) + "</div>")

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title><style>{CSS}</style></head>
<body><div class="wrap">
<h1>{title}</h1>
<div class="sub">生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}　|　
标的：{len(bundle.symbols)} 只　|　区间：{bundle.calendar[0]:%Y-%m-%d} ~ {bundle.calendar[-1]:%Y-%m-%d}　|　
基准：{bundle.benchmark_name}　|　主策略：{main_key}</div>
<div class="card"><h2 style="margin-top:0">核心指标 · {main_key}</h2>{kpi_html}
<div class="sub">成交 {len(main.trades)} 笔 / 回合交易 {len(main.round_trips)} 笔</div></div>
<h2>多策略对比</h2><div class="card">{cmp_html}</div>
{''.join(parts)}
<h2>成交明细（前 200 笔）</h2><div class="card">{trades_html}</div>
<h2>回合交易（前 200 笔）</h2><div class="card">{rt_html}</div>
<div class="disclaimer"><b>风险提示：</b>本报告由程序自动生成，仅用于量化研究与教学演示。
若使用离线合成数据，则数据与真实市场无关；历史回测结果不代表未来收益，不构成任何投资建议。</div>
</div></body></html>"""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html, encoding="utf-8")
    return p