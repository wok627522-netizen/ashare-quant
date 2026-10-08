#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""命令行入口（包内实现）：回测 / 寻优 / 信号 / 启动界面。

示例
----
    python cli.py list
    python cli.py demo --strategy momentum_rotation --start 2021-01-01 --end 2024-12-31
    python cli.py backtest --data ./data --strategy dual_ma --params '{"fast":5,"slow":20}'
    python cli.py optimize --strategy dual_ma --data ./data --metric sharpe
    python cli.py signals --strategy momentum_rotation --capital 500000
    python cli.py app
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from ashare_quant.backtest import BacktestConfig, Backtester, compare_strategies
from ashare_quant.data import load_bundle, load_csv_bundle, save_bundle, synthetic_bundle
from ashare_quant.metrics import METRIC_LABELS, format_metric
from ashare_quant.optimize import grid_search, pick_stable_params
from ashare_quant.signals import export_signals, latest_signals, signal_summary_text
from ashare_quant.strategies import REGISTRY, build_strategy, list_strategies
from ashare_quant.execution import RiskLimits, available_brokers, create_broker
from ashare_quant.live import LiveConfig, LiveScheduler, LiveTrader
from ashare_quant.selector import SelectionConfig, select_history, select_stocks, selection_stats
from ashare_quant.trading_calendar import calendar_info, next_trade_date, refresh_calendar

KEY_METRICS = ["total_return", "annual_return", "annual_vol", "sharpe", "sortino", "calmar",
               "max_drawdown", "win_rate", "profit_factor", "trade_count", "turnover", "exposure"]


def print_metrics(metrics: dict, title: str = "") -> None:
    if title:
        print(f"\n===== {title} =====")
    for k in KEY_METRICS:
        print(f"  {METRIC_LABELS.get(k, k):<12}: {format_metric(k, metrics.get(k, 0.0))}")


def get_bundle(args) -> "object":
    if getattr(args, "realtime", False):
        # 实时行情：新浪日线 + 实时快照（默认取最新）
        from ashare_quant.realtime import fetch_realtime_bundle, market_status
        codes = getattr(args, "codes", "") or ""
        symbols = [c.strip() for c in str(codes).replace("\n", ",").split(",") if c.strip()] or None
        days = int(getattr(args, "days", 250) or 250)
        st = market_status()
        print(f"[实时行情] {st.session}｜最近交易日 {st.last_trade_date.date()}｜"
              f"下一交易日 {st.next_trade_date.date()}｜{st.note}")
        bundle, status, quotes = fetch_realtime_bundle(symbols, days=days, workers=8,
                                                       max_symbols=120)
        print(f"[实时行情] 已获取 {len(bundle.symbols)} 只标的，行情时间 {bundle.quote_time}")
        return bundle
    if getattr(args, "data", None):
        p = Path(args.data)
        if p.suffix == ".pkl":
            return load_bundle(p)
        return load_csv_bundle(p)
    return synthetic_bundle(n_stocks=int(getattr(args, "n_stocks", 12)),
                            start=getattr(args, "start", "2021-01-01"),
                            end=getattr(args, "end", "2024-12-31"),
                            seed=int(getattr(args, "seed", 42)))


def cmd_list(_args) -> int:
    df = list_strategies()
    print(df[["key", "name", "category"]].to_string(index=False))
    print(f"\n共 {len(df)} 个策略。用 python cli.py demo --strategy <key> 试跑。")
    return 0


def cmd_demo(args) -> int:
    bundle = get_bundle(args)
    config = BacktestConfig.from_defaults()
    strategies = {REGISTRY[k].display_name: build_strategy(k) for k in REGISTRY} \
        if args.strategy == "all" else {args.strategy: build_strategy(args.strategy)}
    print(f"数据：{len(bundle.symbols)} 只标的 / {len(bundle)} 个交易日 / "
          f"{bundle.calendar[0].date()} ~ {bundle.calendar[-1].date()}")
    results = compare_strategies(bundle, strategies, config)
    rows = []
    for name, r in results.items():
        rows.append({"策略": name, "累计收益": r.metrics["total_return"],
                     "年化收益": r.metrics["annual_return"], "夏普": r.metrics["sharpe"],
                     "最大回撤": r.metrics["max_drawdown"], "胜率": r.metrics["win_rate"],
                     "换手率": r.metrics["turnover"], "交易次数": r.metrics["trade_count"]})
    df = pd.DataFrame(rows).sort_values("夏普", ascending=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    if len(results) == 1:
        print_metrics(list(results.values())[0].metrics, list(results)[0])
    return 0


def cmd_backtest(args) -> int:
    bundle = get_bundle(args)
    if args.start or args.end:
        bundle = bundle.slice(start=args.start, end=args.end)
    params = json.loads(args.params) if args.params else {}
    strat = build_strategy(args.strategy, **params)
    res = Backtester(BacktestConfig.from_defaults()).run_strategy(bundle, strat, name=strat.label)
    print_metrics(res.metrics, strat.label)
    if args.out:
        p = Path(args.out)
        if p.suffix in (".xlsx", ".xls"):
            res.to_excel(p)
        else:
            res.equity.to_frame("equity").to_csv(p, encoding="utf-8-sig")
        print(f"\n已导出：{p}")
    return 0


def cmd_optimize(args) -> int:
    bundle = get_bundle(args)
    cls = REGISTRY[args.strategy]
    grid = {k: list(v) for k, v in cls.param_space.items()}
    if args.grid:
        grid = json.loads(args.grid)
    print(f"参数网格：{grid}")
    results = grid_search(args.strategy, grid, bundle, BacktestConfig.from_defaults(),
                          metric=args.metric, train_ratio=args.train_ratio, n_jobs=args.jobs)
    if results.empty:
        print("没有有效结果")
        return 1
    from ashare_quant.labels import to_cn
    print("\nTop 10（按样本外得分，全中文标注）")
    cols = [c for c in results.columns if not c.startswith(("is_", "oos_"))][:12]
    print(to_cn(results[cols].head(10)).to_string(index=False))
    print("\n推荐参数：", pick_stable_params(results))
    if args.out:
        results.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"已导出：{args.out}")
    return 0


def cmd_signals(args) -> int:
    bundle = get_bundle(args)
    strat = build_strategy(args.strategy)
    sig = latest_signals(bundle, strat, capital=args.capital)
    print(signal_summary_text(sig, top=args.top))
    if args.out:
        export_signals(sig, args.out)
        print(f"\n已导出：{args.out}")
    return 0


def cmd_app(args) -> int:
    import subprocess
    app = Path(__file__).resolve().parents[1] / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app)]
    if args.port:
        cmd += ["--server.port", str(args.port)]
    if args.headless:
        cmd += ["--server.headless", "true"]
    print("启动：", " ".join(cmd))
    return subprocess.call(cmd)


def cmd_pick(args) -> int:
    """数据选股：输出含"买入日期"的选股清单。"""
    bundle = get_bundle(args)
    cfg = SelectionConfig(top_k=args.top_k, min_amount_yuan=args.min_amount * 1e4,
                          use_timing=not args.no_timing, max_loss_pct=args.stop,
                          take_profit_pct=args.take, max_hold_days=args.hold_days)
    res = select_stocks(bundle, cfg, capital=args.capital, as_of=args.as_of)
    print(res.summary_text())
    print(f"\n筛选漏斗：")
    print(res.funnel.to_string(index=False))
    if not res.picks.empty:
        from ashare_quant.labels import to_cn
        cols = ["buy_date", "buy_timing", "signal_date", "symbol", "name", "rank", "close", "score",
                "target_weight", "shares", "amount", "buy_price_low", "buy_price_high",
                "stop_loss_price", "take_profit_price", "max_hold_days", "reason", "warnings"]
        print("\n选股清单（全中文标注）：")
        print(to_cn(res.picks[[c for c in cols if c in res.picks.columns]]).to_string(index=False))
    if args.out:
        res.picks.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"\n已导出：{args.out}")
    if args.history:
        hist = select_history(bundle, cfg, capital=args.capital, freq=args.history_freq,
                              lookaheads=(5, 20, 60))
        print("\n历史选股统计：")
        print(selection_stats(hist).to_string(index=False))
        if args.out_history:
            hist.to_csv(args.out_history, index=False, encoding="utf-8-sig")
            print(f"已导出历史选股：{args.out_history}")
    return 0


def _make_broker(args):
    kwargs = {}
    if args.broker == "qmt":
        kwargs = {"userdata_path": args.qmt_path, "account_id": args.qmt_account}
    elif args.broker == "easytrader":
        kwargs = {"client": args.et_client, "client_path": args.et_path}
    elif args.broker in ("sim", "manual"):
        kwargs = {"initial_cash": args.capital}
    broker = create_broker(args.broker, **kwargs)
    broker.connect()
    return broker


def _make_trader(args, broker):
    sel = SelectionConfig(top_k=args.top_k, min_amount_yuan=args.min_amount * 1e4,
                          use_timing=not args.no_timing)
    limits = RiskLimits(max_order_amount=args.max_order_amount,
                        max_daily_order_amount=args.max_daily_amount,
                        max_position_weight=args.max_weight,
                        max_industry_weight=args.max_industry)
    cfg = LiveConfig(capital=args.capital, dry_run=not getattr(args, "execute", False),
                     auto_trade=bool(getattr(args, "auto", False)),
                     allow_gap_up=bool(getattr(args, "allow_gap_up", False)))
    return LiveTrader(broker, sel, limits, cfg)


def cmd_plan(args) -> int:
    """只生成交易计划（含买入日期），不执行。"""
    bundle = get_bundle(args)
    broker = _make_broker(args)
    trader = _make_trader(args, broker)
    plan = trader.build_plan(bundle, as_of=args.as_of, capital=args.capital,
                             ignore_trade_window=True)
    print(plan.summary_text())
    df = plan.to_frame()
    if not df.empty:
        print("\n委托清单：")
        print(df.to_string(index=False))
    if len(plan.rejected):
        print("\n被风控拦截：")
        print(plan.rejected.to_string(index=False))
    p = trader.save_plan(plan)
    print(f"\n计划已保存：{p}")
    return 0


def cmd_live(args) -> int:
    """生成并执行交易计划。默认 dry-run，加 --execute 才会真实下单。"""
    bundle = get_bundle(args)
    broker = _make_broker(args)
    trader = _make_trader(args, broker)
    if broker.is_real_money and args.execute and not args.yes:
        print("⚠️ 真实资金通道：请加 --yes 确认执行（建议先用 --broker sim 演练）")
        return 2
    plan = trader.build_plan(bundle, as_of=args.as_of, capital=args.capital,
                             ignore_trade_window=not args.execute)
    print(plan.summary_text())
    report = trader.execute(plan, dry_run=not args.execute, confirm=False, force=args.force)
    print("\n" + report.summary_text())
    trader.save_plan(plan)
    return 0 if not report.failed else 1


def cmd_serve(args) -> int:
    """常驻调度：盘后自动生成计划、次日开盘执行、收盘对账。"""
    bundle = get_bundle(args)
    broker = _make_broker(args)
    trader = _make_trader(args, broker)
    if broker.is_real_money and args.execute and not args.yes:
        print("⚠️ 真实资金通道：请加 --yes 确认开启自动调度")
        return 2
    print(f"调度启动：盘后 {args.plan_time} 生成计划 / 次日 {args.exec_time} 执行 / "
          f"{args.reconcile_time} 对账　dry_run={not args.execute}")
    scheduler = LiveScheduler(trader, lambda: get_bundle(args), plan_time=args.plan_time,
                              exec_time=args.exec_time, reconcile_time=args.reconcile_time,
                              poll_seconds=args.poll, capital=args.capital)
    def on_event(ev):
        print(f"\n[{ev['time']}] {ev['action']}")
        if ev.get("text"):
            print(ev["text"])
        if ev.get("error"):
            print("错误：", ev["error"])
    scheduler.run_forever(max_ticks=args.max_ticks, on_event=on_event)
    return 0


def cmd_calendar(args) -> int:
    """查看/刷新交易日历。"""
    if args.refresh:
        info = refresh_calendar()
    else:
        info = calendar_info()
    for k, v in info.items():
        print(f"  {k}: {v}")
    print(f"  今天之后第一个交易日: {next_trade_date(pd.Timestamp.now().normalize()).date()}")
    return 0


def cmd_brokers(_args) -> int:
    """列出可用交易通道。"""
    print(available_brokers().to_string(index=False))
    print("\n安全提示：默认全部为 dry-run；真实资金通道请先用 sim 演练。")
    return 0

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="A 股量化研究平台命令行工具")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_data_arg(sp):
        sp.add_argument("--data", help="本地 CSV 目录或 .pkl 缓存；缺省用离线合成数据")
        sp.add_argument("--n-stocks", type=int, default=12, help="合成数据股票数")
        sp.add_argument("--start", default="2021-01-01")
        sp.add_argument("--end", default="2024-12-31")
        sp.add_argument("--seed", type=int, default=42)

    sp = sub.add_parser("list", help="列出全部内置策略")
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("demo", help="用合成数据快速回测（离线可跑）")
    add_data_arg(sp)
    sp.add_argument("--strategy", default="all", help="策略 key，或 all 全部对比")
    sp.set_defaults(func=cmd_demo)

    sp = sub.add_parser("backtest", help="单策略回测")
    add_data_arg(sp)
    sp.add_argument("--strategy", required=True)
    sp.add_argument("--params", default="", help='JSON 参数，如 \'{"fast":5}\'')
    sp.add_argument("--out", default="", help="导出 csv/xlsx 路径")
    sp.set_defaults(func=cmd_backtest)

    sp = sub.add_parser("optimize", help="参数寻优 + 样本外验证")
    add_data_arg(sp)
    sp.add_argument("--strategy", required=True)
    sp.add_argument("--grid", default="", help="JSON 参数网格，缺省用策略内置 param_space")
    sp.add_argument("--metric", default="sharpe")
    sp.add_argument("--train-ratio", type=float, default=0.7)
    sp.add_argument("--jobs", type=int, default=1)
    sp.add_argument("--out", default="")
    sp.set_defaults(func=cmd_optimize)

    sp = sub.add_parser("signals", help="生成最新调仓信号")
    add_data_arg(sp)
    sp.add_argument("--strategy", default="momentum_rotation")
    sp.add_argument("--capital", type=float, default=1_000_000.0)
    sp.add_argument("--top", type=int, default=10)
    sp.add_argument("--out", default="")
    sp.set_defaults(func=cmd_signals)

    sp = sub.add_parser("app", help="启动 Streamlit 可视化界面")
    sp.add_argument("--port", type=int, default=8501)
    sp.add_argument("--headless", action="store_true")
    sp.set_defaults(func=cmd_app)
    sp = sub.add_parser("pick", help="数据选股（输出含买入日期的清单）")
    add_data_arg(sp)
    sp.add_argument("--realtime", action="store_true", help="使用实时行情（新浪+东财直连）")
    sp.add_argument("--codes", default="", help="实时模式股票池（逗号分隔，缺省用内置默认池）")
    sp.add_argument("--days", type=int, default=250, help="实时模式历史窗口（交易日）")
    sp.add_argument("--top-k", type=int, default=5)
    sp.add_argument("--capital", type=float, default=1_000_000.0)
    sp.add_argument("--min-amount", type=float, default=5000.0, help="最小日均成交额（万元）")
    sp.add_argument("--stop", type=float, default=0.12, help="止损幅度")
    sp.add_argument("--take", type=float, default=0.20, help="止盈幅度")
    sp.add_argument("--hold-days", type=int, default=60, help="最长持有交易日")
    sp.add_argument("--as-of", default=None, help="选股基准日 YYYY-MM-DD，默认最新交易日")
    sp.add_argument("--no-timing", action="store_true", help="关闭大盘择时")
    sp.add_argument("--history", action="store_true", help="同时回看历史选股质量")
    sp.add_argument("--history-freq", default="M", help="回看频率：W/M/20 等")
    sp.add_argument("--out", default="", help="导出选股 CSV")
    sp.add_argument("--out-history", default="", help="导出历史选股 CSV")
    sp.set_defaults(func=cmd_pick)

    for name, fn, helptext in (("plan", cmd_plan, "生成交易计划（不执行）"),
                               ("live", cmd_live, "生成并执行（默认 dry-run）"),
                               ("serve", cmd_serve, "常驻调度：盘后选股+次日执行")):
        sp = sub.add_parser(name, help=helptext)
        add_data_arg(sp)
        sp.add_argument("--broker", default="sim", help="sim / manual / qmt / easytrader")
        sp.add_argument("--capital", type=float, default=1_000_000.0)
        sp.add_argument("--top-k", type=int, default=5)
        sp.add_argument("--min-amount", type=float, default=5000.0)
        sp.add_argument("--as-of", default=None)
        sp.add_argument("--max-order-amount", type=float, default=100_000.0)
        sp.add_argument("--max-daily-amount", type=float, default=500_000.0)
        sp.add_argument("--max-weight", type=float, default=0.25)
        sp.add_argument("--max-industry", type=float, default=0.40)
        sp.add_argument("--no-timing", action="store_true")
        sp.add_argument("--qmt-path", default=None, help="QMT userdata_mini 目录")
        sp.add_argument("--qmt-account", default=None, help="QMT 资金账号")
        sp.add_argument("--et-client", default="ths", help="easytrader 客户端：ths/tdx/ht")
        sp.add_argument("--et-path", default=None, help="easytrader 下单程序路径")
        sp.add_argument("--out", default="", help="导出 CSV（pick 用）")
        if name == "plan":
            sp.set_defaults(func=fn)
        else:
            sp.add_argument("--execute", action="store_true", help="真实下单（默认 dry-run）")
            sp.add_argument("--yes", action="store_true", help="真实资金二次确认")
            sp.add_argument("--auto", action="store_true", help="关闭人工确认")
            sp.add_argument("--allow-gap-up", action="store_true", help="跳空高于计划价仍买入")
            sp.add_argument("--force", action="store_true", help="忽略买入日期限制立即执行")
            if name == "serve":
                sp.add_argument("--plan-time", default="15:10")
                sp.add_argument("--exec-time", default="09:35")
                sp.add_argument("--reconcile-time", default="15:05")
                sp.add_argument("--poll", type=int, default=30)
                sp.add_argument("--max-ticks", type=int, default=None, help="调试用：最多轮询次数")
            sp.set_defaults(func=fn)

    sp = sub.add_parser("calendar", help="查看/刷新交易日历")
    sp.add_argument("--refresh", action="store_true", help="联网刷新官方日历（需 akshare）")
    sp.set_defaults(func=cmd_calendar)

    sp = sub.add_parser("brokers", help="列出可用交易通道")
    sp.set_defaults(func=cmd_brokers)

    return p

def _setup_stdout() -> None:
    """Windows 控制台编码修正，保证中文指标名正常输出。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def main(argv=None) -> int:
    _setup_stdout()
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())




