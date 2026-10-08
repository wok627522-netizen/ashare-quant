"""实盘交易引擎：把「数据选股」变成「可执行的委托」，并保证安全。

完整流程（T 日盘后 → T+1 开盘执行）
------------------------------------
1. **盘后选股**：用截至 T 日收盘的数据打分，得到选股清单，并写入 **买入日期 = T+1 交易日**；
2. **目标持仓**：把选股权重换算成目标市值 → 目标股数（整手）；
3. **差异订单**：对比券商实际持仓，生成卖出单（清仓/止损/止盈/到期）与买入单；
4. **前置风控**：所有委托过一遍 ``RiskGate``（资金、涨跌停、停牌、权重、熔断、T+1）；
5. **执行**：在**买入日期**的连续竞价时段按限价下单（默认 dry-run，需显式关闭）；
6. **落盘与对账**：委托单/成交/日志落盘，收盘与券商持仓对账。

安全默认值
----------
``LiveConfig.dry_run = True``、``auto_trade = False``，即**默认只生成计划不下单**；
关闭 dry-run 需要显式设置，并且在真实资金通道上会写入审计日志。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import EXPORT_DIR, ROOT
from .data import DataBundle
from .execution import (AccountSnapshot, Broker, Order, OrderResult, OrderSide, OrderStatus,
                        OrderType, Quote, RiskGate, RiskLimits, create_broker,
                        is_trading_time, session_name)
from .rules import CostModel, board_limit, round_lot
from .selector import SelectionConfig, SelectionResult, select_stocks
from .trading_calendar import get_calendar, next_trade_date

__all__ = ["LiveConfig", "PositionMeta", "TradingPlan", "ExecutionReport", "LiveTrader",
           "LiveScheduler"]

LOG_DIR = ROOT / "logs"
STATE_DIR = ROOT / "state"
for _d in (LOG_DIR, STATE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass
class LiveConfig:
    """实盘运行参数。"""

    capital: float = 1_000_000.0
    dry_run: bool = True                 # True = 只生成计划/校验，不真正下单
    auto_trade: bool = False             # False = 每次下单都需要人工确认
    exec_price_buffer: float = 0.01      # 限价 = 最新价 ×(1+缓冲)，且不超过涨停价
    allow_gap_up: bool = False           # 跳空高于计划买入上限时是否仍买入
    enforce_buy_date: bool = True        # 只在计划买入日期执行
    adjust_threshold: float = 0.03       # 已持仓但目标权重变化小于该值时不调仓
    sell_when_no_signal: bool = True     # 不在最新选股清单中的持仓 → 清仓
    use_market_order: bool = False       # 是否允许市价单（默认限价）
    order_type: str = "limit"
    max_hold_days_default: int = 60
    log_dir: Path = field(default_factory=lambda: LOG_DIR)
    state_dir: Path = field(default_factory=lambda: STATE_DIR)
    export_dir: Path = field(default_factory=lambda: EXPORT_DIR)


@dataclass
class PositionMeta:
    """持仓的买入日期与风控价位（用于止损/止盈/到期判断）。"""

    symbol: str
    shares: float
    entry_price: float
    entry_date: Optional[str] = None
    signal_date: Optional[str] = None
    buy_date: Optional[str] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    max_hold_days: int = 60
    name: str = ""

    def held_days(self, today: Optional[date] = None) -> int:
        if not self.entry_date:
            return 0
        try:
            d0 = pd.Timestamp(self.entry_date).date()
            return int(((today or date.today()) - d0).days)
        except Exception:
            return 0


@dataclass
class TradingPlan:
    """一张完整的交易计划：选股 → 目标持仓 → 委托清单（含**买入日期**）。"""

    signal_date: pd.Timestamp
    buy_date: pd.Timestamp
    selection: SelectionResult
    orders: List[Order] = field(default_factory=list)
    approved: List[Order] = field(default_factory=list)
    rejected: pd.DataFrame = field(default_factory=pd.DataFrame)
    notes: List[str] = field(default_factory=list)
    market_state: str = ""
    exposure: float = 0.0
    account: Optional[AccountSnapshot] = None
    quotes: Dict[str, Quote] = field(default_factory=dict)
    dry_run: bool = True
    created_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # ---------------- 展示 ----------------
    def to_frame(self, only_approved: bool = True) -> pd.DataFrame:
        orders = self.approved if only_approved else self.orders
        rows = []
        for o in orders:
            q = self.quotes.get(o.symbol)
            rows.append({
                "买入日期": (pd.Timestamp(o.buy_date).date() if o.buy_date is not None
                          else self.buy_date.date()),
                "选股日期": self.signal_date.date(),
                "代码": o.symbol,
                "名称": (q.name if q else "") or o.symbol,
                "方向": o.side.cn,
                "委托价格": o.price,
                "委托数量": o.shares,
                "预估金额": round(o.amount, 2),
                "最新价": (round(q.last, 3) if q else None),
                "涨停价": (q.limit_up if q else None),
                "跌停价": (q.limit_down if q else None),
                "状态": "待执行" if self.dry_run else "待提交",
                "说明": o.reason,
            })
        return pd.DataFrame(rows)

    def summary_text(self) -> str:
        buys = [o for o in self.approved if o.side is OrderSide.BUY]
        sells = [o for o in self.approved if o.side is OrderSide.SELL]
        lines = [f"【交易计划】选股日 {self.signal_date:%Y-%m-%d} → **买入日期 {self.buy_date:%Y-%m-%d}**",
                 f"市场状态：{self.market_state}　目标仓位：{self.exposure:.0%}　"
                 f"模式：{'dry-run（不实际下单）' if self.dry_run else '实盘下单'}",
                 f"委托 {len(self.approved)} 笔（买入 {len(buys)} / 卖出 {len(sells)}）"
                 f"{'，被拦截 ' + str(len(self.rejected)) + ' 笔' if len(self.rejected) else ''}"]
        for o in buys:
            q = self.quotes.get(o.symbol)
            nm = (q.name if q else "") or ""
            lines.append(f"  ▲ 买入 {nm}{o.symbol}　{o.shares} 股 @ 限价 {o.price:.2f}　"
                         f"买入日期 {pd.Timestamp(o.buy_date).date() if o.buy_date is not None else self.buy_date.date()}")
        for o in sells:
            q = self.quotes.get(o.symbol)
            nm = (q.name if q else "") or ""
            lines.append(f"  ▼ 卖出 {nm}{o.symbol}　{o.shares} 股 @ 限价 {o.price:.2f}　备注：{o.reason}")
        if self.notes:
            lines.append("提示：" + "；".join(self.notes[:5]))
        if len(self.rejected):
            lines.append(f"被拦截 {len(self.rejected)} 笔：" +
                         "；".join(f"{r['代码']} {r['原因']}" for _, r in self.rejected.head(5).iterrows()))
        lines.append("（研究参考，不构成投资建议；实盘请自行确认）")
        return "\n".join(lines)


@dataclass
class ExecutionReport:
    """执行结果。"""

    plan: TradingPlan
    results: List[OrderResult] = field(default_factory=list)
    dry_run: bool = True
    account_before: Optional[AccountSnapshot] = None
    account_after: Optional[AccountSnapshot] = None
    log_path: Optional[str] = None
    ts: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    @property
    def submitted(self) -> List[OrderResult]:
        return [r for r in self.results if r.ok]

    @property
    def failed(self) -> List[OrderResult]:
        return [r for r in self.results if not r.ok]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([r.to_dict() for r in self.results]) if self.results else pd.DataFrame()

    def summary_text(self) -> str:
        lines = [f"【执行结果】{self.ts}　{'dry-run' if self.dry_run else '实盘'}",
                 f"计划买入日期 {self.plan.buy_date:%Y-%m-%d}　提交 {len(self.submitted)} 笔 / "
                 f"失败 {len(self.failed)} 笔"]
        for r in self.results:
            mark = "✓" if r.ok else "✗"
            lines.append(f"  {mark} {r.order.side.cn} {r.order.symbol} {r.order.shares} 股 "
                         f"@ {r.order.price} → {r.status.value} {r.message}")
        if self.account_after is not None:
            lines.append(f"  执行后：总资产 {self.account_after.total:,.2f} 元，"
                         f"可用 {self.account_after.available:,.2f} 元，持仓 {len(self.account_after.positions)} 只")
        return "\n".join(lines)


class LiveTrader:
    """实盘/模拟盘交易引擎。"""

    def __init__(self, broker: Broker,
                 selection_config: Optional[SelectionConfig] = None,
                 risk_limits: Optional[RiskLimits] = None,
                 live_config: Optional[LiveConfig] = None,
                 cost: Optional[CostModel] = None,
                 industry_map: Optional[Dict[str, str]] = None):
        self.broker = broker
        self.sel_cfg = selection_config or SelectionConfig()
        self.limits = risk_limits or RiskLimits()
        self.cfg = live_config or LiveConfig()
        self.cost = cost or CostModel()
        self.gate = RiskGate(self.limits, self.cost)
        self.industry_map = dict(industry_map or {})
        self.state_dir = Path(self.cfg.state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._meta_path = self.state_dir / "positions_meta.json"
        self._meta: Dict[str, dict] = {}
        self._log_path: Optional[Path] = None
        self._load_meta()

    # ------------------------------------------------------------------ #
    # 状态持久化
    # ------------------------------------------------------------------ #
    def _load_meta(self) -> None:
        if self._meta_path.exists():
            try:
                self._meta = json.loads(self._meta_path.read_text(encoding="utf-8"))
            except Exception:
                self._meta = {}

    def _save_meta(self) -> None:
        self._meta_path.write_text(json.dumps(self._meta, ensure_ascii=False, indent=2, default=str),
                                   encoding="utf-8")

    def position_meta(self, symbol: str) -> Optional[PositionMeta]:
        d = self._meta.get(str(symbol))
        return PositionMeta(**d) if d else None

    def set_position_meta(self, meta: PositionMeta) -> None:
        self._meta[meta.symbol] = asdict(meta)
        self._save_meta()

    def drop_position_meta(self, symbol: str) -> None:
        self._meta.pop(str(symbol), None)
        self._save_meta()

    def meta_frame(self) -> pd.DataFrame:
        if not self._meta:
            return pd.DataFrame(columns=["代码", "名称", "买入日期", "买入价", "止损价", "止盈价", "持有天数"])
        rows = []
        for s, d in self._meta.items():
            m = PositionMeta(**d)
            rows.append({"代码": s, "名称": m.name, "买入日期": m.buy_date or m.entry_date,
                         "买入价": m.entry_price, "止损价": m.stop_loss, "止盈价": m.take_profit,
                         "持有天数": m.held_days()})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    # 日志
    # ------------------------------------------------------------------ #
    def log(self, msg: str, level: str = "INFO") -> None:
        self.cfg.log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = Path(self.cfg.log_dir) / f"live_{datetime.now():%Y%m%d}.log"
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S} [{level}] {msg}"
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def read_log(self, tail: int = 200) -> str:
        p = Path(self.cfg.log_dir) / f"live_{datetime.now():%Y%m%d}.log"
        if not p.exists():
            return ""
        lines = p.read_text(encoding="utf-8").splitlines()
        return "\n".join(lines[-tail:])

    # ------------------------------------------------------------------ #
    # 生成交易计划
    # ------------------------------------------------------------------ #
    def build_plan(self, bundle: DataBundle, as_of: Optional[pd.Timestamp] = None,
                   capital: Optional[float] = None,
                   quotes: Optional[Dict[str, Quote]] = None,
                   ignore_trade_window: bool = True) -> TradingPlan:
        """生成 T+1 交易计划（含买入日期），所有委托已通过风控闸门。"""
        capital = float(capital or self.cfg.capital)
        selection = select_stocks(bundle, self.sel_cfg, capital=capital, as_of=as_of)
        symbols = list(bundle.symbols)
        if not self.industry_map:
            self.industry_map = {s: bundle.industry_of(s) for s in symbols}
        if quotes is None:
            from .quotes import get_quotes
            fetch = set(symbols) | set(selection.picks["symbol"].tolist() if len(selection.picks) else [])
            quotes = get_quotes(sorted(fetch), broker=self.broker, bundle=bundle, as_of=selection.signal_date)
        try:
            self.broker.set_quotes(quotes)      # type: ignore[attr-defined]
        except Exception:
            pass
        account = self.broker.account()
        self.gate.reset_day(selection.signal_date.date(), account.total)
        self.gate.update_circuit_breaker(account.total)

        picks = selection.picks
        target: Dict[str, float] = {}
        pick_row: Dict[str, pd.Series] = {}
        if len(picks):
            for _, r in picks.iterrows():
                target[str(r["symbol"])] = float(r["target_weight"])
                pick_row[str(r["symbol"])] = r
        target_symbols = set(target)

        orders: List[Order] = []
        notes: List[str] = []

        # ---------- 1) 卖出：清仓 / 止损 / 止盈 / 到期 ----------
        for sym, pos in account.positions.items():
            if pos.shares <= 0:
                continue
            q = quotes.get(sym)
            price = q.last if q else pos.price
            meta = self.position_meta(sym)
            reason = ""
            if sym not in target_symbols:
                if self.cfg.sell_when_no_signal:
                    reason = "不在最新选股清单，清仓"
            elif meta is not None:
                if meta.stop_loss and price and price <= meta.stop_loss:
                    reason = f"触发止损（{price:.2f} ≤ {meta.stop_loss:.2f}）"
                elif meta.take_profit and price and price >= meta.take_profit:
                    reason = f"触发止盈（{price:.2f} ≥ {meta.take_profit:.2f}）"
                elif meta.held_days() >= int(meta.max_hold_days or self.cfg.max_hold_days_default):
                    reason = f"持有到期（{meta.held_days()} 天 ≥ {meta.max_hold_days} 天）"
            if reason:
                avail = float(pos.available)
                if avail <= 0:
                    notes.append(f"{sym} 计划卖出但可卖为 0（T+1 未解禁），本次跳过")
                    continue
                sell_price = round(float(price) * (1 - self.cfg.exec_price_buffer), 2) if price else None
                if q is not None and q.limit_down:
                    sell_price = max(sell_price or 0.0, q.limit_down)
                orders.append(Order(symbol=sym, side=OrderSide.SELL, shares=int(avail),
                                    price=sell_price or float(pos.cost), reason=reason,
                                    signal_date=selection.signal_date, buy_date=selection.buy_date,
                                    order_type=OrderType.LIMIT))

        # ---------- 2) 买入：按选股权重（用实时价重算股数） ----------
        equity = float(account.total or capital)
        for sym, weight in sorted(target.items(), key=lambda kv: -kv[1]):
            q = quotes.get(sym)
            if q is None or q.last <= 0:
                notes.append(f"{sym} 无有效行情，放弃买入")
                continue
            row = pick_row.get(sym)
            if row is not None and not self.cfg.allow_gap_up:
                cap_price = float(row.get("buy_price_high") or 0)
                if cap_price and q.last > cap_price:
                    notes.append(f"{sym} 现价 {q.last:.2f} 高于计划买入上限 {cap_price:.2f}（跳空），放弃买入")
                    continue
            pos = account.positions.get(sym)
            cur_mv = float(pos.market_value) if pos else 0.0
            cur_w = cur_mv / equity if equity else 0.0
            if abs(weight - cur_w) <= self.cfg.adjust_threshold:
                continue
            target_value = weight * equity
            delta_value = target_value - cur_mv
            if delta_value <= 0:
                continue
            limit_price = round(q.last * (1 + self.cfg.exec_price_buffer), 2)
            if q.limit_up:
                limit_price = min(limit_price, q.limit_up)
            shares = int(round_lot(delta_value / max(limit_price, 0.01), self.limits.lot_size))
            if shares <= 0:
                continue
            orders.append(Order(symbol=sym, side=OrderSide.BUY, shares=shares, price=limit_price,
                                reason=f"选股入选（综合分 {row.get('score') if row is not None else '-'}）",
                                signal_date=selection.signal_date, buy_date=selection.buy_date,
                                order_type=OrderType.MARKET if self.cfg.use_market_order else OrderType.LIMIT,
                                meta={"stop_loss": float(row.get("stop_loss_price", 0) or 0) if row is not None else 0,
                                      "take_profit": float(row.get("take_profit_price", 0) or 0) if row is not None else 0,
                                      "max_hold_days": int(row.get("max_hold_days", self.cfg.max_hold_days_default)) if row is not None else self.cfg.max_hold_days_default,
                                      "name": q.name}))

        # ---------- 3) 前置风控 ----------
        approved, rejected, risk_notes = self.gate.check_plan(
            orders, account, quotes,
            ctx={"now": datetime.now(), "industry": self.industry_map,
                 "ignore_trade_window": ignore_trade_window})
        notes.extend(risk_notes)
        if len(rejected):
            for _, r in rejected.iterrows():
                notes.append(f"{r['代码']} 被拦截：{r['原因']}")

        plan = TradingPlan(signal_date=selection.signal_date, buy_date=selection.buy_date,
                           selection=selection, orders=orders, approved=approved,
                           rejected=rejected, notes=notes, market_state=selection.market_state,
                           exposure=selection.exposure, account=account, quotes=quotes,
                           dry_run=self.cfg.dry_run)
        self.log(f"生成交易计划：signal={plan.signal_date:%Y-%m-%d} buy={plan.buy_date:%Y-%m-%d} "
                 f"orders={len(orders)} approved={len(approved)} rejected={len(rejected)} "
                 f"dry_run={plan.dry_run}")
        return plan

    # ------------------------------------------------------------------ #
    # 执行
    # ------------------------------------------------------------------ #
    def execute(self, plan: TradingPlan, dry_run: Optional[bool] = None,
                confirm: bool = True, now: Optional[datetime] = None,
                force: bool = False) -> ExecutionReport:
        """按计划下单。默认沿用 ``cfg.dry_run``，并在真实资金通道上做二次确认。"""
        now = now or datetime.now()
        dry = self.cfg.dry_run if dry_run is None else bool(dry_run)
        report = ExecutionReport(plan=plan, dry_run=dry, account_before=self.broker.account())
        # 买入日期校验
        if self.cfg.enforce_buy_date and not force:
            if pd.Timestamp(now.date()) != pd.Timestamp(plan.buy_date).normalize():
                msg = (f"今天 {now.date()} 不是计划买入日期 {plan.buy_date.date()}，未执行"
                       f"（如需补执行请设置 force=True）")
                report.log_path = str(self._log_path) if self._log_path else None
                self.log(msg, level="WARN")
                return report
        # 交易时段校验
        if not dry and self.limits.enforce_trade_window and not is_trading_time(now, self.limits.trade_windows):
            msg = f"当前 {now:%H:%M}（{session_name(now)}）非可报单时段，未执行"
            self.log(msg, level="WARN")
            return report
        # 二次确认
        if not dry and confirm and self.broker.is_real_money and self.cfg.auto_trade is False:
            self.log("真实资金通道需要人工确认，请通过界面/CLI 显式确认后执行", level="WARN")
            return report
        if not plan.approved:
            self.log("计划中没有可执行委托", level="WARN")
            return report

        for od in plan.approved:
            res = self.broker.place_order(od, dry_run=dry)
            report.results.append(res)
            self.log(f"{'DRY' if dry else 'LIVE'} {od.side.cn} {od.symbol} {od.shares}股 "
                     f"@{od.price} → {res.status.value} {res.message}")
            if res.ok and not dry:
                # 成交/委托成功后记录持仓元信息
                if od.side is OrderSide.BUY:
                    meta = PositionMeta(
                        symbol=od.symbol, shares=float(res.filled_shares or od.shares),
                        entry_price=float(res.avg_price or od.price or 0),
                        entry_date=str(now.date()), signal_date=str(plan.signal_date.date()),
                        buy_date=str(plan.buy_date.date()),
                        stop_loss=float(od.meta.get("stop_loss") or 0) or None,
                        take_profit=float(od.meta.get("take_profit") or 0) or None,
                        max_hold_days=int(od.meta.get("max_hold_days") or self.cfg.max_hold_days_default),
                        name=str(od.meta.get("name") or ""))
                    self.set_position_meta(meta)
                else:
                    pos = self.broker.account().positions.get(od.symbol)
                    if pos is None or pos.shares <= 0:
                        self.drop_position_meta(od.symbol)
        report.account_after = self.broker.account()
        # 落盘
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            p = Path(self.cfg.export_dir) / f"orders_{ts}.csv"
            p.parent.mkdir(parents=True, exist_ok=True)
            plan.to_frame().to_csv(p, index=False, encoding="utf-8-sig")
            if len(plan.rejected):
                plan.rejected.to_csv(Path(self.cfg.export_dir) / f"orders_rejected_{ts}.csv",
                                     index=False, encoding="utf-8-sig")
            df = report.to_frame()
            if not df.empty:
                df.to_csv(Path(self.cfg.export_dir) / f"order_results_{ts}.csv", index=False, encoding="utf-8-sig")
            report.log_path = str(self._log_path) if self._log_path else None
        except Exception as exc:
            self.log(f"落盘失败：{exc}", level="ERROR")
        self.log(f"执行完成：提交 {len(report.submitted)} 笔，失败 {len(report.failed)} 笔")
        return report

    def run(self, bundle: DataBundle, as_of: Optional[pd.Timestamp] = None,
            dry_run: Optional[bool] = None, confirm: bool = True,
            now: Optional[datetime] = None, force: bool = False,
            capital: Optional[float] = None) -> Tuple[TradingPlan, ExecutionReport]:
        """一步到位：生成计划 → 执行（默认 dry-run）。"""
        plan = self.build_plan(bundle, as_of=as_of, capital=capital)
        report = self.execute(plan, dry_run=dry_run, confirm=confirm, now=now, force=force)
        return plan, report

    # ------------------------------------------------------------------ #
    # 对账
    # ------------------------------------------------------------------ #
    def reconcile(self, plan: Optional[TradingPlan] = None) -> pd.DataFrame:
        """把券商实际持仓与本地记录的持仓元信息/计划做对账。"""
        acct = self.broker.account()
        rows = []
        for sym, pos in acct.positions.items():
            meta = self.position_meta(sym)
            in_meta = meta is not None
            plan_target = None
            if plan is not None and len(plan.selection.picks):
                sel = plan.selection.picks
                hit = sel[sel["symbol"] == sym]
                plan_target = float(hit["target_weight"].iloc[0]) if len(hit) else 0.0
            rows.append({"代码": sym, "名称": pos.name or (meta.name if meta else ""),
                         "券商持仓": pos.shares, "可卖": pos.available,
                         "成本价": round(pos.cost, 3), "现价": round(pos.price, 3),
                         "市值": round(pos.market_value, 2), "盈亏率": round(pos.pnl_pct, 4),
                         "本地有记录": "是" if in_meta else "否",
                         "买入日期": (meta.buy_date or meta.entry_date) if meta else "",
                         "计划目标权重": plan_target,
                         "对账结论": "正常" if in_meta else "⚠️ 本地无该持仓记录（手动买入？）"})
        # 本地有记录但券商没有
        for sym, d in self._meta.items():
            if sym not in acct.positions:
                rows.append({"代码": sym, "名称": d.get("name", ""), "券商持仓": 0, "可卖": 0,
                             "成本价": d.get("entry_price"), "现价": None, "市值": 0,
                             "盈亏率": None, "本地有记录": "是",
                             "买入日期": d.get("buy_date") or d.get("entry_date"),
                             "计划目标权重": None,
                             "对账结论": "⚠️ 本地有记录但券商无持仓（已卖出？）"})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    # 计划落盘/读取
    # ------------------------------------------------------------------ #
    def save_plan(self, plan: TradingPlan, path: Optional[Path] = None) -> Path:
        p = Path(path) if path else (Path(self.cfg.state_dir) / f"plan_{plan.buy_date:%Y%m%d}.json")
        payload = {
            "signal_date": str(plan.signal_date.date()), "buy_date": str(plan.buy_date.date()),
            "market_state": plan.market_state, "exposure": plan.exposure, "dry_run": plan.dry_run,
            "created_at": plan.created_at, "notes": plan.notes,
            "orders": [o.to_dict() for o in plan.approved],
            "rejected": plan.rejected.to_dict(orient="records") if len(plan.rejected) else [],
            "picks": plan.selection.picks.to_dict(orient="records") if len(plan.selection.picks) else [],
        }
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return p

    def load_plan_orders(self, buy_date) -> pd.DataFrame:
        p = Path(self.cfg.state_dir) / f"plan_{pd.Timestamp(buy_date):%Y%m%d}.json"
        if not p.exists():
            return pd.DataFrame()
        data = json.loads(p.read_text(encoding="utf-8"))
        return pd.DataFrame(data.get("orders", []))


class LiveScheduler:
    """极简调度器：盘后生成计划、次日开盘执行、收盘对账（可在 CLI 里常驻运行）。"""

    def __init__(self, trader: LiveTrader,
                 data_provider: Callable[[], DataBundle],
                 plan_time: str = "15:10", exec_time: str = "09:35", reconcile_time: str = "15:05",
                 poll_seconds: int = 30, capital: Optional[float] = None):
        self.trader = trader
        self.data_provider = data_provider
        self.plan_time = plan_time
        self.exec_time = exec_time
        self.reconcile_time = reconcile_time
        self.poll = int(poll_seconds)
        self.capital = capital
        self._done: Dict[str, str] = {}

    @staticmethod
    def _hm(x: str) -> dtime:
        h, m = str(x).split(":")
        return dtime(int(h), int(m))

    def tick(self, now: Optional[datetime] = None) -> Dict[str, Any]:
        """执行一次调度判断（幂等：同一任务同一天只做一次）。"""
        now = now or datetime.now()
        today = str(now.date())
        out: Dict[str, Any] = {"time": now.strftime("%Y-%m-%d %H:%M:%S"), "action": "idle"}
        # 1) 盘后生成计划
        if now.time() >= self._hm(self.plan_time) and self._done.get("plan") != today:
            try:
                bundle = self.data_provider()
                plan = self.trader.build_plan(bundle, capital=self.capital)
                p = self.trader.save_plan(plan)
                self._done["plan"] = today
                out.update({"action": "build_plan", "buy_date": str(plan.buy_date.date()),
                            "orders": len(plan.approved), "path": str(p),
                            "text": plan.summary_text()})
            except Exception as exc:
                out.update({"action": "build_plan_failed", "error": str(exc)})
            return out
        # 2) 买入日期开盘执行
        if now.time() >= self._hm(self.exec_time) and self._done.get("exec") != today:
            try:
                plan_path = Path(self.trader.cfg.state_dir) / f"plan_{now:%Y%m%d}.json"
                if plan_path.exists():
                    bundle = self.data_provider()
                    plan = self.trader.build_plan(bundle, capital=self.capital)
                    if pd.Timestamp(plan.buy_date).date() == now.date():
                        report = self.trader.execute(plan, confirm=False)
                        self._done["exec"] = today
                        out.update({"action": "execute", "submitted": len(report.submitted),
                                    "failed": len(report.failed), "text": report.summary_text()})
            except Exception as exc:
                out.update({"action": "execute_failed", "error": str(exc)})
            return out
        # 3) 收盘对账
        if now.time() >= self._hm(self.reconcile_time) and self._done.get("reconcile") != today:
            try:
                df = self.trader.reconcile()
                self._done["reconcile"] = today
                out.update({"action": "reconcile", "positions": int(len(df))})
            except Exception as exc:
                out.update({"action": "reconcile_failed", "error": str(exc)})
        return out

    def run_forever(self, max_ticks: Optional[int] = None,
                    on_event: Optional[Callable[[Dict[str, Any]], None]] = None) -> None:
        """阻塞轮询。Ctrl+C 退出（实盘建议配合系统计划任务/服务使用）。"""
        ticks = 0
        while True:
            ev = self.tick()
            if on_event and ev.get("action") != "idle":
                on_event(ev)
            ticks += 1
            if max_ticks is not None and ticks >= max_ticks:
                return
            time.sleep(self.poll)

