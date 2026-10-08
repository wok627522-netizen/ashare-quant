"""下单前置风控闸门（Pre-trade Risk Gate）。

**这是实盘最关键的一层**：任何委托在进入券商通道之前，都必须先通过这里。
闸门会做两类处理：

* **拦截（block）**：违法/明显危险 → 直接拒绝，例如停牌、涨停买入、可卖不足、熔断状态；
* **削减（adjust）**：超出限额但方向正确 → 自动下调股数到合规值，例如单笔金额超限、单票超配。

覆盖的检查项（逐条对应 A 股实盘最容易踩的坑）：

1. 交易时段（默认只在 9:30-11:30 / 13:00-15:00 允许报单）
2. 黑白名单、ST、停牌、无有效行情
3. 涨跌停：涨停不追买、跌停不杀卖
4. 限价保护：委托价偏离最新价超过阈值即拦截（防止乌龙指）
5. 单笔金额上限 / 单日累计下单金额 / 单日委托笔数
6. 单票权重上限、行业权重上限、总仓位上限
7. 现金充足性与最低现金储备（预留手续费）
8. T+1：卖出数量不得超过可用数量
9. 当日亏损熔断、账户最大回撤熔断（触发后只允许卖出）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from ..rules import CostModel, get_board, round_lot
from .base import AccountSnapshot, Order, OrderSide, OrderStatus, OrderType, Quote

__all__ = ["RiskLimits", "RiskDecision", "RiskGate", "is_trading_time", "SESSION_TABLE"]


SESSION_TABLE = [
    ("09:15", "09:25", "开盘集合竞价"),
    ("09:30", "11:30", "上午连续竞价"),
    ("13:00", "14:57", "下午连续竞价"),
    ("14:57", "15:00", "收盘集合竞价"),
]


def _to_time(x: str) -> time:
    h, m = str(x).split(":")
    return time(int(h), int(m))


def is_trading_time(now: Optional[datetime] = None,
                    windows: Sequence[Tuple[str, str]] = (("09:30", "11:30"), ("13:00", "15:00"))) -> bool:
    """是否处于可报单时段（默认连续竞价时段）。"""
    now = now or datetime.now()
    if now.weekday() >= 5:
        return False
    t = now.time()
    for a, b in windows:
        if _to_time(a) <= t <= _to_time(b):
            return True
    return False


def session_name(now: Optional[datetime] = None) -> str:
    now = now or datetime.now()
    t = now.time()
    for a, b, name in SESSION_TABLE:
        if _to_time(a) <= t <= _to_time(b):
            return name
    return "非交易时段"


@dataclass
class RiskLimits:
    """实盘风控阈值（全部可在界面/配置中调整）。"""

    # 金额与笔数
    max_order_amount: float = 100_000.0        # 单笔最大委托金额（元）
    min_order_amount: float = 0.0              # 小于该金额不下单
    max_daily_order_amount: float = 500_000.0  # 单日累计下单金额
    max_daily_orders: int = 40                 # 单日最大委托笔数
    max_position_weight: float = 0.25          # 单票市值占账户比例上限
    max_industry_weight: float = 0.40          # 单行业上限
    max_total_exposure: float = 1.00           # 总仓位上限
    min_cash_reserve: float = 0.02             # 最低现金储备比例
    # 价格保护
    max_price_deviation: float = 0.03          # 限价偏离最新价上限（3%）
    allow_market_order: bool = False           # 是否允许市价单
    # 熔断
    daily_loss_limit: float = 0.05             # 当日亏损 ≥5% → 停止买入
    max_drawdown_stop: float = 0.20            # 账户回撤 ≥20% → 停止买入
    # 标的限制
    forbid_st: bool = True
    forbid_suspended: bool = True
    forbid_limit_up_buy: bool = True
    forbid_limit_down_sell: bool = True
    blacklist: Tuple[str, ...] = ()
    whitelist: Tuple[str, ...] = ()            # 非空时只允许这些标的
    # 时段
    enforce_trade_window: bool = True
    trade_windows: Tuple[Tuple[str, str], ...] = (("09:30", "11:30"), ("13:00", "15:00"))
    lot_size: int = 100
    buy_fee_rate: float = 0.00035              # 买入费率（佣金+过户费+滑点，用于资金校验）

    def describe(self) -> pd.DataFrame:
        rows = [
            ("单笔最大金额", f"{self.max_order_amount:,.0f} 元"),
            ("单日累计下单金额", f"{self.max_daily_order_amount:,.0f} 元"),
            ("单日最大委托笔数", f"{self.max_daily_orders} 笔"),
            ("单票权重上限", f"{self.max_position_weight:.0%}"),
            ("行业权重上限", f"{self.max_industry_weight:.0%}"),
            ("总仓位上限", f"{self.max_total_exposure:.0%}"),
            ("最低现金储备", f"{self.min_cash_reserve:.0%}"),
            ("限价偏离保护", f"±{self.max_price_deviation:.1%}"),
            ("允许市价单", "是" if self.allow_market_order else "否"),
            ("当日亏损熔断", f"{self.daily_loss_limit:.0%}"),
            ("账户回撤熔断", f"{self.max_drawdown_stop:.0%}"),
            ("禁止 ST", "是" if self.forbid_st else "否"),
            ("涨停不追买", "是" if self.forbid_limit_up_buy else "否"),
            ("跌停不杀卖", "是" if self.forbid_limit_down_sell else "否"),
            ("限制交易时段", "是" if self.enforce_trade_window else "否"),
        ]
        return pd.DataFrame(rows, columns=["风控项", "阈值"])


@dataclass
class RiskDecision:
    """风控结论。``adjusted_shares`` 为通过后被下调的股数。"""

    ok: bool
    code: str
    reason: str = ""
    adjusted_shares: int = 0
    level: str = "block"          # block / adjust / pass
    detail: Dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.ok


class RiskGate:
    """前置风控闸门（有状态：记录当日下单金额/笔数/起始权益）。"""

    def __init__(self, limits: Optional[RiskLimits] = None, cost: Optional[CostModel] = None):
        self.limits = limits or RiskLimits()
        self.cost = cost or CostModel()
        self.reset_day(datetime.now().date(), 0.0)

    # ---------------- 日内状态 ----------------
    def reset_day(self, date, start_equity: float = 0.0) -> None:
        self._date = date
        self._start_equity = float(start_equity)
        self._peak_equity = float(start_equity)
        self._daily_amount = 0.0
        self._daily_orders = 0
        self._halted = False
        self._halt_reason = ""

    def mark_equity(self, equity: float) -> None:
        """更新账户权益（用于熔断判断）。"""
        self._peak_equity = max(self._peak_equity, float(equity))

    @property
    def halted(self) -> bool:
        return self._halted

    def daily_state(self) -> Dict[str, Any]:
        return {"日期": self._date, "起始权益": self._start_equity,
                "当日下单金额": self._daily_amount, "当日委托笔数": self._daily_orders,
                "熔断": "是" if self._halted else "否", "熔断原因": self._halt_reason}

    # ---------------- 熔断 ----------------
    def update_circuit_breaker(self, equity: float) -> bool:
        """按当日亏损/最大回撤判断是否熔断（触发后只允许卖出）。"""
        if self._start_equity <= 0:
            return self._halted
        self.mark_equity(equity)
        day_ret = equity / self._start_equity - 1.0
        dd = equity / self._peak_equity - 1.0 if self._peak_equity > 0 else 0.0
        if day_ret <= -abs(self.limits.daily_loss_limit):
            self._halted, self._halt_reason = True, f"当日亏损 {day_ret:.2%} 触发熔断，仅允许卖出"
        elif dd <= -abs(self.limits.max_drawdown_stop):
            self._halted, self._halt_reason = True, f"账户回撤 {dd:.2%} 触发熔断，仅允许卖出"
        return self._halted

    # ---------------- 单笔检查 ----------------
    def check_order(self, order: Order, account: Optional[AccountSnapshot] = None,
                    quote: Optional[Quote] = None,
                    ctx: Optional[Dict[str, Any]] = None) -> RiskDecision:
        """检查单笔委托。

        Returns
        -------
        RiskDecision
            ``ok=True`` 表示可以下单（可能已把股数下调为 ``adjusted_shares``）。
        """
        lim = self.limits
        ctx = dict(ctx or {})
        now = ctx.get("now") or datetime.now()
        side = order.side
        sym = order.symbol
        account = account or AccountSnapshot()
        industry = ctx.get("industry", {})
        requested = int(order.shares)

        def block(code: str, reason: str) -> RiskDecision:
            return RiskDecision(False, code, reason, level="block")

        # 1) 交易时段
        if lim.enforce_trade_window and not ctx.get("ignore_trade_window") and not is_trading_time(now, lim.trade_windows):
            return block("TIME", f"当前 {now:%H:%M}（{session_name(now)}）不可报单")
        # 2) 名单
        if lim.whitelist and sym not in set(lim.whitelist):
            return block("WHITELIST", f"{sym} 不在白名单内")
        if sym in set(lim.blacklist):
            return block("BLACKLIST", f"{sym} 在黑名单内")
        # 3) 行情有效性
        if quote is None or quote.last <= 0:
            return block("NO_QUOTE", f"{sym} 无有效行情，无法校验价格")
        if lim.forbid_st and ("ST" in str(quote.name).upper() or "*ST" in str(quote.name)):
            return block("ST", f"{sym} 为 ST/风险警示股")
        if lim.forbid_suspended and (quote.suspended or quote.volume <= 0):
            return block("SUSPENDED", f"{sym} 停牌，无法交易")
        # 4) 涨跌停
        if side is OrderSide.BUY and lim.forbid_limit_up_buy and quote.at_limit_up:
            return block("LIMIT_UP", f"{sym} 已涨停（{quote.last:.2f}），不追买")
        if side is OrderSide.SELL and lim.forbid_limit_down_sell and quote.at_limit_down:
            return block("LIMIT_DOWN", f"{sym} 已跌停（{quote.last:.2f}），不杀卖")
        # 5) 价格保护
        price = order.price
        if order.order_type == OrderType.MARKET:
            if not lim.allow_market_order:
                return block("MARKET_ORDER", "当前风控不允许市价单，请改用限价单")
            price = quote.last
        if price is None or price <= 0:
            return block("BAD_PRICE", "委托价格无效")
        dev = abs(price / quote.last - 1.0)
        if lim.max_price_deviation and dev > lim.max_price_deviation:
            return block("PRICE_DEV", f"委托价 {price:.2f} 偏离最新价 {quote.last:.2f} 达 {dev:.1%}，"
                                      f"超过保护阈值 {lim.max_price_deviation:.1%}")
        # 6) 熔断
        if self._halted and side is OrderSide.BUY:
            return block("CIRCUIT_BREAKER", self._halt_reason or "已触发熔断，禁止买入")
        # 7) 数量与整手
        shares = requested
        if shares <= 0:
            return block("BAD_SHARES", "委托数量必须大于 0")
        lot = int(lim.lot_size or 100)
        if side is OrderSide.BUY and shares % lot != 0:
            shares = round_lot(shares, lot)
            if shares <= 0:
                return block("LOT", f"买入数量不足 1 手（{lot} 股）")
        # 8) 卖出可卖数量（T+1）
        if side is OrderSide.SELL:
            pos = account.positions.get(sym)
            avail = float(pos.available) if pos else 0.0
            if avail <= 0:
                return block("T1_NO_AVAILABLE", f"{sym} 可卖数量为 0（T+1 未解禁或未持仓）")
            if shares > avail:
                shares = int(avail)
                if shares <= 0:
                    return block("T1_NO_AVAILABLE", f"{sym} 可卖数量不足")
        # 9) 金额上限（单笔/单日/最小金额）
        amount = shares * float(price)
        if lim.min_order_amount and amount < lim.min_order_amount:
            return block("MIN_AMOUNT", f"委托金额 {amount:,.0f} 低于下限 {lim.min_order_amount:,.0f} 元")
        if lim.max_order_amount and amount > lim.max_order_amount:
            allowed = round_lot(lim.max_order_amount / float(price), lot if side is OrderSide.BUY else 1)
            shares = min(shares, max(int(allowed), 0))
            amount = shares * float(price)
            if shares <= 0:
                return block("MAX_ORDER", f"单笔金额超过上限 {lim.max_order_amount:,.0f} 元，且不足 1 手")
        remain_daily = lim.max_daily_order_amount - self._daily_amount
        if remain_daily <= 0:
            return block("DAILY_AMOUNT", "已达单日累计下单金额上限")
        if amount > remain_daily:
            allowed = round_lot(remain_daily / float(price), lot if side is OrderSide.BUY else 1)
            shares = min(shares, max(int(allowed), 0))
            amount = shares * float(price)
            if shares <= 0:
                return block("DAILY_AMOUNT", "剩余单日额度不足 1 手")
        if self._daily_orders >= lim.max_daily_orders:
            return block("DAILY_COUNT", f"已达单日委托笔数上限 {lim.max_daily_orders} 笔")
        # 10) 买入资金与权重
        equity = float(account.total or (account.cash + account.market_value) or 0.0)
        if side is OrderSide.BUY:
            need = amount * (1.0 + float(lim.buy_fee_rate))
            usable = float(account.available or account.cash) - lim.min_cash_reserve * equity
            if usable <= 0:
                return block("NO_CASH", "可用资金低于最低现金储备")
            if need > usable:
                allowed = round_lot(usable / (float(price) * (1 + lim.buy_fee_rate)), lot)
                shares = min(shares, int(allowed))
                amount = shares * float(price)
                need = amount * (1.0 + float(lim.buy_fee_rate))
                if shares <= 0 or need > usable:
                    return block("NO_CASH", f"可用资金 {usable:,.0f} 元不足以买入 1 手")
            # 单票权重
            pos = account.positions.get(sym)
            cur_mv = float(pos.market_value) if pos else 0.0
            if equity > 0 and lim.max_position_weight:
                cap = lim.max_position_weight * equity
                if cur_mv + amount > cap:
                    room = max(cap - cur_mv, 0.0)
                    allowed = round_lot(room / float(price), lot)
                    shares = min(shares, int(allowed))
                    amount = shares * float(price)
                    if shares <= 0:
                        return block("MAX_WEIGHT",
                                     f"{sym} 单票权重已达上限 {lim.max_position_weight:.0%}（现 {cur_mv / equity:.1%}）")
            # 行业权重
            ind = industry.get(sym)
            if ind and equity > 0 and lim.max_industry_weight:
                ind_mv = sum(p.market_value for s, p in account.positions.items() if industry.get(s) == ind)
                cap = lim.max_industry_weight * equity
                room = max(cap - ind_mv, 0.0)
                if amount > room:
                    allowed = round_lot(room / float(price), lot)
                    shares = min(shares, int(allowed))
                    amount = shares * float(price)
                    if shares <= 0:
                        return block("MAX_INDUSTRY",
                                     f"行业「{ind}」权重已达上限 {lim.max_industry_weight:.0%}")
            # 总仓位
            if equity > 0 and lim.max_total_exposure:
                total_after = account.market_value + amount
                if total_after > lim.max_total_exposure * equity:
                    room = max(lim.max_total_exposure * equity - account.market_value, 0.0)
                    allowed = round_lot(room / float(price), lot)
                    shares = min(shares, int(allowed))
                    amount = shares * float(price)
                    if shares <= 0:
                        return block("MAX_EXPOSURE",
                                     f"总仓位已达上限 {lim.max_total_exposure:.0%}")

        if shares <= 0:
            return block("ZERO_SHARES", "风控削减后委托数量为 0")
        level = "adjust" if shares < requested else "pass"
        reason = "" if level == "pass" else f"风控下调：{requested} → {shares} 股"
        return RiskDecision(True, "OK", reason, adjusted_shares=int(shares), level=level,
                            detail={"amount": round(amount, 2)})

    # ---------------- 批量检查 ----------------
    def check_plan(self, orders: Sequence[Order], account: AccountSnapshot,
                   quotes: Dict[str, Quote], ctx: Optional[Dict[str, Any]] = None,
                   reserve_ratio: float = 0.0) -> Tuple[List[Order], pd.DataFrame, List[str]]:
        """批量检查一张委托清单。

        Returns
        -------
        (approved, rejected_table, notes)
            ``approved`` 中是**已按风控削减后的** Order（可能股数与原始请求不同），
            ``rejected_table`` 列出被拦截的委托及原因。
        """
        ctx = dict(ctx or {})
        approved: List[Order] = []
        rejected: List[dict] = []
        notes: List[str] = []
        # 先卖后买：卖出释放的资金可用于买入
        ordered = sorted(orders, key=lambda o: 0 if o.side is OrderSide.SELL else 1)
        # 用副本推进账户状态，避免"同一批多笔买入重复使用同一笔现金"
        cash = float(account.available or account.cash)
        mv = float(account.market_value)
        positions = {s: p for s, p in account.positions.items()}
        sold_proceeds = 0.0

        for od in ordered:
            q = quotes.get(od.symbol)
            od_adj = Order(symbol=od.symbol, side=od.side, shares=od.shares, price=od.price,
                           order_type=od.order_type, reason=od.reason, signal_date=od.signal_date,
                           buy_date=od.buy_date, client_id=od.client_id, meta=dict(od.meta))
            snap = AccountSnapshot(total=float(account.total), cash=cash, available=cash,
                                   market_value=mv, positions=positions)
            if od.side is OrderSide.SELL and od.shares > 0:
                sold_proceeds += od.shares * float(od.price or (q.last if q else 0.0))
            dec = self.check_order(od_adj, snap, q, ctx)
            if not dec.ok:
                rejected.append({"代码": od.symbol, "方向": od.side.cn, "请求股数": od.shares,
                                 "价格": od.price, "状态": "拦截", "原因": dec.reason, "代码编号": dec.code})
                continue
            final_shares = int(dec.adjusted_shares or od.shares)
            od_adj.shares = final_shares
            approved.append(od_adj)
            if dec.level == "adjust":
                notes.append(f"{od.symbol} {od.side.cn}：{od.shares} → {final_shares} 股（{dec.reason}）")
            # 推进状态
            amount = final_shares * float(od.price or (q.last if q else 0.0))
            if od.side is OrderSide.BUY:
                cash -= amount * (1 + self.limits.buy_fee_rate)
                mv += amount
                pos = positions.get(od.symbol)
                if pos:
                    pos.shares += final_shares
                    pos.market_value  # noqa: B018  保持对象语义
                else:
                    from .base import Position as _P
                    positions[od.symbol] = _P(symbol=od.symbol, shares=final_shares, available=0.0,
                                              cost=float(od.price or (q.last if q else 0.0)),
                                              price=float(od.price or (q.last if q else 0.0)),
                                              name=(q.name if q else ""))
            else:
                cash += amount * (1 - 0.0006)
                mv = max(mv - amount, 0.0)
                pos = positions.get(od.symbol)
                if pos:
                    pos.shares = max(pos.shares - final_shares, 0.0)
                    if pos.shares <= 0:
                        positions.pop(od.symbol, None)
        # 记录当日额度
        for od in approved:
            self._daily_amount += od.amount
            self._daily_orders += 1
        return approved, pd.DataFrame(rejected), notes

    # ---------------- 账前体检 ----------------
    def pre_trade_report(self, account: AccountSnapshot,
                         quotes: Dict[str, Quote]) -> pd.DataFrame:
        """下单前账户体检表（界面展示用）。"""
        rows = []
        eq = float(account.total or (account.cash + account.market_value))
        rows.append({"检查项": "总资产", "数值": f"{eq:,.2f} 元", "结论": ""})
        rows.append({"检查项": "可用资金", "数值": f"{account.available:,.2f} 元",
                     "结论": "正常" if account.available > 0 else "无可用资金"})
        rows.append({"检查项": "当前仓位", "数值": f"{account.position_weight:.1%}",
                     "结论": "超限" if account.position_weight > self.limits.max_total_exposure else "正常"})
        rows.append({"检查项": "持仓数量", "数值": f"{len(account.positions)} 只", "结论": ""})
        sup = [s for s, q in quotes.items() if q.suspended]
        rows.append({"检查项": "停牌持仓", "数值": f"{len(sup)} 只", "结论": "；".join(sup)})
        rows.append({"检查项": "当日下单金额", "数值": f"{self._daily_amount:,.0f} / {self.limits.max_daily_order_amount:,.0f} 元",
                     "结论": "已达上限" if self._daily_amount >= self.limits.max_daily_order_amount else "正常"})
        rows.append({"检查项": "当日委托笔数", "数值": f"{self._daily_orders} / {self.limits.max_daily_orders} 笔",
                     "结论": "已达上限" if self._daily_orders >= self.limits.max_daily_orders else "正常"})
        rows.append({"检查项": "熔断状态", "数值": "已熔断" if self._halted else "正常",
                     "结论": self._halt_reason})
        return pd.DataFrame(rows)
