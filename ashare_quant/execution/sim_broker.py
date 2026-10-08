"""本地模拟盘通道（SimBroker）。

用途：
1. **实盘演练**：用真实行情、真实费用模型，但不花钱，验证整套流程；
2. **策略验证**：与回测引擎互相印证（成交/费用/T+1 逻辑一致）；
3. **故障兜底**：真实通道不可用时可切换到此通道继续运行。

状态持久化在 ``exports/sim_account.json``，支持中断恢复。
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ..config import EXPORT_DIR
from ..rules import CostModel, round_lot
from ..trading_calendar import next_trade_date
from .base import (AccountSnapshot, Broker, BrokerError, Fill, Order, OrderResult,
                   OrderSide, OrderStatus, OrderType, Position, Quote)

__all__ = ["SimBroker"]


class SimBroker(Broker):
    name = "sim"
    display_name = "本地模拟盘"
    supports_realtime_quote = False
    is_real_money = False

    def __init__(self, state_path: Optional[Path] = None, initial_cash: float = 1_000_000.0,
                 cost: Optional[CostModel] = None, reset: bool = False):
        self.state_path = Path(state_path) if state_path else (EXPORT_DIR / "sim_account.json")
        self.initial_cash = float(initial_cash)
        self.cost = cost or CostModel()
        self._connected = False
        self._quotes: Dict[str, Quote] = {}
        if reset and self.state_path.exists():
            self.state_path.unlink()
        self._load()

    # ---------------- 状态 ----------------
    def _default_state(self) -> dict:
        return {"initial_cash": self.initial_cash, "cash": self.initial_cash,
                "positions": {}, "frozen": {}, "orders": [], "fills": [], "history": []}

    def _load(self) -> None:
        if self.state_path.exists():
            try:
                self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            except Exception:
                self.state = self._default_state()
        else:
            self.state = self._default_state()
        for k, v in self._default_state().items():
            self.state.setdefault(k, v)

    def save(self) -> str:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state, ensure_ascii=False, indent=2,
                                              default=str), encoding="utf-8")
        return str(self.state_path)

    def reset(self, initial_cash: Optional[float] = None) -> None:
        if initial_cash:
            self.initial_cash = float(initial_cash)
        self.state = self._default_state()
        self.save()

    # ---------------- 连接 ----------------
    def connect(self) -> bool:
        self._connected = True
        return True

    def set_quotes(self, quotes: Dict[str, Quote]) -> None:
        """注入最新行情（由 LiveTrader 调用）。"""
        self._quotes.update(quotes)

    def quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        return {s: self._quotes[s] for s in symbols if s in self._quotes}

    # ---------------- 账户 ----------------
    def _position_price(self, sym: str) -> float:
        q = self._quotes.get(sym)
        if q and q.last > 0:
            return q.last
        return float(self.state["positions"].get(sym, {}).get("cost", 0.0))

    def account(self) -> AccountSnapshot:
        positions: Dict[str, Position] = {}
        mv = 0.0
        for sym, p in self.state["positions"].items():
            px = self._position_price(sym)
            pos = Position(symbol=sym, shares=float(p.get("shares", 0)),
                           available=float(p.get("available", 0)), cost=float(p.get("cost", 0)),
                           price=px, name=p.get("name", ""))
            positions[sym] = pos
            mv += pos.market_value
        cash = float(self.state["cash"])
        return AccountSnapshot(total=cash + mv, cash=cash, available=cash, market_value=mv,
                               positions=positions, broker=self.display_name)

    def roll_to(self, trade_date) -> None:
        """T+1 解禁：把可用日期 <= trade_date 的冻结股份转为可卖。"""
        d = pd.Timestamp(trade_date).normalize()
        frozen = self.state.get("frozen", {})
        for sym, lots in list(frozen.items()):
            keep, release = [], 0.0
            for lot in lots:
                if pd.Timestamp(lot[0]) <= d:
                    release += float(lot[1])
                else:
                    keep.append(lot)
            frozen[sym] = keep
            if release and sym in self.state["positions"]:
                self.state["positions"][sym]["available"] = \
                    float(self.state["positions"][sym].get("available", 0)) + release
        self.save()

    # ---------------- 交易 ----------------
    def place_order(self, order: Order, dry_run: bool = True) -> OrderResult:
        if not self.connected:
            raise BrokerError("模拟盘未初始化，请先 connect()")
        q = self._quotes.get(order.symbol)
        price = float(order.price or (q.last if q else 0.0))
        if price <= 0:
            return OrderResult(order, OrderStatus.REJECTED, message="无有效价格", order_id="")
        if dry_run:
            return OrderResult(order, OrderStatus.SUBMITTED, order_id=f"DRY{datetime.now():%H%M%S%f}",
                               message="dry-run：仅校验，未提交", avg_price=price)
        # 成交价含滑点
        fill_price = self.cost.slippage_price(price, order.side.value)
        shares = int(order.shares)
        amount = fill_price * shares
        fee = self.cost.fees(amount, order.side.value)
        cash = float(self.state["cash"])
        pos = self.state["positions"].get(order.symbol, {"shares": 0.0, "available": 0.0,
                                                         "cost": 0.0, "name": q.name if q else ""})
        if order.side is OrderSide.BUY:
            need = amount + fee["total_fee"]
            if need > cash + 1e-6:
                return OrderResult(order, OrderStatus.REJECTED, message=f"资金不足：需 {need:,.2f}，可用 {cash:,.2f}")
            self.state["cash"] = cash - need
            old_sh = float(pos.get("shares", 0))
            new_sh = old_sh + shares
            old_cost = float(pos.get("cost", 0)) * old_sh
            pos["shares"] = new_sh
            pos["cost"] = round((old_cost + amount + fee["total_fee"]) / new_sh, 4) if new_sh else 0.0
            pos["name"] = (q.name if q else pos.get("name", ""))
            # T+1 冻结：优先使用委托自带的买入日期，便于历史回放/测试
            base_day = pd.Timestamp(order.buy_date) if order.buy_date is not None \
                else pd.Timestamp(datetime.now().date())
            avail_day = next_trade_date(base_day)
            self.state.setdefault("frozen", {}).setdefault(order.symbol, []).append(
                [str(avail_day.date()), shares])
            self.state["positions"][order.symbol] = pos
        else:
            avail = float(pos.get("available", 0))
            if shares > avail + 1e-6:
                return OrderResult(order, OrderStatus.REJECTED,
                                   message=f"可卖不足：请求 {shares}，可卖 {avail:.0f}（T+1 约束）")
            self.state["cash"] = cash + amount - fee["total_fee"]
            pos["shares"] = float(pos.get("shares", 0)) - shares
            pos["available"] = avail - shares
            if pos["shares"] <= 1e-6:
                self.state["positions"].pop(order.symbol, None)
            else:
                self.state["positions"][order.symbol] = pos
        oid = f"SIM{datetime.now():%Y%m%d%H%M%S%f}"
        status = OrderStatus.FILLED
        self.state["orders"].append({"order_id": oid, "ts": datetime.now().isoformat(),
                                     **order.to_dict(), "status": status.value})
        self.state["fills"].append({"order_id": oid, "ts": datetime.now().isoformat(),
                                    "symbol": order.symbol, "side": order.side.value,
                                    "shares": shares, "price": round(fill_price, 3),
                                    "amount": round(amount, 2),
                                    "fee": round(fee["total_fee"], 2)})
        self.save()
        return OrderResult(order, status, order_id=oid, filled_shares=shares,
                           avg_price=round(fill_price, 3), fee=fee["total_fee"],
                           message="模拟成交")

    def orders_today(self) -> pd.DataFrame:
        rows = [r for r in self.state.get("orders", []) if str(r.get("ts", "")).startswith(datetime.now().strftime("%Y-%m-%d"))]
        if not rows:
            rows = self.state.get("orders", [])[-30:]
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=["时间", "代码", "方向", "价格", "数量", "状态"])
        return pd.DataFrame({"时间": pd.to_datetime(df["ts"]).dt.strftime("%H:%M:%S"),
                             "代码": df["symbol"], "方向": df["side_cn"], "价格": df["price"],
                             "数量": df["shares"], "状态": df["status"]})

    def fills_today(self) -> pd.DataFrame:
        rows = [r for r in self.state.get("fills", []) if str(r.get("ts", "")).startswith(datetime.now().strftime("%Y-%m-%d"))]
        if not rows:
            rows = self.state.get("fills", [])[-30:]
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=["时间", "代码", "方向", "价格", "数量", "金额"])
        return pd.DataFrame({"时间": pd.to_datetime(df["ts"]).dt.strftime("%H:%M:%S"),
                             "代码": df["symbol"], "方向": df["side"],
                             "价格": df["price"], "数量": df["shares"], "金额": df["amount"]})

