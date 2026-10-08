"""手动执行通道（ManualBroker）：程序生成委托单 → 人工在券商 App 下单 → 回填成交。

这是**最稳妥、最合规**的"半自动实盘"方式：

1. 程序完成选股、风控、仓位与股数计算，输出一张委托单（CSV/Excel，含买入日期与限价）；
2. 你在券商 App 里照着下单（或把 CSV 导入券商的批量下单工具）；
3. 把实际成交价/数量填回（或在界面里点"标记已成交"），程序继续跟踪持仓与后续卖出信号。

它不依赖任何券商接口，任何账户都能用；代价是需要人工执行。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ..config import EXPORT_DIR
from ..rules import CostModel
from .base import (AccountSnapshot, Broker, Fill, Order, OrderResult, OrderSide,
                   OrderStatus, Position, Quote)

__all__ = ["ManualBroker"]


class ManualBroker(Broker):
    name = "manual"
    display_name = "手动执行（导出委托单）"
    supports_realtime_quote = False
    is_real_money = True          # 最终由人工在真实账户执行

    def __init__(self, state_path: Optional[Path] = None, initial_cash: float = 1_000_000.0,
                 out_dir: Optional[Path] = None, cost: Optional[CostModel] = None, reset: bool = False):
        self.state_path = Path(state_path) if state_path else (EXPORT_DIR / "manual_account.json")
        self.out_dir = Path(out_dir) if out_dir else EXPORT_DIR
        self.initial_cash = float(initial_cash)
        self.cost = cost or CostModel()
        self._connected = False
        self._quotes: Dict[str, Quote] = {}
        self._pending: List[Order] = []
        if reset and self.state_path.exists():
            self.state_path.unlink()
        self._load()

    def _default_state(self) -> dict:
        return {"initial_cash": self.initial_cash, "cash": self.initial_cash,
                "positions": {}, "orders": [], "fills": []}

    def _load(self) -> None:
        self.state = json.loads(self.state_path.read_text(encoding="utf-8")) \
            if self.state_path.exists() else self._default_state()
        for k, v in self._default_state().items():
            self.state.setdefault(k, v)

    def save(self) -> str:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state, ensure_ascii=False, indent=2, default=str),
                                   encoding="utf-8")
        return str(self.state_path)

    def connect(self) -> bool:
        self._connected = True
        return True

    def set_quotes(self, quotes: Dict[str, Quote]) -> None:
        self._quotes.update(quotes)

    def quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        return {s: self._quotes[s] for s in symbols if s in self._quotes}

    def account(self) -> AccountSnapshot:
        positions, mv = {}, 0.0
        for sym, p in self.state["positions"].items():
            px = float(self._quotes.get(sym).last) if self._quotes.get(sym) else float(p.get("cost", 0))
            pos = Position(symbol=sym, shares=float(p.get("shares", 0)),
                           available=float(p.get("shares", 0)),   # 手动模式下不做 T+1 冻结
                           cost=float(p.get("cost", 0)), price=px, name=p.get("name", ""))
            positions[sym] = pos
            mv += pos.market_value
        cash = float(self.state["cash"])
        return AccountSnapshot(total=cash + mv, cash=cash, available=cash, market_value=mv,
                               positions=positions, broker=self.display_name)

    def place_order(self, order: Order, dry_run: bool = True) -> OrderResult:
        if dry_run:
            return OrderResult(order, OrderStatus.SUBMITTED, order_id=f"PLAN{datetime.now():%H%M%S%f}",
                               message="dry-run：未生成委托单")
        oid = f"MAN{datetime.now():%Y%m%d%H%M%S%f}"
        self.state["orders"].append({"order_id": oid, "ts": datetime.now().isoformat(),
                                     **order.to_dict(), "status": OrderStatus.SUBMITTED.value})
        self._pending.append(order)
        self.save()
        return OrderResult(order, OrderStatus.SUBMITTED, order_id=oid,
                           message="已生成委托单，请在券商 App 手动执行后回填成交")

    # ---------------- 导出与回填 ----------------
    def export_orders(self, path: Optional[Path] = None) -> Path:
        """把待执行委托导出为 CSV（含买入日期、限价、股数、金额）。"""
        p = Path(path) if path else (self.out_dir / f"manual_orders_{datetime.now():%Y%m%d_%H%M%S}.csv")
        p.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for o in self._pending:
            d = o.to_dict()
            rows.append({"买入/卖出日期": d["buy_date"] or datetime.now().strftime("%Y-%m-%d"),
                         "代码": o.symbol, "方向": o.side.cn, "委托价格": o.price,
                         "委托数量": o.shares, "预估金额": round(o.amount, 2),
                         "订单类型": "限价", "备注": o.reason})
        pd.DataFrame(rows).to_csv(p, index=False, encoding="utf-8-sig")
        return p

    def mark_filled(self, symbol: str, side: str, shares: float, price: float,
                    fee: float = 0.0, date: Optional[str] = None) -> dict:
        """人工回填成交（更新本地镜像账户）。"""
        side_e = OrderSide(side)
        sym = str(symbol).zfill(6)
        amount = float(shares) * float(price)
        pos = self.state["positions"].get(sym, {"shares": 0.0, "cost": 0.0, "name": ""})
        if side_e is OrderSide.BUY:
            self.state["cash"] = float(self.state["cash"]) - amount - fee
            old_sh = float(pos.get("shares", 0))
            new_sh = old_sh + float(shares)
            old_cost = float(pos.get("cost", 0)) * old_sh
            pos["cost"] = round((old_cost + amount + fee) / new_sh, 4) if new_sh else 0.0
            pos["shares"] = new_sh
            self.state["positions"][sym] = pos
        else:
            self.state["cash"] = float(self.state["cash"]) + amount - fee
            pos["shares"] = max(float(pos.get("shares", 0)) - float(shares), 0.0)
            if pos["shares"] <= 0:
                self.state["positions"].pop(sym, None)
            else:
                self.state["positions"][sym] = pos
        rec = {"ts": datetime.now().isoformat(), "date": date or str(datetime.now().date()),
               "symbol": sym, "side": side_e.value, "shares": float(shares),
               "price": float(price), "amount": round(amount, 2), "fee": float(fee)}
        self.state["fills"].append(rec)
        self._pending = [o for o in self._pending if not (o.symbol == sym and o.side is side_e)]
        self.save()
        return rec

    def import_fills(self, path: Path) -> pd.DataFrame:
        """从 CSV 批量回填成交，列名：symbol/side/shares/price/fee/date。"""
        df = pd.read_csv(path)
        out = []
        for _, r in df.iterrows():
            out.append(self.mark_filled(r["symbol"], r["side"], r["shares"], r["price"],
                                        float(r.get("fee", 0) or 0), r.get("date")))
        return pd.DataFrame(out)

    def orders_today(self) -> pd.DataFrame:
        rows = [r for r in self.state.get("orders", []) if str(r.get("ts", ""))[:10] == str(datetime.now().date())]
        if not rows:
            rows = self.state.get("orders", [])[-30:]
        if not rows:
            return pd.DataFrame(columns=["时间", "代码", "方向", "价格", "数量", "状态"])
        df = pd.DataFrame(rows)
        return pd.DataFrame({"时间": pd.to_datetime(df["ts"]).dt.strftime("%H:%M:%S"),
                             "代码": df["symbol"], "方向": df["side_cn"], "价格": df["price"],
                             "数量": df["shares"], "状态": df["status"]})

    def fills_today(self) -> pd.DataFrame:
        rows = self.state.get("fills", [])
        if not rows:
            return pd.DataFrame(columns=["时间", "代码", "方向", "价格", "数量", "金额"])
        df = pd.DataFrame(rows)
        return pd.DataFrame({"时间": df["date"], "代码": df["symbol"], "方向": df["side"],
                             "价格": df["price"], "数量": df["shares"], "金额": df["amount"]})
