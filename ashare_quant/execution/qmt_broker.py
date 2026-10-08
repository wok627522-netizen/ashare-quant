r"""迅投 QMT / xtquant 实盘通道适配器（**真实资金**）。

前置条件（缺一不可）
--------------------
1. 在券商开通 **QMT（迅投）** 权限（多数券商要求资产 30~50 万，需向客户经理申请）；
2. 本机安装并登录券商的 QMT 客户端（极速版/独立交易端），确认"极速交易"已开启；
3. 安装 xtquant 包（一般随 QMT 客户端自带，路径形如
   ``<QMT安装目录>\bin.x64\Lib\site-packages\xtquant``，把它加入 ``PYTHONPATH``，
   或直接使用 QMT 客户端自带的 Python 环境）；
4. 找到 ``userdata_mini`` 目录（例如 ``D:\\国金QMT交易端\\userdata_mini``）。

配置示例
--------
    broker = QMTBroker(userdata_path=r"D:\\国金QMT交易端\\userdata_mini",
                       account_id="1234567", account_type="STOCK")
    broker.connect()
    broker.place_order(Order("600519", OrderSide.BUY, 100, price=1700.0), dry_run=False)

安全说明
--------
* 本类所有下单都可由上层 ``RiskGate`` 先行校验，也可单独设置 ``dry_run``；
* ``order_stock`` 返回的委托号可用于撤单与状态查询；
* **首次实盘前请务必先用 QMT 的模拟账号跑通**（account_type 用 "STOCK" 且使用模拟资金账号）。
"""

from __future__ import annotations

import random
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from ..rules import board_limit, round_lot
from .base import (AccountSnapshot, Broker, BrokerError, NotConnectedError, Order,
                   OrderResult, OrderSide, OrderStatus, OrderType, Position, Quote)

__all__ = ["QMTBroker", "to_qmt_code", "from_qmt_code"]


def to_qmt_code(symbol: str) -> str:
    """6 位代码 → QMT 代码：600519 → 600519.SH。"""
    s = "".join(ch for ch in str(symbol) if ch.isdigit())[:6]
    if s.startswith(("6", "9")):
        return f"{s}.SH"
    if s.startswith(("4", "8", "92")):
        return f"{s}.BJ"
    return f"{s}.SZ"


def from_qmt_code(code: str) -> str:
    return str(code).split(".")[0].zfill(6)


class QMTBroker(Broker):
    name = "qmt"
    display_name = "迅投 QMT（实盘）"
    supports_realtime_quote = True
    is_real_money = True

    def __init__(self, userdata_path: Optional[str] = None, account_id: Optional[str] = None,
                 account_type: str = "STOCK", session_id: Optional[int] = None,
                 auto_subscribe: bool = True, price_type: str = "FIX_PRICE",
                 retry: int = 2):
        self.userdata_path = userdata_path
        self.account_id = account_id
        self.account_type = account_type
        self.session_id = int(session_id or random.randint(100000, 999999))
        self.auto_subscribe = auto_subscribe
        self.price_type = price_type
        self.retry = int(retry)
        self._connected = False
        self._trader = None
        self._account = None
        self._xtconstant = None
        self._xtdata = None
        self._last_error = ""
        self._order_log: List[dict] = []

    # ---------------- 依赖与连接 ----------------
    def _import_xt(self):
        try:
            from xtquant import xtconstant, xtdata  # noqa: F401
            from xtquant.xttrader import XtQuantTrader  # noqa: F401
            from xtquant.xttype import StockAccount  # noqa: F401
            return xtconstant, xtdata, XtQuantTrader, StockAccount
        except ImportError as exc:
            raise BrokerError(
                "未找到 xtquant。请确认已安装 QMT 客户端，并把其 bin.x64\\Lib\\site-packages "
                "加入 PYTHONPATH（或使用 QMT 自带 Python 运行）。原始错误：" + str(exc)) from exc

    def connect(self) -> bool:
        if not self.userdata_path or not self.account_id:
            raise BrokerError("必须提供 userdata_path（QMT 的 userdata_mini 目录）与 account_id（资金账号）")
        xtconstant, xtdata, XtQuantTrader, StockAccount = self._import_xt()
        self._xtconstant, self._xtdata = xtconstant, xtdata
        try:
            trader = XtQuantTrader(self.userdata_path, self.session_id)
            trader.start()
            res = trader.connect()
            if res != 0:
                raise BrokerError(f"QMT connect 失败，返回码 {res}（请确认 QMT 客户端已登录）")
            account = StockAccount(self.account_id, self.account_type)
            sub = trader.subscribe(account)
            if self.auto_subscribe and sub not in (0, None):
                # 订阅失败不致命，但需要提示
                self._last_error = f"账户订阅返回 {sub}"
            self._trader, self._account = trader, account
            self._connected = True
            return True
        except BrokerError:
            raise
        except Exception as exc:
            raise BrokerError(f"QMT 连接异常：{exc}") from exc

    def disconnect(self) -> None:
        try:
            if self._trader is not None:
                self._trader.stop()
        except Exception:
            pass
        self._connected = False

    def _require(self):
        if not self.connected or self._trader is None:
            raise NotConnectedError("QMT 未连接，请先 connect()")

    # ---------------- 查询 ----------------
    def account(self) -> AccountSnapshot:
        self._require()
        asset = self._trader.query_stock_asset(self._account)
        cash = float(getattr(asset, "cash", 0.0) or 0.0)
        frozen = float(getattr(asset, "frozen_cash", 0.0) or 0.0)
        mv = float(getattr(asset, "market_value", 0.0) or 0.0)
        total = float(getattr(asset, "total_asset", cash + mv) or (cash + mv))
        positions: Dict[str, Position] = {}
        try:
            for p in (self._trader.query_stock_positions(self._account) or []):
                sym = from_qmt_code(getattr(p, "stock_code", ""))
                if not sym:
                    continue
                shares = float(getattr(p, "volume", 0) or 0)
                if shares <= 0:
                    continue
                positions[sym] = Position(
                    symbol=sym, shares=shares,
                    available=float(getattr(p, "can_use_volume", 0) or 0),
                    cost=float(getattr(p, "open_price", 0) or 0),
                    price=float(getattr(p, "market_value", 0) or 0) / shares if shares else 0.0,
                    name="")
        except Exception as exc:
            self._last_error = f"持仓查询失败：{exc}"
        return AccountSnapshot(total=total, cash=cash, available=cash - frozen, market_value=mv,
                               frozen=frozen, positions=positions, broker=self.display_name)

    def quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        self._require()
        out: Dict[str, Quote] = {}
        try:
            codes = [to_qmt_code(s) for s in symbols]
            ticks = self._xtdata.get_full_tick(codes) or {}
            for s in symbols:
                code = to_qmt_code(s)
                t = ticks.get(code) or {}
                if not t:
                    continue
                last = float(t.get("lastPrice", 0) or 0)
                prev = float(t.get("lastClose", 0) or 0)
                if last <= 0:
                    continue
                pct = board_limit(s, False)
                out[s] = Quote(symbol=s, last=last, prev_close=prev,
                               open=float(t.get("open", 0) or 0),
                               high=float(t.get("high", 0) or 0),
                               low=float(t.get("low", 0) or 0),
                               volume=float(t.get("volume", 0) or 0),
                               amount=float(t.get("amount", 0) or 0),
                               limit_up=round(prev * (1 + pct), 2) if prev else None,
                               limit_down=round(prev * (1 - pct), 2) if prev else None,
                               suspended=(last <= 0 or float(t.get("volume", 0) or 0) <= 0),
                               ts=datetime.now(), source="QMT")
        except Exception as exc:
            self._last_error = f"行情查询失败：{exc}"
        return out

    def orders_today(self) -> pd.DataFrame:
        self._require()
        try:
            orders = self._trader.query_stock_orders(self._account, True) or []
        except Exception:
            return pd.DataFrame()
        rows = []
        for o in orders:
            rows.append({"时间": str(getattr(o, "order_time", "")),
                         "代码": from_qmt_code(getattr(o, "stock_code", "")),
                         "方向": "买入" if getattr(o, "order_type", 0) == 23 else "卖出",
                         "价格": getattr(o, "price", 0),
                         "数量": getattr(o, "order_volume", 0),
                         "已成交": getattr(o, "traded_volume", 0),
                         "状态": getattr(o, "order_status", ""),
                         "委托编号": getattr(o, "order_id", "")})
        return pd.DataFrame(rows)

    def fills_today(self) -> pd.DataFrame:
        self._require()
        try:
            trades = self._trader.query_stock_trades(self._account) or []
        except Exception:
            return pd.DataFrame()
        rows = []
        for t in trades:
            rows.append({"时间": str(getattr(t, "traded_time", "")),
                         "代码": from_qmt_code(getattr(t, "stock_code", "")),
                         "方向": "买入" if getattr(t, "order_type", 0) == 23 else "卖出",
                         "价格": getattr(t, "traded_price", 0),
                         "数量": getattr(t, "traded_volume", 0),
                         "金额": getattr(t, "traded_amount", 0)})
        return pd.DataFrame(rows)

    # ---------------- 交易 ----------------
    def place_order(self, order: Order, dry_run: bool = True) -> OrderResult:
        self._require()
        code = to_qmt_code(order.symbol)
        if dry_run:
            return OrderResult(order, OrderStatus.SUBMITTED, order_id=f"DRY{datetime.now():%H%M%S%f}",
                               message="dry-run：未向 QMT 发送委托", avg_price=float(order.price or 0))
        xt = self._xtconstant
        # 价格类型：默认限价 FIX_PRICE；市价可选 LATEST_PRICE
        if order.order_type == OrderType.MARKET:
            price_type = getattr(xt, "LATEST_PRICE", 5)
            price = 0.0
        else:
            price_type = getattr(xt, "FIX_PRICE", 11)
            price = float(order.price or 0.0)
        side_const = getattr(xt, "STOCK_BUY", 23) if order.side is OrderSide.BUY else getattr(xt, "STOCK_SELL", 24)
        last_err = ""
        for attempt in range(max(1, self.retry + 1)):
            try:
                oid = self._trader.order_stock(self._account, code, side_const, int(order.shares),
                                               price_type, price, "ashare_quant",
                                               order.reason or "quant")
                if oid is None or (isinstance(oid, int) and oid < 0):
                    last_err = f"QMT 返回委托号 {oid}（通常表示被拒：资金/持仓/权限不足）"
                    time.sleep(0.6)
                    continue
                rec = {"ts": datetime.now().isoformat(), "order_id": oid, **order.to_dict()}
                self._order_log.append(rec)
                return OrderResult(order, OrderStatus.SUBMITTED, order_id=str(oid),
                                   message="已提交 QMT 委托", avg_price=price)
            except Exception as exc:
                last_err = str(exc)
                time.sleep(0.6 * (attempt + 1))
        return OrderResult(order, OrderStatus.REJECTED, message=f"QMT 下单失败：{last_err}")

    def cancel(self, order_id: str) -> OrderResult:
        self._require()
        try:
            res = self._trader.cancel_order_stock_async(self._account, int(order_id))
            ok = res in (0, None)
            dummy = Order(symbol="000000", side=OrderSide.BUY, shares=0, price=0.01)
            return OrderResult(dummy, OrderStatus.CANCELLED if ok else OrderStatus.REJECTED,
                               order_id=str(order_id), message=f"撤单返回 {res}")
        except Exception as exc:
            dummy = Order(symbol="000000", side=OrderSide.BUY, shares=0, price=0.01)
            return OrderResult(dummy, OrderStatus.REJECTED, order_id=str(order_id),
                               message=f"撤单失败：{exc}")

    # ---------------- 诊断 ----------------
    def diagnostics(self) -> Dict[str, Any]:
        """连接自检信息（界面展示用）。"""
        info = {"通道": self.display_name, "userdata_path": self.userdata_path,
                "account_id": self.account_id, "已连接": self.connected,
                "最近错误": self._last_error}
        if self.connected:
            try:
                a = self.account()
                info.update({"总资产": a.total, "可用资金": a.available, "持仓数": len(a.positions)})
            except Exception as exc:
                info["查询错误"] = str(exc)
        return info

