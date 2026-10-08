"""交易执行层：券商无关的抽象接口与数据结构。

支持的通道（见 ``create_broker``）：

======  ==========================  ==========  ====================================
名称     说明                         真实下单     前置条件
======  ==========================  ==========  ====================================
sim      本地模拟盘（推荐先跑）         否          无
manual   生成委托单 + 手动执行 + 回读   否（人工）   任意券商 App
qmt      迅投 QMT / xtquant            **是**      券商开通 QMT、客户端登录、xtquant
easytrader  easytrader 客户端自动化      **是**      本机券商客户端登录 + easytrader
======  ==========================  ==========  ====================================

设计原则
--------
1. **所有真实下单都经过 RiskGate 前置风控**，默认 ``dry_run=True``；
2. 券商差异被隔离在 Broker 子类里，上层策略/选股代码完全不用改；
3. 任何真实下单前都有二次确认 + 落盘日志（可追溯）。
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd

__all__ = ["OrderSide", "OrderType", "OrderStatus", "Order", "OrderResult", "Fill",
           "Position", "AccountSnapshot", "Quote", "Broker", "BrokerError",
           "NotConnectedError", "UnsupportedError"]


class BrokerError(RuntimeError):
    """券商通道异常基类。"""


class NotConnectedError(BrokerError):
    """未连接 / 未登录。"""


class UnsupportedError(BrokerError):
    """当前通道不支持该操作。"""


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

    @property
    def cn(self) -> str:
        return "买入" if self is OrderSide.BUY else "卖出"


class OrderType(str, Enum):
    LIMIT = "limit"      # 限价（默认，A 股最常用）
    MARKET = "market"    # 市价（部分券商支持，风险较高）
    BEST5 = "best5"      # 最优五档即时成交剩余撤销


class OrderStatus(str, Enum):
    PENDING = "待报"
    SUBMITTED = "已报"
    PARTIAL = "部分成交"
    FILLED = "全部成交"
    CANCELLED = "已撤"
    REJECTED = "废单"


@dataclass
class Order:
    """一笔委托。``buy_date`` 用于实盘对齐"计划买入日期"。"""

    symbol: str
    side: OrderSide
    shares: int
    price: Optional[float] = None            # 限价；None 表示市价
    order_type: OrderType = OrderType.LIMIT
    reason: str = ""
    signal_date: Optional[pd.Timestamp] = None
    buy_date: Optional[pd.Timestamp] = None
    client_id: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.symbol = str(self.symbol).zfill(6)
        self.shares = int(self.shares)
        if isinstance(self.side, str):
            self.side = OrderSide(self.side)
        if self.order_type == OrderType.LIMIT and (self.price is None or self.price <= 0):
            raise ValueError(f"{self.symbol}: 限价委托必须给出正价格")
        if not self.client_id:
            self.client_id = f"{datetime.now():%H%M%S}{abs(hash((self.symbol, self.side, self.shares))) % 10000:04d}"

    @property
    def amount(self) -> float:
        return float(self.shares) * float(self.price or 0.0)

    def to_dict(self) -> dict:
        return {"client_id": self.client_id, "symbol": self.symbol, "side": self.side.value,
                "side_cn": self.side.cn, "shares": self.shares,
                "price": self.price, "order_type": self.order_type.value,
                "amount": round(self.amount, 2), "reason": self.reason,
                "signal_date": None if self.signal_date is None else str(pd.Timestamp(self.signal_date).date()),
                "buy_date": None if self.buy_date is None else str(pd.Timestamp(self.buy_date).date())}


@dataclass
class OrderResult:
    """委托结果（含失败原因，用于界面展示）。"""

    order: Order
    status: OrderStatus
    order_id: str = ""
    filled_shares: float = 0.0
    avg_price: float = 0.0
    fee: float = 0.0
    message: str = ""
    raw: Any = None
    ts: datetime = field(default_factory=datetime.now)

    @property
    def ok(self) -> bool:
        return self.status in (OrderStatus.SUBMITTED, OrderStatus.PARTIAL, OrderStatus.FILLED)

    def to_dict(self) -> dict:
        d = self.order.to_dict()
        d.update({"状态": self.status.value, "委托编号": self.order_id,
                  "成交数量": self.filled_shares, "成交均价": self.avg_price,
                  "费用": round(self.fee, 2), "说明": self.message,
                  "时间": self.ts.strftime("%Y-%m-%d %H:%M:%S")})
        return d


@dataclass
class Fill:
    """成交回报。"""

    symbol: str
    side: OrderSide
    shares: float
    price: float
    fee: float = 0.0
    date: Optional[pd.Timestamp] = None
    order_id: str = ""
    note: str = ""

    @property
    def amount(self) -> float:
        return float(self.shares) * float(self.price)


@dataclass
class Position:
    """持仓。``available`` 为可卖数量（T+1 后解禁）。"""

    symbol: str
    shares: float
    available: float = 0.0
    cost: float = 0.0            # 成本价
    price: float = 0.0           # 最新价
    name: str = ""

    @property
    def market_value(self) -> float:
        return float(self.shares) * float(self.price)

    @property
    def pnl(self) -> float:
        return (float(self.price) - float(self.cost)) * float(self.shares)

    @property
    def pnl_pct(self) -> float:
        return (float(self.price) / float(self.cost) - 1.0) if self.cost else 0.0


@dataclass
class AccountSnapshot:
    """账户快照。"""

    total: float = 0.0
    cash: float = 0.0
    available: float = 0.0
    market_value: float = 0.0
    frozen: float = 0.0
    positions: Dict[str, Position] = field(default_factory=dict)
    date: Optional[pd.Timestamp] = None
    broker: str = ""
    raw: Any = None

    @property
    def position_weight(self) -> float:
        return float(self.market_value) / float(self.total) if self.total else 0.0

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for s, p in self.positions.items():
            rows.append({"代码": s, "名称": p.name or s, "持仓": p.shares, "可卖": p.available,
                         "成本价": round(p.cost, 3), "现价": round(p.price, 3),
                         "市值": round(p.market_value, 2), "盈亏": round(p.pnl, 2),
                         "盈亏率": p.pnl_pct})
        return pd.DataFrame(rows)


@dataclass
class Quote:
    """实时行情快照。"""

    symbol: str
    last: float = 0.0
    prev_close: float = 0.0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    volume: float = 0.0
    amount: float = 0.0
    limit_up: Optional[float] = None
    limit_down: Optional[float] = None
    suspended: bool = False
    name: str = ""
    ts: Optional[datetime] = None
    source: str = ""

    @property
    def chg(self) -> float:
        return (self.last / self.prev_close - 1.0) if self.prev_close else 0.0

    @property
    def at_limit_up(self) -> bool:
        return bool(self.limit_up and self.last >= self.limit_up - 1e-6)

    @property
    def at_limit_down(self) -> bool:
        return bool(self.limit_down and self.last <= self.limit_down + 1e-6)


class Broker(abc.ABC):
    """券商通道抽象基类。"""

    name: str = "broker"
    display_name: str = "券商通道"
    supports_realtime_quote: bool = False
    is_real_money: bool = False          # True 表示真实资金通道

    # ---------------- 生命周期 ----------------
    @abc.abstractmethod
    def connect(self) -> bool:
        """建立连接 / 登录校验。"""

    def disconnect(self) -> None:
        pass

    @property
    def connected(self) -> bool:
        return bool(getattr(self, "_connected", False))

    # ---------------- 查询 ----------------
    @abc.abstractmethod
    def account(self) -> AccountSnapshot:
        """查询资金与持仓。"""

    def positions(self) -> Dict[str, Position]:
        return self.account().positions

    @abc.abstractmethod
    def quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        """查询行情快照。不支持实时的通道可用本地行情兜底。"""

    def orders_today(self) -> pd.DataFrame:
        """今日委托（默认空）。"""
        return pd.DataFrame(columns=["时间", "代码", "方向", "价格", "数量", "状态"])

    def fills_today(self) -> pd.DataFrame:
        """今日成交（默认空）。"""
        return pd.DataFrame(columns=["时间", "代码", "方向", "价格", "数量", "金额"])

    # ---------------- 交易 ----------------
    @abc.abstractmethod
    def place_order(self, order: Order, dry_run: bool = True) -> OrderResult:
        """下单。``dry_run=True`` 时只做校验与模拟，不真正发出委托。"""

    def cancel(self, order_id: str) -> OrderResult:
        raise UnsupportedError(f"{self.display_name} 不支持撤单")

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name} connected={self.connected}>"
