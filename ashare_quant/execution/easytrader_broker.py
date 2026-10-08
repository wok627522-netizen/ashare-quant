r"""easytrader 通道适配器（**真实资金**，客户端 GUI 自动化）。

原理：用 ``easytrader`` 库驱动本机已登录的券商/行情客户端（同花顺、通达信等）下单。
优点是没有资金门槛、多数账户可用；缺点是**依赖客户端界面**，客户端升级/换皮肤可能失效，
且不如下单接口稳定 —— 请先用小资金验证。

前置条件
--------
1. 本机安装并登录 同花顺（或通达信、华泰等）客户端，确保能手动下单；
2. ``pip install easytrader``；
3. 首次使用建议用配置文件方式：``user.prepare("config.json")``（easytrader 官方文档）。

用法
----
    broker = EasyTraderBroker(client="ths", client_path=r"C:\\同花顺\\xiadan.exe")
    broker.connect()
    broker.place_order(Order("600519", OrderSide.BUY, 100, price=1700.0), dry_run=False)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from ..rules import board_limit
from .base import (AccountSnapshot, Broker, BrokerError, NotConnectedError, Order,
                   OrderResult, OrderSide, OrderStatus, OrderType, Position, Quote)

__all__ = ["EasyTraderBroker"]


class EasyTraderBroker(Broker):
    name = "easytrader"
    display_name = "easytrader（客户端自动化，实盘）"
    supports_realtime_quote = False
    is_real_money = True

    def __init__(self, client: str = "ths", client_path: Optional[str] = None,
                 config_path: Optional[str] = None, use_config: bool = False,
                 retry: int = 1, **kwargs: Any):
        self.client = client
        self.client_path = client_path
        self.config_path = config_path
        self.use_config = use_config
        self.retry = int(retry)
        self.kwargs = kwargs
        self._user = None
        self._connected = False
        self._last_error = ""
        self._quotes: Dict[str, Quote] = {}
        self._orders: List[dict] = []

    # ---------------- 连接 ----------------
    def connect(self) -> bool:
        try:
            import easytrader
        except ImportError as exc:
            raise BrokerError("未安装 easytrader：pip install easytrader") from exc
        try:
            user = easytrader.use(self.client)
            if self.use_config and self.config_path:
                user.prepare(self.config_path)
            elif self.client_path:
                user.connect(self.client_path)
            else:
                user.prepare(self.config_path or "config.json")
            self._user = user
            self._connected = True
            # 触发一次查询，确认客户端真的可用
            _ = self.account()
            return True
        except Exception as exc:
            self._connected = False
            raise BrokerError(f"easytrader 连接失败（请确认客户端已登录并可手动下单）：{exc}") from exc

    def _require(self):
        if not self._connected or self._user is None:
            raise NotConnectedError("easytrader 未连接，请先 connect()")

    # ---------------- 查询 ----------------
    @staticmethod
    def _pick(d: Dict[str, Any], *keys, default: float = 0.0) -> float:
        for k in keys:
            if k in d and d[k] not in (None, "", "--"):
                try:
                    return float(str(d[k]).replace(",", ""))
                except Exception:
                    continue
        return default

    def account(self) -> AccountSnapshot:
        self._require()
        bal = self._user.balance or {}
        if isinstance(bal, list):
            bal = bal[0] if bal else {}
        total = self._pick(bal, "总资产", "资产", "总资产值", default=0.0)
        cash = self._pick(bal, "可用金额", "可用余额", "可用资金", default=0.0)
        frozen = self._pick(bal, "冻结金额", "冻结资金", default=0.0)
        positions: Dict[str, Position] = {}
        mv = 0.0
        try:
            for p in (self._user.position or []):
                sym = str(p.get("证券代码") or p.get("股票代码") or "").zfill(6)
                if not sym or sym == "000000":
                    continue
                shares = self._pick(p, "股票余额", "证券数量", "当前持仓")
                if shares <= 0:
                    continue
                price = self._pick(p, "市价", "参考市价", "最新价")
                pos = Position(symbol=sym, shares=shares,
                               available=self._pick(p, "可用余额", "可卖数量", "可用股份"),
                               cost=self._pick(p, "成本价", "参考成本价"),
                               price=price, name=str(p.get("证券名称") or ""))
                positions[sym] = pos
                mv += pos.market_value
        except Exception as exc:
            self._last_error = f"持仓查询失败：{exc}"
        if total <= 0:
            total = cash + mv
        return AccountSnapshot(total=total, cash=cash, available=cash, market_value=mv,
                               frozen=frozen, positions=positions, broker=self.display_name)

    def set_quotes(self, quotes: Dict[str, Quote]) -> None:
        """easytrader 不提供行情，由上层注入（AkShare 快照 / 本地行情）。"""
        self._quotes.update(quotes)

    def quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        return {s: self._quotes[s] for s in symbols if s in self._quotes}

    def orders_today(self) -> pd.DataFrame:
        self._require()
        try:
            data = self._user.today_entrusts or []
        except Exception:
            return pd.DataFrame()
        rows = []
        for d in data:
            rows.append({"时间": str(d.get("委托时间", "")),
                         "代码": str(d.get("证券代码", "")).zfill(6),
                         "方向": str(d.get("买卖标志", "") or d.get("操作", "")),
                         "价格": self._pick(d, "委托价格"),
                         "数量": self._pick(d, "委托数量"),
                         "状态": str(d.get("状态说明", "") or d.get("备注", ""))})
        return pd.DataFrame(rows)

    def fills_today(self) -> pd.DataFrame:
        self._require()
        try:
            data = self._user.today_trades or []
        except Exception:
            return pd.DataFrame()
        return pd.DataFrame(data) if data else pd.DataFrame()

    # ---------------- 交易 ----------------
    def place_order(self, order: Order, dry_run: bool = True) -> OrderResult:
        self._require()
        if dry_run:
            return OrderResult(order, OrderStatus.SUBMITTED, order_id=f"DRY{datetime.now():%H%M%S%f}",
                               message="dry-run：未向客户端发送委托", avg_price=float(order.price or 0))
        if order.order_type == OrderType.MARKET:
            return OrderResult(order, OrderStatus.REJECTED,
                               message="easytrader 通道仅支持限价委托，请改用限价单")
        price = float(order.price or 0.0)
        last_err = ""
        for attempt in range(max(1, self.retry + 1)):
            try:
                if order.side is OrderSide.BUY:
                    res = self._user.buy(order.symbol, price=price, amount=int(order.shares))
                else:
                    res = self._user.sell(order.symbol, price=price, amount=int(order.shares))
                oid = ""
                if isinstance(res, dict):
                    oid = str(res.get("entrust_no") or res.get("委托编号") or "")
                    if res.get("error") or res.get("msg"):
                        last_err = str(res.get("error") or res.get("msg"))
                        continue
                self._orders.append({"ts": datetime.now().isoformat(), "order_id": oid,
                                     **order.to_dict()})
                return OrderResult(order, OrderStatus.SUBMITTED, order_id=oid,
                                   message="已通过客户端提交委托", avg_price=price, raw=res)
            except Exception as exc:
                last_err = str(exc)
        return OrderResult(order, OrderStatus.REJECTED, message=f"easytrader 下单失败：{last_err}")

    def cancel(self, order_id: str) -> OrderResult:
        self._require()
        dummy = Order(symbol="000000", side=OrderSide.BUY, shares=0, price=0.01)
        try:
            self._user.cancel_entrust(order_id)
            return OrderResult(dummy, OrderStatus.CANCELLED, order_id=str(order_id), message="已撤单")
        except Exception as exc:
            return OrderResult(dummy, OrderStatus.REJECTED, order_id=str(order_id),
                               message=f"撤单失败：{exc}")

    def diagnostics(self) -> Dict[str, Any]:
        info = {"通道": self.display_name, "客户端": self.client, "客户端路径": self.client_path,
                "已连接": self._connected, "最近错误": self._last_error}
        if self._connected:
            try:
                a = self.account()
                info.update({"总资产": a.total, "可用资金": a.available, "持仓数": len(a.positions)})
            except Exception as exc:
                info["查询错误"] = str(exc)
        return info

