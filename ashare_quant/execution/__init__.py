"""执行层入口：券商通道工厂 + 风险闸门导出。"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from .base import (AccountSnapshot, Broker, BrokerError, Fill, NotConnectedError, Order,
                   OrderResult, OrderSide, OrderStatus, OrderType, Position, Quote,
                   UnsupportedError)
from .risk_gate import RiskDecision, RiskGate, RiskLimits, is_trading_time, session_name
from .sim_broker import SimBroker
from .manual_broker import ManualBroker

__all__ = [
    "Broker", "Order", "OrderSide", "OrderType", "OrderStatus", "OrderResult", "Fill",
    "Position", "AccountSnapshot", "Quote", "BrokerError", "NotConnectedError", "UnsupportedError",
    "RiskGate", "RiskLimits", "RiskDecision", "is_trading_time", "session_name",
    "SimBroker", "ManualBroker", "QMTBroker", "EasyTraderBroker",
    "create_broker", "available_brokers", "BROKER_TABLE",
]

BROKER_TABLE = [
    {"key": "sim", "名称": "本地模拟盘", "真实资金": "否", "实时行情": "外部注入",
     "前置条件": "无，开箱可用", "适用": "策略演练 / 流程验证"},
    {"key": "manual", "名称": "手动执行（导出委托单）", "真实资金": "是（人工下单）", "实时行情": "外部注入",
     "前置条件": "任意券商 App", "适用": "最稳妥的实盘方式，无接口依赖"},
    {"key": "qmt", "名称": "迅投 QMT（xtquant）", "真实资金": "是", "实时行情": "内置",
     "前置条件": "券商开通 QMT + 客户端登录 + xtquant", "适用": "全自动实盘（主流选择）"},
    {"key": "easytrader", "名称": "easytrader 客户端自动化", "真实资金": "是", "实时行情": "外部注入",
     "前置条件": "本机客户端登录 + pip install easytrader", "适用": "无资金门槛的半自动实盘"},
]


def available_brokers() -> pd.DataFrame:
    """可用通道列表（界面展示用）。"""
    return pd.DataFrame(BROKER_TABLE)


def create_broker(name: str, **kwargs: Any) -> Broker:
    """按名称创建通道实例。

    >>> create_broker("sim", initial_cash=500000).display_name
    '本地模拟盘'
    """
    key = str(name or "").strip().lower()
    if key in ("sim", "simulation", "模拟", "模拟盘"):
        return SimBroker(**kwargs)
    if key in ("manual", "手动", "人工"):
        return ManualBroker(**kwargs)
    if key in ("qmt", "xtquant", "迅投"):
        from .qmt_broker import QMTBroker
        return QMTBroker(**kwargs)
    if key in ("easytrader", "et", "客户端"):
        from .easytrader_broker import EasyTraderBroker
        return EasyTraderBroker(**kwargs)
    raise KeyError(f"未知通道：{name}；可选：sim / manual / qmt / easytrader")
