"""A 股交易规则引擎。

覆盖 A 股回测中最容易做错、也最影响收益真实性的几条硬约束：

1. **涨跌停**：主板 ±10%，创业板/科创板 ±20%，北交所 ±30%，ST/*ST ±5%；
   涨停价 = 前收盘 × (1+幅度)，按 0.01 元四舍五入。
2. **T+1**：当日买入的股票次日才能卖出（``can_sell`` 依据可卖数量判断）。
3. **整手交易**：买入必须为 100 股整数倍（科创板单笔最低 200 股，且可以 1 股递增）。
4. **交易成本**：佣金（含最低 5 元）、印花税（仅卖出，0.05%）、过户费（双边 0.001%）、滑点。
5. **停牌/一字板**：停牌日不可交易；开盘即涨停无法买入，开盘即跌停无法卖出。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math

from .config import BOARD_LIMIT, ST_LIMIT

__all__ = [
    "get_board", "board_limit", "round_price", "limit_prices", "round_lot",
    "CostModel", "TradingRules",
]


def get_board(symbol: str) -> str:
    """根据股票代码判断所属板块。

    >>> get_board("600519"), get_board("300750"), get_board("688981")
    ('MAIN', 'GEM', 'STAR')
    """
    code = "".join(ch for ch in str(symbol) if ch.isdigit())[:6]
    if code.startswith("688") or code.startswith("689"):
        return "STAR"
    if code.startswith(("300", "301", "302")):
        return "GEM"
    if code.startswith(("4", "8", "92")) or str(symbol).upper().startswith("BJ"):
        return "BSE"
    return "MAIN"


def board_limit(symbol: str, is_st: bool = False) -> float:
    """返回该股票单日涨跌幅限制（小数，如 0.1）。"""
    if is_st:
        return ST_LIMIT
    return BOARD_LIMIT.get(get_board(symbol), 0.10)


def round_price(x: float) -> float:
    """A 股报价最小变动单位为 0.01 元，四舍五入（避免二进制误差）。"""
    return float(math.floor(float(x) * 100 + 0.5) / 100.0)


def limit_prices(prev_close: float, symbol: str, is_st: bool = False,
                 uncapped: bool = False) -> Tuple[Optional[float], Optional[float]]:
    """计算涨停价与跌停价。``uncapped=True`` 表示新股上市无涨跌幅限制期。"""
    if prev_close is None or not math.isfinite(float(prev_close)) or float(prev_close) <= 0:
        return None, None
    if uncapped:
        return None, None
    pct = board_limit(symbol, is_st)
    prev = float(prev_close)
    return round_price(prev * (1 + pct)), round_price(prev * (1 - pct))


def round_lot(shares: float, lot: int = 100) -> int:
    """向下取整到整手股数。"""
    if shares is None or shares <= 0 or lot <= 0:
        return 0
    return int(shares // lot) * int(lot)


@dataclass
class CostModel:
    """交易成本模型。所有比例均以成交金额为基数。"""

    commission_rate: float = 0.00025     # 佣金万 2.5
    commission_min: float = 5.0          # 最低 5 元
    stamp_tax_rate: float = 0.0005       # 印花税（仅卖出）
    transfer_fee_rate: float = 0.00001   # 过户费（双边）
    slippage_bps: float = 5.0            # 滑点（bps，双边）

    def slippage_price(self, price: float, side: str) -> float:
        """按方向调整成交价：买入向上滑点，卖出向下滑点。"""
        adj = self.slippage_bps / 10000.0
        return price * (1 + adj) if side == "buy" else price * (1 - adj)

    def fees(self, amount: float, side: str) -> dict:
        """返回单笔交易的费用明细。``amount`` 为成交金额（价格×股数）。"""
        amount = max(float(amount), 0.0)
        commission = max(amount * self.commission_rate, self.commission_min) if amount > 0 else 0.0
        commission = min(commission, amount)  # 极端情况下费用不超过成交额
        stamp = amount * self.stamp_tax_rate if side == "sell" else 0.0
        transfer = amount * self.transfer_fee_rate
        total = commission + stamp + transfer
        return {
            "commission": commission,
            "stamp_tax": stamp,
            "transfer_fee": transfer,
            "total_fee": total,
        }

    @property
    def buy_rate(self) -> float:
        return self.commission_rate + self.transfer_fee_rate + self.slippage_bps / 10000.0

    @property
    def sell_rate(self) -> float:
        return (self.commission_rate + self.stamp_tax_rate
                + self.transfer_fee_rate + self.slippage_bps / 10000.0)


@dataclass
class TradingRules:
    """市场制度开关。回测时用于模拟真实约束，实盘信号时用于生成可执行价格。"""

    t_plus_1: bool = True
    enforce_price_limit: bool = True
    enforce_suspension: bool = True
    lot_size: int = 100
    star_min_lot: int = 200               # 科创板单笔最低 200 股
    allow_odd_lot_sell: bool = True       # 允许零股卖出（持仓不足 100 股时）

    # ---------- 可交易性判断 ----------
    def can_buy(self, open_price: Optional[float], limit_up: Optional[float],
                suspended: bool = False, volume: float = 1.0) -> Tuple[bool, str]:
        if suspended or not volume:
            return False, "停牌"
        if open_price is None or open_price <= 0 or not math.isfinite(float(open_price)):
            return False, "无有效报价"
        if self.enforce_price_limit and limit_up and open_price >= limit_up - 1e-9:
            return False, "开盘涨停无法买入"
        return True, ""

    def can_sell(self, open_price: Optional[float], limit_down: Optional[float],
                 suspended: bool = False, volume: float = 1.0, available: float = 0.0) -> Tuple[bool, str]:
        if suspended or not volume:
            return False, "停牌"
        if open_price is None or open_price <= 0 or not math.isfinite(float(open_price)):
            return False, "无有效报价"
        if available <= 0:
            return False, "T+1 未解禁/无可卖数量"
        if self.enforce_price_limit and limit_down and open_price <= limit_down + 1e-9:
            return False, "开盘跌停无法卖出"
        return True, ""

    def tradable_qty(self, desired_qty: float, side: str, symbol: str,
                     price: Optional[float] = None, cash: float = math.inf) -> int:
        """将期望股数调整为合规的委托股数（整手 / 科创板 / 资金约束）。"""
        desired_qty = float(desired_qty)
        if desired_qty <= 0:
            return 0
        lot = self.lot_size
        if get_board(symbol) == "STAR" and side == "buy":
            # 科创板：单笔申报 ≥ 200 股，超过部分可 1 股递增
            if desired_qty < self.star_min_lot:
                return 0
            qty = int(desired_qty)
        else:
            qty = round_lot(desired_qty, lot)
        if side == "sell" and self.allow_odd_lot_sell:
            qty = int(desired_qty)  # 零股可一次性卖出
        if side == "buy" and price and math.isfinite(float(price)) and price > 0:
            max_qty = round_lot(cash / float(price), lot)
            qty = min(qty, max_qty)
        return max(int(qty), 0)