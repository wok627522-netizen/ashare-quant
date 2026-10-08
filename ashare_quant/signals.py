"""实盘/模拟盘信号生成。

把策略的最新目标权重，翻译成"明天开盘该做什么"的可执行清单：

* 每只股票的建议动作（新建仓 / 加仓 / 减仓 / 清仓 / 持有）
* 建议股数（自动整手、科创板 200 股起）
* 参考价、调仓金额、目标仓位、涨跌停价与风险提示
* 支持本地 JSON 模拟盘账户，按信号记账并跟踪净值

⚠️ 本模块只做信号与模拟记账，**不接入券商下单接口**。A 股程序化交易有合规要求，
实盘请走券商官方的 API/量化终端，并自行确认合规性与风险。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .data import DataBundle
from .rules import board_limit, get_board, round_lot

__all__ = ["latest_signals", "export_signals", "PaperAccount", "signal_summary_text"]

ACTION_ORDER = ["清仓", "减仓", "新建仓", "加仓", "持有", "观望"]


def latest_signals(data: DataBundle, strategy, capital: float = 1_000_000.0,
                   current_holdings: Optional[Dict[str, float]] = None,
                   band: float = 0.005, lot: int = 100) -> pd.DataFrame:
    """生成最新一期的调仓信号表。

    Parameters
    ----------
    current_holdings : dict, optional
        ``{股票代码: 持股数量}``，来自模拟盘或实盘账户；不传则按上一期目标权重推算。
    band : float
        权重变化小于该阈值视为"持有"，避免噪声交易。
    """
    w = strategy.weight_matrix(data)
    if w.empty:
        return pd.DataFrame()
    w = w.dropna(how="all")
    last_date = w.index[-1]
    target = w.loc[last_date].fillna(0.0)
    prev = w.loc[w.index[-2]].fillna(0.0) if len(w) > 1 else pd.Series(0.0, index=w.columns)

    last_close = data.close_matrix().loc[:last_date].ffill().iloc[-1]
    rows: List[dict] = []
    for sym in w.columns:
        price = float(last_close.get(sym, np.nan) or np.nan)
        tgt_w = float(target.get(sym, 0.0))
        prev_w = float(prev.get(sym, 0.0))
        if current_holdings is not None and sym in current_holdings and price == price:
            cur_w = float(current_holdings[sym]) * price / max(capital, 1e-9)
        else:
            cur_w = prev_w
            if current_holdings is not None:
                cur_w = 0.0
        delta_w = tgt_w - cur_w
        if price != price or price <= 0:
            action, shares, amount, note = "观望", 0, 0.0, "无有效行情"
        elif tgt_w <= 1e-9 and cur_w <= band:
            action, shares, amount, note = "观望", 0, 0.0, "策略空仓"
        elif abs(delta_w) <= band:
            action, amount, note = "持有", 0.0, "权重变动极小"
            shares = int(round_lot(cur_w * capital / price, lot)) if cur_w > 0 else 0
        elif delta_w > 0:
            action = "新建仓" if cur_w <= band else "加仓"
            amount = delta_w * capital
            shares = int(round_lot(amount / price, lot))
            note = ""
        else:
            action = "清仓" if tgt_w <= band else "减仓"
            amount = -delta_w * capital
            shares = int(round_lot(amount / price, lot))
            if cur_w > 0 and tgt_w <= band:
                shares = int(round_lot(cur_w * capital / price, lot))
            note = ""

        st = data.is_st(sym)
        limit = board_limit(sym, st)
        prev_close = price
        up, dn = round(prev_close * (1 + limit), 2), round(prev_close * (1 - limit), 2)
        susp = bool(data.suspended_matrix()[sym].iloc[-1]) if sym in data.prices else True
        if susp:
            note = (note + "；当前停牌").strip("；")
        if action in ("新建仓", "加仓") and not susp:
            note = (note + "；注意次日开盘涨幅过高/一字涨停可能无法买入").strip("；")
        if action in ("减仓", "清仓") and st:
            note = (note + "；ST 股票涨跌幅 ±5%").strip("；")
        rows.append({
            "symbol": sym, "name": data.name_of(sym), "industry": data.industry_of(sym),
            "board": get_board(sym), "is_st": st, "date": last_date,
            "close": round(price, 3) if price == price else np.nan,
            "limit_up": up if price == price else np.nan,
            "limit_down": dn if price == price else np.nan,
            "cur_weight": cur_w, "target_weight": tgt_w, "delta_weight": delta_w,
            "amount": round(amount, 2), "shares": shares, "action": action,
            "note": note,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    order = {a: i for i, a in enumerate(ACTION_ORDER)}
    df["_o"] = df["action"].map(order).fillna(99)
    df = df.sort_values(["_o", "delta_weight"], ascending=[True, False]).drop(columns="_o")
    return df.reset_index(drop=True)


def signal_summary_text(signals: pd.DataFrame, top: int = 10) -> str:
    """把信号表转成一段中文摘要（可直接贴到聊天/邮件）。"""
    if signals is None or signals.empty:
        return "无可用信号"
    d = pd.Timestamp(signals["date"].iloc[0]).date()
    buys = signals[signals["action"].isin(["新建仓", "加仓"])]
    sells = signals[signals["action"].isin(["减仓", "清仓"])]
    lines = [f"【量化信号】{d}", f"买入/加仓 {len(buys)} 只，卖出/减仓 {len(sells)} 只"]
    for _, r in buys.head(top).iterrows():
        lines.append(f"  ▲ {r['name']}({r['symbol']}) 目标仓位 {r['target_weight']:.1%}，"
                     f"约 {int(r['shares'])} 股 @ {r['close']}")
    for _, r in sells.head(top).iterrows():
        lines.append(f"  ▼ {r['name']}({r['symbol']}) 目标仓位 {r['target_weight']:.1%}，"
                     f"约 {int(r['shares'])} 股 @ {r['close']}")
    lines.append("（仅供研究参考，不构成投资建议）")
    return "\n".join(lines)


def export_signals(signals: pd.DataFrame, path: Union[str, Path]) -> Path:
    """导出信号 CSV（utf-8-sig，Excel 可直接打开）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    signals.to_csv(p, index=False, encoding="utf-8-sig")
    return p


# --------------------------------------------------------------------------- #
# 模拟盘账户
# --------------------------------------------------------------------------- #
@dataclass
class PaperAccount:
    """极简模拟盘账户（JSON 持久化），用于按信号跟踪组合表现。"""

    initial_cash: float = 1_000_000.0
    cash: float = 0.0
    positions: Dict[str, dict] = field(default_factory=dict)   # {symbol: {shares, cost}}
    history: List[dict] = field(default_factory=list)
    path: Optional[str] = None
    created: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def __post_init__(self) -> None:
        if not self.cash:
            self.cash = float(self.initial_cash)

    # ---------------- 持久化 ----------------
    @classmethod
    def load(cls, path: Union[str, Path], initial_cash: float = 1_000_000.0) -> "PaperAccount":
        p = Path(path)
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            return cls(initial_cash=data.get("initial_cash", initial_cash),
                       cash=data.get("cash", initial_cash),
                       positions=data.get("positions", {}),
                       history=data.get("history", []), path=str(p),
                       created=data.get("created", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        acct = cls(initial_cash=initial_cash, path=str(p))
        acct.save()
        return acct

    def save(self) -> str:
        if not self.path:
            self.path = "paper_account.json"
        payload = {"initial_cash": self.initial_cash, "cash": self.cash,
                   "positions": self.positions, "history": self.history,
                   "created": self.created, "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        Path(self.path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return self.path

    # ---------------- 估值 ----------------
    def market_value(self, prices: Dict[str, float]) -> float:
        mv = 0.0
        for sym, pos in self.positions.items():
            px = float(prices.get(sym, pos.get("cost", 0.0)) or 0.0)
            mv += float(pos.get("shares", 0)) * px
        return mv

    def total_equity(self, prices: Dict[str, float]) -> float:
        return self.cash + self.market_value(prices)

    def holding_table(self, prices: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        prices = prices or {}
        rows = []
        for sym, pos in self.positions.items():
            shares = float(pos.get("shares", 0))
            cost = float(pos.get("cost", 0.0))
            px = float(prices.get(sym, cost) or cost)
            mv = shares * px
            rows.append({"symbol": sym, "shares": shares, "cost": cost, "price": px,
                         "market_value": mv, "pnl": (px - cost) * shares,
                         "pnl_pct": (px / cost - 1.0) if cost > 0 else 0.0,
                         "weight": np.nan})
        df = pd.DataFrame(rows)
        if not df.empty and df["market_value"].sum() > 0:
            total = self.cash + df["market_value"].sum()
            df["weight"] = df["market_value"] / total
        return df

    # ---------------- 交易 ----------------
    def apply_signals(self, signals: pd.DataFrame, prices: Dict[str, float],
                      date: Optional[str] = None, cost_rate: float = 0.00035) -> pd.DataFrame:
        """按信号表在给定价格上模拟成交（含简化成本）。"""
        fills = []
        d = date or datetime.now().strftime("%Y-%m-%d")
        if signals is None or signals.empty:
            return pd.DataFrame(fills)
        for _, r in signals.iterrows():
            sym, action, shares = str(r["symbol"]), str(r["action"]), int(r.get("shares", 0))
            if shares <= 0 or action in ("持有", "观望"):
                continue
            px = float(prices.get(sym, r.get("close", 0.0)) or 0.0)
            if px <= 0:
                continue
            if action in ("新建仓", "加仓"):
                need = shares * px * (1 + cost_rate)
                if need > self.cash:
                    shares = int(round_lot(self.cash / (px * (1 + cost_rate)), 100))
                    need = shares * px * (1 + cost_rate)
                if shares <= 0:
                    continue
                self.cash -= need
                pos = self.positions.get(sym, {"shares": 0.0, "cost": 0.0})
                old_sh, old_cost = float(pos["shares"]), float(pos["cost"])
                new_sh = old_sh + shares
                avg = (old_sh * old_cost + shares * px) / new_sh if new_sh > 0 else px
                self.positions[sym] = {"shares": new_sh, "cost": round(avg, 4)}
                fills.append({"date": d, "symbol": sym, "side": "buy", "shares": shares,
                              "price": px, "amount": round(shares * px, 2)})
            else:
                pos = self.positions.get(sym)
                if not pos:
                    continue
                sell_sh = min(shares, float(pos["shares"]))
                if sell_sh <= 0:
                    continue
                self.cash += sell_sh * px * (1 - cost_rate)
                remain = float(pos["shares"]) - sell_sh
                if remain <= 1e-6:
                    self.positions.pop(sym, None)
                else:
                    self.positions[sym] = {"shares": remain, "cost": pos["cost"]}
                fills.append({"date": d, "symbol": sym, "side": "sell", "shares": sell_sh,
                              "price": px, "amount": round(sell_sh * px, 2)})
        equity = self.total_equity(prices)
        self.history.append({"date": d, "equity": equity, "cash": self.cash,
                             "market_value": self.market_value(prices),
                             "n_holdings": len(self.positions)})
        self.save()
        return pd.DataFrame(fills)

    def summary(self, prices: Dict[str, float]) -> pd.DataFrame:
        eq = self.total_equity(prices)
        mv = self.market_value(prices)
        return pd.DataFrame([
            ("初始资金", f"{self.initial_cash:,.0f}"),
            ("总资产", f"{eq:,.0f}"),
            ("可用现金", f"{self.cash:,.0f}"),
            ("持仓市值", f"{mv:,.0f}"),
            ("仓位", f"{mv / eq:.1%}" if eq > 0 else "0%"),
            ("累计收益", f"{eq / self.initial_cash - 1:.2%}"),
            ("持仓数量", f"{len(self.positions)}"),
        ], columns=["项目", "数值"])