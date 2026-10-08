"""事件驱动回测引擎（A 股实盘约束版）。

执行时序（关键，杜绝未来函数）
------------------------------
1. 第 T 日 **收盘后**，用截至 T 日收盘的数据计算目标权重；
2. 第 T+1 日 **开盘**按目标权重下单（含滑点）；
3. 买卖受涨跌停、停牌、整手、最低佣金、印花税、过户费、T+1 解禁约束；
4. 收盘按当日收盘价做市；触发止损 / 回撤熔断则在下一交易日开盘减仓。

支持的 A 股特征
---------------
* T+1：当日买入次日方可卖出（按交易日记解禁）
* 涨跌停：主板/创业板/科创板/北交所/ST 分别校验，开盘一字板无法成交
* 停牌：成交量 0 或 suspended 标记为停牌，不可交易
* 整手：买入 100 股整数倍（科创板 200 股起）
* 成本：佣金（万 2.5，最低 5 元）＋印花税 0.05%（卖出）＋过户费 0.001%＋滑点
* 风控：ATR 止损、移动止损、单票/行业权重上限、最大回撤熔断
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import indicators as ind
from .config import RISK_FREE_RATE, TRADE_DAYS_PER_YEAR
from .data import DataBundle
from .metrics import (METRIC_LABELS, drawdown_series, monthly_return_table,
                      performance_summary, round_trip_trades, format_metric)
from .rules import CostModel, TradingRules, round_lot

__all__ = ["BacktestConfig", "BacktestResult", "Backtester", "compare_strategies"]


@dataclass
class BacktestConfig:
    """回测配置。"""

    initial_cash: float = 1_000_000.0
    cost: CostModel = field(default_factory=CostModel)
    rules: TradingRules = field(default_factory=TradingRules)
    max_position_weight: float = 0.30        # 单票权重上限
    max_industry_weight: float = 0.50        # 单行业权重上限
    rebalance_band: float = 0.01             # 权重偏离容忍度（减少无效调仓）
    min_trade_amount: float = 2000.0         # 最小成交金额，过滤碎单
    stop_loss_atr: float = 0.0               # ATR 止损倍数，0=关闭
    atr_window: int = 14
    trailing_stop: float = 0.0               # 移动止损（如 0.15 表示回撤 15% 离场）
    max_drawdown_stop: float = 0.0           # 组合最大回撤熔断线，0=关闭
    circuit_breaker_cooldown: int = 20       # 熔断后冷却交易日
    risk_free_rate: float = RISK_FREE_RATE
    periods: int = TRADE_DAYS_PER_YEAR
    lot_size: int = 100

    @classmethod
    def from_defaults(cls, **kw) -> "BacktestConfig":
        from .config import BacktestDefaults
        d = BacktestDefaults()
        cfg = cls(initial_cash=d.initial_cash,
                  cost=CostModel(d.commission_rate, d.commission_min, d.stamp_tax_rate,
                                 d.transfer_fee_rate, d.slippage_bps),
                  rules=TradingRules(t_plus_1=d.t_plus_1, lot_size=d.lot_size),
                  max_position_weight=d.max_position_weight,
                  max_industry_weight=d.max_industry_weight,
                  stop_loss_atr=d.stop_loss_atr, trailing_stop=d.trailing_stop,
                  max_drawdown_stop=d.max_drawdown_stop)
        for k, v in kw.items():
            if v is not None and hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

    def describe(self) -> pd.DataFrame:
        rows = [
            ("初始资金", f"{self.initial_cash:,.0f} 元"),
            ("佣金率", f"{self.cost.commission_rate * 10000:.2f} bp（最低 {self.cost.commission_min:.0f} 元）"),
            ("印花税", f"{self.cost.stamp_tax_rate * 1000:.2f}‰（卖出）"),
            ("过户费", f"{self.cost.transfer_fee_rate * 1000:.4f}‰（双边）"),
            ("滑点", f"{self.cost.slippage_bps:.1f} bp"),
            ("整手", f"{self.rules.lot_size} 股"),
            ("T+1", "已启用" if self.rules.t_plus_1 else "关闭"),
            ("涨跌停约束", "已启用" if self.rules.enforce_price_limit else "关闭"),
            ("停牌约束", "已启用" if self.rules.enforce_suspension else "关闭"),
            ("单票上限", f"{self.max_position_weight * 100:.0f}%"),
            ("行业上限", f"{self.max_industry_weight * 100:.0f}%"),
            ("ATR 止损", f"{self.stop_loss_atr:.1f} 倍" if self.stop_loss_atr else "关闭"),
            ("移动止损", f"{self.trailing_stop * 100:.0f}%" if self.trailing_stop else "关闭"),
            ("回撤熔断", f"{self.max_drawdown_stop * 100:.0f}%" if self.max_drawdown_stop else "关闭"),
        ]
        return pd.DataFrame(rows, columns=["参数", "取值"])


@dataclass
class BacktestResult:
    """回测结果容器。"""

    name: str
    equity: pd.Series
    benchmark: Optional[pd.Series]
    daily: pd.DataFrame
    weights: pd.DataFrame
    holdings: pd.DataFrame
    trades: pd.DataFrame
    orders: pd.DataFrame
    metrics: Dict[str, float]
    round_trips: pd.DataFrame
    config: Dict[str, object] = field(default_factory=dict)
    events: pd.DataFrame = field(default_factory=pd.DataFrame)

    # ---------- 展示辅助 ----------
    def stats_table(self, keys: Optional[Sequence[str]] = None) -> pd.DataFrame:
        keys = list(keys) if keys else list(METRIC_LABELS.keys())
        rows = [{"指标": METRIC_LABELS.get(k, k), "数值": format_metric(k, self.metrics.get(k, 0.0))}
                for k in keys if k in self.metrics]
        return pd.DataFrame(rows)

    def yearly_returns(self) -> pd.DataFrame:
        e = self.equity.dropna()
        if e.empty:
            return pd.DataFrame()
        per = e.index.to_period("Y")
        last = e.groupby(per).last()
        ret = last.pct_change()
        ret.iloc[0] = last.iloc[0] / e.iloc[0] - 1.0
        bench = None
        if self.benchmark is not None and len(self.benchmark.dropna()) > 1:
            b = self.benchmark.dropna().reindex(e.index).ffill().dropna()
            bl = b.groupby(b.index.to_period("Y")).last()
            br = bl.pct_change()
            br.iloc[0] = bl.iloc[0] / b.iloc[0] - 1.0
            bench = br
        out = pd.DataFrame({"年份": [p.year for p in last.index], "策略收益": ret.values})
        if bench is not None:
            out["基准收益"] = bench.reindex(last.index).values
            out["超额"] = out["策略收益"] - out["基准收益"]
        return out

    def monthly_table(self) -> pd.DataFrame:
        return monthly_return_table(self.equity)

    def drawdown(self) -> pd.Series:
        return drawdown_series(self.equity)

    def to_excel(self, path) -> str:
        """导出多表 Excel（中文表头，需要 openpyxl）。"""
        from .labels import to_cn, to_cn_excel
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            to_cn_excel(writer, self)
            if len(self.events):
                to_cn(self.events).to_excel(writer, sheet_name="风控事件", index=False)
        return str(path)


class Backtester:
    """事件驱动回测器。"""

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()

    # ------------------------------------------------------------------ #
    # 权重约束
    # ------------------------------------------------------------------ #
    def apply_constraints(self, weights: pd.DataFrame, data: DataBundle) -> pd.DataFrame:
        """单票上限 + 行业上限 + 归一化（迭代削峰，避免破坏总仓位）。"""
        cfg = self.config
        w = weights.reindex(columns=data.symbols).fillna(0.0).clip(lower=0.0)
        total = w.sum(axis=1)
        w = w.clip(upper=cfg.max_position_weight)
        if cfg.max_industry_weight and cfg.max_industry_weight > 0:
            industry = {}
            for s in w.columns:
                industry.setdefault(data.industry_of(s), []).append(s)
            for _ in range(3):
                breach = False
                for _, cols in industry.items():
                    col_sum = w[cols].sum(axis=1)
                    over = col_sum > cfg.max_industry_weight
                    if over.any():
                        breach = True
                        scale = (cfg.max_industry_weight / col_sum[over]).clip(upper=1.0)
                        w.loc[over, cols] = w.loc[over, cols].mul(scale, axis=0)
                if not breach:
                    break
        # 目标仓位不超过原始总仓位（行业/单票约束只减不增）
        new_total = w.sum(axis=1)
        scale = (total / new_total.replace(0.0, np.nan)).clip(upper=1.0).fillna(0.0)
        return w.mul(scale, axis=0).fillna(0.0)

    # ------------------------------------------------------------------ #
    # 主流程
    # ------------------------------------------------------------------ #
    def run(self, data: DataBundle, target_weights: pd.DataFrame,
            name: str = "策略", progress_cb=None) -> BacktestResult:
        cfg = self.config
        rules = cfg.rules
        cost = cfg.cost

        dates = pd.DatetimeIndex(target_weights.index.intersection(data.calendar))
        if len(dates) < 20:
            raise ValueError("回测区间过短（<20 个交易日），请扩大时间范围")
        symbols = list(data.symbols)
        tw = self.apply_constraints(target_weights.reindex(dates).fillna(0.0), data)

        open_m = data.open_matrix().reindex(dates)
        close_m = data.close_matrix().reindex(dates)
        high_m, low_m = data.high_matrix().reindex(dates), data.low_matrix().reindex(dates)
        vol_m = data.volume_matrix().reindex(dates)
        susp_m = data.suspended_matrix().reindex(dates)
        up_m = pd.DataFrame({s: data.prices[s]["limit_up"] for s in symbols}).reindex(dates)
        dn_m = pd.DataFrame({s: data.prices[s]["limit_down"] for s in symbols}).reindex(dates)

        if cfg.stop_loss_atr and cfg.stop_loss_atr > 0:
            atr_m = pd.DataFrame({s: ind.atr(data.prices[s]["high"], data.prices[s]["low"],
                                             data.prices[s]["close"], cfg.atr_window) for s in symbols}
                                 ).reindex(dates)
        else:
            atr_m = pd.DataFrame(np.nan, index=dates, columns=symbols)

        cash = float(cfg.initial_cash)
        qty: Dict[str, float] = {s: 0.0 for s in symbols}
        frozen: Dict[str, List[Tuple[pd.Timestamp, float]]] = {s: [] for s in symbols}
        entry_px: Dict[str, float] = {}
        peak_px: Dict[str, float] = {}
        stopped: Dict[str, bool] = {s: False for s in symbols}
        halt_until: Optional[pd.Timestamp] = None
        peak_equity = float(cfg.initial_cash)
        pending: Optional[pd.Series] = None

        rows: List[dict] = []
        trade_rows: List[dict] = []
        order_rows: List[dict] = []
        weight_rows: List[dict] = []
        event_rows: List[dict] = []

        def next_session(i: int) -> pd.Timestamp:
            return dates[i + 1] if i + 1 < len(dates) else dates[-1] + pd.Timedelta(days=1)

        def frozen_shares(sym: str, day: pd.Timestamp) -> float:
            return float(sum(lt[1] for lt in frozen[sym] if lt[0] > day))

        def raw_open(sym: str, day: pd.Timestamp) -> Optional[float]:
            v = open_m.at[day, sym]
            if pd.isna(v) or v <= 0:
                v = close_m.at[day, sym]
            if pd.isna(v) or v <= 0:
                return None
            return float(v)

        def mark_price(sym: str, day: pd.Timestamp) -> float:
            v = raw_open(sym, day)
            return float(v) if v else 0.0

        def equity_at(day: pd.Timestamp) -> float:
            mv = sum(qty[s] * (close_m.at[day, s] if pd.notna(close_m.at[day, s]) else 0.0)
                     for s in symbols)
            return cash + float(mv)

        def record_trade(day, sym, side, price, shares, fee, note=""):
            amount = price * shares
            return {"date": day, "symbol": sym, "name": data.name_of(sym), "side": side,
                    "price": round(price, 3), "shares": shares, "amount": round(amount, 2),
                    "commission": round(fee["commission"], 2), "stamp_tax": round(fee["stamp_tax"], 2),
                    "transfer_fee": round(fee["transfer_fee"], 2), "total_fee": round(fee["total_fee"], 2),
                    "note": note}

        def execute(day: pd.Timestamp, i: int, targets: pd.Series) -> None:
            """在第 i 个交易日的开盘按目标权重调仓。"""
            nonlocal cash
            op = {s: mark_price(s, day) for s in symbols}
            eq = cash + sum(qty[s] * op[s] for s in symbols)
            if eq <= 0:
                return
            desired: Dict[str, float] = {}
            for s in symbols:
                w = float(targets.get(s, 0.0) or 0.0)
                desired[s] = (w * eq / op[s]) if op[s] > 0 else 0.0

            # ---------- 1) 卖出（先释放现金） ----------
            for s in symbols:
                cur = qty[s]
                delta = desired[s] - cur
                if not (delta < -1e-9 and cur > 0):
                    continue
                raw = raw_open(s, day)
                if raw is None:
                    order_rows.append({"date": day, "symbol": s, "side": "sell", "shares": 0,
                                       "status": "失败", "reason": "无有效报价"})
                    continue
                avail = cur - frozen_shares(s, day)
                ok, why = rules.can_sell(raw, dn_m.at[day, s], bool(susp_m.at[day, s]),
                                         float(vol_m.at[day, s] or 0.0), avail)
                if not ok:
                    order_rows.append({"date": day, "symbol": s, "side": "sell", "shares": 0,
                                       "status": "受阻", "reason": why})
                    continue
                shares = min(-delta, avail)
                shares = int(shares) if rules.allow_odd_lot_sell and -delta >= avail - 1e-9 \
                    else round_lot(shares, rules.lot_size)
                if shares <= 0:
                    continue
                price = float(cost.slippage_price(raw, "sell"))
                amount = price * shares
                if amount < cfg.min_trade_amount and shares < avail:
                    continue
                fee = cost.fees(amount, "sell")
                cash += amount - fee["total_fee"]
                qty[s] = cur - shares
                trade_rows.append(record_trade(day, s, "sell", price, shares, fee, "调仓/止损"))
                order_rows.append({"date": day, "symbol": s, "side": "sell", "shares": shares,
                                   "status": "成交", "reason": ""})
                if qty[s] <= 1e-9:
                    qty[s] = 0.0
                    entry_px.pop(s, None)
                    peak_px.pop(s, None)
                    stopped[s] = False

            # ---------- 2) 买入（按可用资金等比缩放） ----------
            buys = []
            for s in symbols:
                delta = desired[s] - qty[s]
                if delta > 1e-9:
                    p = raw_open(s, day)
                    if p:
                        buys.append((s, delta, float(cost.slippage_price(p, "buy"))))
            if buys:
                need = sum(d * p * (1.0 + cost.buy_rate) for _, d, p in buys)
                scale = min(1.0, (cash * 0.995) / need) if need > 0 else 0.0
                for s, delta, price in buys:
                    raw = raw_open(s, day)
                    if raw is None or scale <= 0:
                        continue
                    ok, why = rules.can_buy(raw, up_m.at[day, s], bool(susp_m.at[day, s]),
                                            float(vol_m.at[day, s] or 0.0))
                    if not ok:
                        order_rows.append({"date": day, "symbol": s, "side": "buy", "shares": 0,
                                           "status": "受阻", "reason": why})
                        continue
                    want = delta * scale
                    shares = rules.tradable_qty(want, "buy", s, price=price, cash=cash)
                    while shares > 0:
                        amount = price * shares
                        fee = cost.fees(amount, "buy")
                        if amount + fee["total_fee"] <= cash:
                            break
                        shares -= rules.lot_size
                    if shares <= 0:
                        continue
                    amount = price * shares
                    if amount < cfg.min_trade_amount:
                        continue
                    fee = cost.fees(amount, "buy")
                    cash -= amount + fee["total_fee"]
                    qty[s] += shares
                    entry_px.setdefault(s, price)
                    peak_px[s] = max(peak_px.get(s, price), price)
                    avail_day = next_session(i) if rules.t_plus_1 else day
                    frozen[s].append((avail_day, float(shares)))
                    trade_rows.append(record_trade(day, s, "buy", price, shares, fee, "开仓/加仓"))
                    order_rows.append({"date": day, "symbol": s, "side": "buy", "shares": shares,
                                       "status": "成交", "reason": ""})

        # ------------------------------------------------------------------ #
        # 逐日推进
        # ------------------------------------------------------------------ #
        for i, day in enumerate(dates):
            # 解禁（T+1 到期）
            for s in symbols:
                if frozen[s]:
                    frozen[s] = [lt for lt in frozen[s] if lt[0] > day]

            if pending is not None:
                execute(day, i, pending)

            eq = equity_at(day)
            peak_equity = max(peak_equity, eq)
            mv = eq - cash
            row = {"date": day, "cash": cash, "market_value": mv, "equity": eq,
                   "position_weight": (mv / eq) if eq > 0 else 0.0,
                   "n_holdings": sum(1 for s in symbols if qty[s] > 0),
                   "day_return": (eq / rows[-1]["equity"] - 1.0) if rows and rows[-1]["equity"] else 0.0,
                   "drawdown": eq / peak_equity - 1.0}
            rows.append(row)
            weight_rows.append({"date": day, **{s: (qty[s] * (close_m.at[day, s] if pd.notna(close_m.at[day, s]) else 0.0) / eq
                                                       if eq > 0 else 0.0) for s in symbols}})
            if progress_cb and i % 20 == 0:
                progress_cb(i / len(dates))

            # ---------- 收盘风控 ----------
            raw_target = tw.loc[day] if day in tw.index else pd.Series(0.0, index=symbols)
            target = raw_target.astype(float).copy()

            for s in symbols:
                px = close_m.at[day, s]
                if qty[s] > 0 and pd.notna(px) and px > 0:
                    peak_px[s] = max(peak_px.get(s, float(px)), float(px))
                    ep = entry_px.get(s, float(px))
                    hit, reason = False, ""
                    if cfg.trailing_stop and cfg.trailing_stop > 0:
                        if float(px) <= peak_px[s] * (1.0 - cfg.trailing_stop):
                            hit, reason = True, f"移动止损（高点回撤 {cfg.trailing_stop:.0%}）"
                    if not hit and cfg.stop_loss_atr and cfg.stop_loss_atr > 0:
                        a = atr_m.at[day, s]
                        if pd.notna(a) and a > 0 and float(px) < ep - cfg.stop_loss_atr * float(a):
                            hit, reason = True, f"ATR 止损（{cfg.stop_loss_atr:.1f}×ATR）"
                    if hit and not stopped[s]:
                        stopped[s] = True
                        event_rows.append({"date": day, "symbol": s, "event": "止损",
                                           "detail": reason, "price": float(px)})
                # 策略自身已清仓 → 解除止损锁定，允许后续重新开仓
                if stopped.get(s) and float(raw_target.get(s, 0.0)) <= 1e-9 and qty[s] <= 0:
                    stopped[s] = False
            if any(stopped[s] for s in symbols):
                target[[s for s in symbols if stopped[s]]] = 0.0

            # 回撤熔断
            if cfg.max_drawdown_stop and cfg.max_drawdown_stop > 0:
                dd_now = eq / peak_equity - 1.0
                if dd_now <= -abs(cfg.max_drawdown_stop) and (halt_until is None or day >= halt_until):
                    halt_until = dates[min(i + int(cfg.circuit_breaker_cooldown), len(dates) - 1)]
                    event_rows.append({"date": day, "symbol": "组合", "event": "回撤熔断",
                                       "detail": f"回撤 {dd_now:.2%}，冷却至 {halt_until.date()}",
                                       "price": eq})
            if halt_until is not None and day < halt_until:
                target[:] = 0.0

            # 计算权重偏离：小于阈值的目标调整不再触发（降低换手）
            if cfg.rebalance_band and cfg.rebalance_band > 0 and eq > 0:
                cur_w = pd.Series({s: qty[s] * (close_m.at[day, s] if pd.notna(close_m.at[day, s]) else 0.0) / eq
                                   for s in symbols})
                small = (target - cur_w).abs() < cfg.rebalance_band * 0.5
                target[small & (target > 0)] = cur_w[small & (target > 0)]
            pending = target.astype(float)

        # ------------------------------------------------------------------ #
        # 汇总
        # ------------------------------------------------------------------ #
        daily = pd.DataFrame(rows).set_index("date")
        equity = daily["equity"].rename(name)
        bench = None
        if data.benchmark is not None and len(data.benchmark.dropna()) > 5:
            b = data.benchmark.reindex(dates).ffill().dropna()
            if len(b) > 1:
                bench = (b / b.iloc[0] * cfg.initial_cash).rename(data.benchmark_name)
                daily["benchmark"] = bench.reindex(dates)
        daily["benchmark_return"] = (bench.reindex(dates) / cfg.initial_cash - 1.0) if bench is not None else np.nan

        weights = pd.DataFrame(weight_rows).set_index("date")
        holdings = weights.mul(equity, axis=0)
        trades = pd.DataFrame(trade_rows)
        orders = pd.DataFrame(order_rows)
        events = pd.DataFrame(event_rows)
        rt = round_trip_trades(trades)
        metrics = performance_summary(equity, bench, trades, weights, cfg.periods, cfg.risk_free_rate, rt)

        return BacktestResult(name=name, equity=equity, benchmark=bench, daily=daily,
                              weights=weights, holdings=holdings, trades=trades, orders=orders,
                              metrics=metrics, round_trips=rt,
                              config={"config": cfg.describe(), "n_symbols": len(symbols),
                                      "start": str(dates[0].date()), "end": str(dates[-1].date())},
                              events=events)

    # ------------------------------------------------------------------ #
    # 便捷接口
    # ------------------------------------------------------------------ #
    def run_strategy(self, data: DataBundle, strategy, name: Optional[str] = None,
                     progress_cb=None) -> BacktestResult:
        """先生成信号再回测。``strategy`` 需实现 ``weight_matrix(data)``。"""
        weights = strategy.weight_matrix(data)
        return self.run(data, weights, name=name or getattr(strategy, "label", "策略"),
                        progress_cb=progress_cb)


def compare_strategies(data: DataBundle, strategies: Dict[str, object],
                       config: Optional[BacktestConfig] = None,
                       progress_cb=None) -> Dict[str, BacktestResult]:
    """同一数据集上批量回测多个策略，便于横向比较。"""
    bt = Backtester(config)
    out: Dict[str, BacktestResult] = {}
    keys = list(strategies)
    for i, (name, strat) in enumerate(strategies.items()):
        out[name] = bt.run_strategy(data, strat, name=name,
                                    progress_cb=(lambda p, i=i, n=len(keys): progress_cb((i + p) / n))
                                    if progress_cb else None)
    return out
