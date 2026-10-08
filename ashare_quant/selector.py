"""数据驱动选股器：多因子打分 + 硬性过滤 + 大盘择时 → 带「买入日期」的选股清单。

输出每一只入选股票都包含：

* ``signal_date`` 选股日（用当日收盘数据算分）
* ``buy_date``    **买入日期**（下一个交易日开盘执行，自动跳过节假日）
* ``buy_price_low / buy_price_high`` 限价区间（跳空过高则放弃）
* ``stop_loss_price / take_profit_price`` 风控价位
* ``target_weight / shares`` 目标权重与建议股数（整手）
* ``reason`` 入选理由（贡献最大的因子）与 ``warnings`` 风险提示

同时提供 ``select_history``：回看历史上每个调仓日的选股与**买入日期**，并给出买入后
5/20/60 日的事后收益，用于评估选股质量（该列仅用于事后验证，绝不进入信号）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, time as dtime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import indicators as ind
from .data import DataBundle
from .factors import FACTOR_LABELS, composite_score, factor_panel
from .rules import board_limit, get_board, round_lot
from .strategies.base import cross_zscore, rebalance_mask, top_k_weights
from .trading_calendar import get_calendar, next_trade_date

__all__ = ["SelectionConfig", "SelectionResult", "select_stocks", "select_history",
           "selection_stats", "DEFAULT_FACTOR_WEIGHTS", "FILTER_LABELS"]

DEFAULT_FACTOR_WEIGHTS: Dict[str, float] = {
    "mom60": 0.30,      # 中期动量
    "rev5": 0.10,       # 短期反转
    "vol20": 0.20,      # 低波动
    "trend": 0.25,      # 趋势强度
    "amount20": 0.15,   # 流动性（小市值溢价）
}

FILTER_LABELS: Dict[str, str] = {
    "universe": "初始股票池",
    "suspended": "剔除：停牌",
    "st": "剔除：ST/风险警示",
    "new": "剔除：次新/数据不足",
    "price": "剔除：价格区间外",
    "liquidity": "剔除：成交额不足",
    "limit": "剔除：涨跌停附近",
    "trend": "剔除：趋势不达标",
    "timing": "剔除：大盘择时关闭",
}


@dataclass
class SelectionConfig:
    """选股参数（界面/CLI/实盘共用一套）。"""

    # ---- 选股 ----
    top_k: int = 5
    factor_weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_FACTOR_WEIGHTS))
    # ---- 硬性过滤 ----
    min_amount_yuan: float = 5e7      # 近 20 日日均成交额下限（5000 万）
    min_price: float = 2.0
    max_price: float = 500.0
    max_day_change: float = 0.095     # 当日涨跌幅绝对值超限则剔除（不追涨停、不接跌停）
    min_bars: int = 120               # 最少 K 线数量（近似"上市满半年"）
    trend_ma: int = 60                # 收盘价需站上该均线（0=关闭）
    exclude_st: bool = True
    exclude_suspended: bool = True
    exclude_new: bool = True
    # ---- 大盘择时 ----
    use_timing: bool = True
    timing_ma: int = 60
    # ---- 仓位 ----
    total_exposure: float = 1.0
    max_weight: float = 0.25
    weight_mode: str = "equal"        # equal / score
    # ---- 下单与风控价位 ----
    buy_price_buffer: float = 0.02    # 限价上限 = 参考价 ×(1+buffer)，跳空过高则放弃
    max_loss_pct: float = 0.12        # 止损最大亏损幅度
    atr_stop_mult: float = 2.5        # ATR 止损倍数
    atr_window: int = 14
    take_profit_pct: float = 0.20     # 止盈幅度
    max_hold_days: int = 60           # 最长持有交易日（到期强制评估）
    max_pct_of_adv: float = 0.005     # 容量约束：单票市值 <= 20日均成交额×0.5%

    def __post_init__(self) -> None:
        if not self.factor_weights:
            self.factor_weights = dict(DEFAULT_FACTOR_WEIGHTS)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["factor_weights"] = dict(self.factor_weights)
        return d


@dataclass
class SelectionResult:
    """选股结果。``picks`` 即带买入日期的选股清单。"""

    signal_date: pd.Timestamp
    buy_date: pd.Timestamp
    picks: pd.DataFrame
    buy_timing: str = "下一交易日开盘"
    is_realtime: bool = False
    market_open: bool = False
    quote_time: Optional[str] = None
    market_state: str = "正常"
    exposure: float = 1.0
    universe: int = 0
    funnel: pd.DataFrame = field(default_factory=pd.DataFrame)
    config: dict = field(default_factory=dict)

    @property
    def n_picks(self) -> int:
        return int(len(self.picks))

    def is_intraday(self) -> bool:
        return "今日" in str(self.buy_timing)

    def summary_text(self) -> str:
        tag = ("实时行情源（盘中实时）" if self.market_open
               else ("实时行情源（当前休市/收盘，用最近成交价）" if self.is_realtime else "离线/历史数据"))
        lines = [f"【选股清单】选股日 {self.signal_date:%Y-%m-%d} → 买入日期 {self.buy_date:%Y-%m-%d}"
                 f"（{self.buy_timing}）　数据：{tag}" + (f"　行情时间 {self.quote_time}" if self.quote_time else ""),
                 f"市场状态：{self.market_state}　建议总仓位：{self.exposure:.0%}　"
                 f"股票池 {self.universe} 只 → 入选 {self.n_picks} 只"]
        if self.picks.empty:
            lines.append("  本期无入选标的（空仓等待）")
        for i, r in self.picks.iterrows():
            lines.append(f"  {i+1}. {r['name']}({r['symbol']}) 买入日期 {r['buy_date']:%Y-%m-%d}　"
                         f"限价 {r['buy_price_low']:.2f}~{r['buy_price_high']:.2f}　"
                         f"约 {int(r['shares'])} 股　止损 {r['stop_loss_price']:.2f}　"
                         f"止盈 {r['take_profit_price']:.2f}")
        lines.append("（研究参考，不构成投资建议）")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 内部工具
# --------------------------------------------------------------------------- #
def _market_state(bundle: DataBundle, upto: pd.Timestamp, timing_ma: int) -> Tuple[str, float]:
    """大盘择时：指数在均线上方 → 满仓，否则空仓。返回 (状态, 建议总仓位)。"""
    ref = bundle.benchmark if bundle.benchmark is not None else bundle.close_matrix().mean(axis=1)
    ref = pd.Series(ref).dropna()
    ref = ref[ref.index <= upto]
    if len(ref) < max(20, timing_ma // 2):
        return "数据不足（默认满仓）", 1.0
    ma = ind.sma(ref, timing_ma)
    fast = ind.sma(ref, max(5, timing_ma // 4))
    above = bool(ref.iloc[-1] > ma.iloc[-1])
    trend_up = bool(fast.iloc[-1] > ma.iloc[-1])
    if above and trend_up:
        return "多头（指数在均线上方）", 1.0
    if above:
        return "震荡（指数勉强站上均线）", 0.5
    return "空头（指数在均线下方）", 0.0


def _reason_text(row: pd.Series, weights: Dict[str, float], top: int = 2) -> str:
    """挑出贡献最大的因子作为入选理由。"""
    contrib = {}
    for k, w in weights.items():
        v = row.get(f"z_{k}", np.nan)
        if pd.notna(v) and w:
            contrib[k] = float(v) * float(w)
    if not contrib:
        return "综合打分入选"
    items = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top]
    return "、".join(f"{FACTOR_LABELS.get(k, k)}{'强' if v > 0 else '偏弱'}" for k, v in items)


def _build_universe_frame(bundle: DataBundle, as_of: pd.Timestamp, panel: Dict[str, pd.DataFrame],
                          cfg: SelectionConfig) -> Tuple[pd.DataFrame, pd.Series]:
    """在 as_of 日构建候选池明细（含各过滤条件标记）。"""
    close_m = bundle.close_matrix()
    amount_m = bundle.amount_matrix()
    susp_m = bundle.suspended_matrix()
    hist = close_m.loc[:as_of]
    row: Dict[str, pd.Series] = {}
    row["close"] = hist.ffill().iloc[-1] if len(hist) else pd.Series(dtype=float)
    row["bars"] = hist.notna().sum()
    row["amount20"] = amount_m.loc[:as_of].tail(20).mean()
    row["chg"] = hist.ffill().pct_change().iloc[-1] if len(hist) > 1 else pd.Series(dtype=float)
    row["suspended"] = susp_m.loc[as_of] if as_of in susp_m.index else pd.Series(False, index=close_m.columns)
    # 关闭均线过滤（trend_ma=0）时不计算均线，避免 rolling(0) 报错
    if cfg.trend_ma and int(cfg.trend_ma) > 0 and len(hist):
        ma_ref = ind.sma(hist.ffill(), int(cfg.trend_ma)).iloc[-1]
    else:
        ma_ref = pd.Series(np.nan, index=close_m.columns)
    row["ma"] = ma_ref
    df = pd.DataFrame(row)
    df["name"] = [bundle.name_of(s) for s in df.index]
    df["industry"] = [bundle.industry_of(s) for s in df.index]
    df["board"] = [get_board(s) for s in df.index]
    df["is_st"] = [bundle.is_st(s) for s in df.index]

    # 过滤条件（True 表示通过）
    df["pass_suspended"] = ~df["suspended"].astype(bool) if cfg.exclude_suspended else True
    df["pass_st"] = (~df["is_st"]) if cfg.exclude_st else True
    df["pass_new"] = (df["bars"] >= cfg.min_bars) if cfg.exclude_new else True
    df["pass_price"] = df["close"].between(cfg.min_price, cfg.max_price)
    df["pass_liquidity"] = df["amount20"] >= cfg.min_amount_yuan
    df["pass_limit"] = df["chg"].abs() < cfg.max_day_change
    if cfg.trend_ma and cfg.trend_ma > 0:
        df["pass_trend"] = df["close"] > df["ma"]
    else:
        df["pass_trend"] = True
    df["pass_all"] = df[[c for c in df.columns if c.startswith("pass_")]].all(axis=1)
    return df, df["pass_all"]


# --------------------------------------------------------------------------- #
# 主入口：单期选股
# --------------------------------------------------------------------------- #
def resolve_buy_date(as_of: pd.Timestamp, calendar: Optional[pd.DatetimeIndex] = None,
                     now: Optional[datetime] = None,
                     intraday: bool = False) -> Tuple[pd.Timestamp, str]:
    """判断买入日期：盘中实时行情 → 今日；其余 → 下一交易日开盘。"""
    cal = calendar if calendar is not None else get_calendar()
    now = now or datetime.now()
    today = pd.Timestamp(now.date())
    t = now.time()
    is_td = today in set(cal)
    in_session = bool(is_td and ((dtime(9, 30) <= t <= dtime(11, 30))
                                 or (dtime(13, 0) <= t <= dtime(15, 0))))
    if intraday and in_session and pd.Timestamp(as_of).normalize() == today:
        return today, f"今日盘中（{t.strftime('%H:%M')} 实时价买入）"
    nxt = next_trade_date(as_of, 1, cal)
    if pd.Timestamp(as_of).normalize() >= today and not in_session:
        return nxt, "下一交易日开盘（当前非交易时段）"
    return nxt, "下一交易日开盘"


def select_stocks(bundle: DataBundle, config: Optional[SelectionConfig] = None,
                  capital: float = 1_000_000.0, as_of: Optional[pd.Timestamp] = None,
                  calendar: Optional[pd.DatetimeIndex] = None,
                  now: Optional[datetime] = None,
                  intraday: Optional[bool] = None,
                  panel: Optional[Dict[str, pd.DataFrame]] = None) -> SelectionResult:
    """按最新（或指定）交易日数据选出股票，并给出明确的**买入日期**。

    Parameters
    ----------
    as_of : Timestamp, optional
        选股基准日（用该日及之前的收盘数据），默认取行情最后一根 K 线；
        买入日期自动取它的下一个交易日。
    """
    cfg = config or SelectionConfig()
    cal = calendar if calendar is not None else (bundle.calendar if len(bundle) else get_calendar())
    as_of = pd.Timestamp(as_of).normalize() if as_of is not None else pd.Timestamp(bundle.calendar[-1])
    if as_of not in set(bundle.calendar):
        earlier = bundle.calendar[bundle.calendar <= as_of]
        as_of = pd.Timestamp(earlier[-1]) if len(earlier) else pd.Timestamp(bundle.calendar[0])
    # 买入日期：盘中实时行情 → 今日；休市/收盘 → 下一交易日开盘
    rt_status = getattr(bundle, "realtime_status", None)
    if intraday is None:
        intraday = bool(getattr(rt_status, "is_open", False))
    quote_time = getattr(bundle, "quote_time", None)
    buy_date, buy_timing = resolve_buy_date(as_of, cal, now=now, intraday=bool(intraday))
    rt_kwargs = {
        "is_realtime": bool(rt_status is not None),
        "market_open": bool(getattr(rt_status, "is_open", False)),
        "quote_time": (quote_time.strftime("%Y-%m-%d %H:%M:%S") if hasattr(quote_time, "strftime")
                       else (str(quote_time) if quote_time else None)),
    }

    # panel 可外部传入（例如用 IC 分析筛选出的有效因子面板）；空 dict 视为未提供
    if panel is None or len(panel) == 0:
        want = [k for k in (cfg.factor_weights or {}) if k]
        panel = None
        try:
            # 富因子面板（20 个因子）：只计算用户选中的那几个，避免静默丢因子
            from .factor_research import build_factor_panel as _rich_panel
            panel = _rich_panel(bundle, factors=(want or None))
        except Exception:
            panel = None
        if not panel:
            panel = factor_panel(bundle)
        if want:
            missing = [k for k in want if k not in panel]
            if missing:
                # 兜底：用标准面板补上能补的
                std = factor_panel(bundle)
                for k in missing:
                    if k in std:
                        panel[k] = std[k]
    universe_df, passed = _build_universe_frame(bundle, as_of, panel, cfg)
    funnel_rows = [{"步骤": FILTER_LABELS["universe"], "通过数量": int(len(universe_df))}]
    for key, label in (("pass_suspended", FILTER_LABELS["suspended"]),
                       ("pass_st", FILTER_LABELS["st"]), ("pass_new", FILTER_LABELS["new"]),
                       ("pass_price", FILTER_LABELS["price"]),
                       ("pass_liquidity", FILTER_LABELS["liquidity"]),
                       ("pass_limit", FILTER_LABELS["limit"]),
                       ("pass_trend", FILTER_LABELS["trend"])):
        funnel_rows.append({"步骤": label, "通过数量": int(universe_df[key].sum())})

    # 市场择时
    state, exposure = _market_state(bundle, as_of, cfg.timing_ma)
    if cfg.use_timing and exposure <= 0:
        funnel_rows.append({"步骤": FILTER_LABELS["timing"], "通过数量": 0})
        return SelectionResult(signal_date=as_of, buy_date=buy_date, buy_timing=buy_timing,
                               picks=pd.DataFrame(), market_state=state, exposure=0.0,
                               universe=len(universe_df), funnel=pd.DataFrame(funnel_rows),
                               config=cfg.to_dict(), **rt_kwargs)
    if not cfg.use_timing:
        state, exposure = "择时关闭", 1.0
    exposure = min(exposure, cfg.total_exposure)

    # 因子打分（只取 as_of 这一行的横截面）
    weights = {k: v for k, v in cfg.factor_weights.items() if k in panel}
    score_full = composite_score({k: panel[k] for k in weights}, weights)
    zscores = {k: cross_zscore(panel[k]).loc[as_of] for k in weights if as_of in panel[k].index}
    score_row = score_full.loc[as_of] if as_of in score_full.index else pd.Series(dtype=float)

    candidates = universe_df[passed].copy()
    candidates["score"] = score_row.reindex(candidates.index)
    for k, s in zscores.items():
        candidates[f"z_{k}"] = s.reindex(candidates.index)
    candidates = candidates[candidates["score"].notna()]
    if candidates.empty:
        funnel_rows.append({"步骤": "无有效打分", "通过数量": 0})
        return SelectionResult(signal_date=as_of, buy_date=buy_date, buy_timing=buy_timing,
                               picks=pd.DataFrame(), market_state=state, exposure=0.0,
                               universe=len(universe_df), funnel=pd.DataFrame(funnel_rows),
                               config=cfg.to_dict(), **rt_kwargs)

    # 选前 K 名并分配权重
    score_mat = candidates["score"].to_frame().T
    score_mat.index = pd.DatetimeIndex([as_of])
    w_mat = top_k_weights(score_mat, k=int(cfg.top_k), max_weight=float(cfg.max_weight),
                          weight_mode=cfg.weight_mode, total_exposure=float(exposure))
    w_row = w_mat.iloc[0]
    chosen = w_row[w_row > 0].sort_values(ascending=False)
    funnel_rows.append({"步骤": f"入选（Top {cfg.top_k}）", "通过数量": int(len(chosen))})

    # ATR（用于止损）
    atr_row = {}
    for s in chosen.index:
        df = bundle.get(s)
        try:
            atr_row[s] = float(ind.atr(df["high"], df["low"], df["close"], cfg.atr_window).loc[:as_of].iloc[-1])
        except Exception:
            atr_row[s] = np.nan

    rows: List[dict] = []
    for rank, (sym, weight) in enumerate(chosen.items(), start=1):
        info = candidates.loc[sym]
        close = float(info["close"])
        atr = atr_row.get(sym, np.nan)
        st = bool(info["is_st"])
        limit_pct = board_limit(sym, st)
        limit_up_next = round(close * (1 + limit_pct), 2)
        price_mid = round(close * (1 + cfg.buy_price_buffer / 2), 2)
        price_high = round(min(close * (1 + cfg.buy_price_buffer), limit_up_next), 2)
        price_low = round(close * (1 - cfg.buy_price_buffer), 2)
        atr_stop = close - cfg.atr_stop_mult * atr if pd.notna(atr) else np.nan
        stop_price = round(max(close * (1 - cfg.max_loss_pct), atr_stop) if pd.notna(atr_stop)
                           else close * (1 - cfg.max_loss_pct), 2)
        take_price = round(close * (1 + cfg.take_profit_pct), 2)
        target_value = float(capital) * float(weight)
        # 容量约束：单票买入金额不超过 20 日均成交额的一定比例（默认 0.5%）。
        # 否则小盘股的回测收益在实盘根本买不到（冲击成本会吃掉收益）。
        _adv20 = float(info.get("amount20", 0.0) or 0.0)
        if cfg.max_pct_of_adv and _adv20 > 0:
            target_value = min(target_value, _adv20 * float(cfg.max_pct_of_adv))
        shares = int(round_lot(target_value / max(price_mid, 0.01), 100))
        amount = round(shares * price_mid, 2)
        warns = []
        if close >= limit_up_next * 0.995:
            warns.append("已接近涨停，注意买入价格上限")
        if info["amount20"] < cfg.min_amount_yuan * 1.5:
            warns.append("成交额接近下限，注意流动性")
        if st:
            warns.append("ST 股票，涨跌幅 ±5%")
        rows.append({
            "signal_date": as_of, "buy_date": buy_date, "buy_timing": buy_timing,
            "symbol": sym, "name": info["name"],
            "industry": info["industry"], "board": info["board"], "rank": rank,
            "close": round(close, 2), "score": round(float(info["score"]), 4),
            "target_weight": round(float(weight), 4), "shares": shares, "amount": amount,
            "buy_price_low": price_low, "buy_price_high": price_high, "ref_price": price_mid,
            "next_limit_up": limit_up_next, "limit_pct": limit_pct,
            "atr": round(float(atr), 3) if pd.notna(atr) else np.nan,
            "stop_loss_price": stop_price, "take_profit_price": take_price,
            "max_hold_days": int(cfg.max_hold_days),
            "amount20_wan": round(float(info["amount20"]) / 1e4, 1),
            "day_chg": round(float(info["chg"]), 4),
            "reason": _reason_text(info, weights),
            "warnings": "；".join(warns),
        })
        for k in weights:
            rows[-1][f"factor_{FACTOR_LABELS.get(k, k)}"] = round(float(info.get(f"z_{k}", np.nan)), 3)

    picks = pd.DataFrame(rows)
    return SelectionResult(signal_date=as_of, buy_date=buy_date, buy_timing=buy_timing, picks=picks,
                           **rt_kwargs,
                           market_state=state, exposure=float(exposure),
                           universe=int(len(universe_df)), funnel=pd.DataFrame(funnel_rows),
                           config=cfg.to_dict())


# --------------------------------------------------------------------------- #
# 历史选股（用于验证选股质量与展示"买入日期"）
# --------------------------------------------------------------------------- #
def select_history(bundle: DataBundle, config: Optional[SelectionConfig] = None,
                   capital: float = 1_000_000.0, freq: object = "W",
                   start: Optional[pd.Timestamp] = None, end: Optional[pd.Timestamp] = None,
                   lookaheads: Sequence[int] = (5, 20, 60),
                   calendar: Optional[pd.DatetimeIndex] = None) -> pd.DataFrame:
    """回看历史每期选股（含买入日期），并附加买入后 N 日收益用于事后评估。"""
    cfg = config or SelectionConfig()
    close = bundle.close_matrix()
    if close.empty:
        return pd.DataFrame()
    cal = calendar if calendar is not None else bundle.calendar
    dates = pd.DatetimeIndex(bundle.calendar)
    mask = rebalance_mask(dates, freq)
    sel_dates = dates[mask.to_numpy()]
    if start is not None:
        sel_dates = sel_dates[sel_dates >= pd.Timestamp(start)]
    if end is not None:
        sel_dates = sel_dates[sel_dates <= pd.Timestamp(end)]
    sel_dates = sel_dates[sel_dates >= dates[min(cfg.min_bars, len(dates) - 1)]]
    if len(sel_dates) > 400:
        sel_dates = sel_dates[-400:]

    rows: List[dict] = []
    for d in sel_dates:
        try:
            res = select_stocks(bundle, cfg, capital=capital, as_of=d, calendar=cal)
        except Exception:
            continue
        if res.picks.empty:
            rows.append({"signal_date": d, "buy_date": res.buy_date, "symbol": None,
                         "name": "空仓", "market_state": res.market_state,
                         "target_weight": 0.0, "score": np.nan})
            continue
        for _, p in res.picks.iterrows():
            rec = p.to_dict()
            rec["market_state"] = res.market_state
            # 事后收益（仅用于事后验证，不参与选股）
            if p["buy_date"] in close.index:
                base_idx = close.index.get_loc(p["buy_date"])
                base_px = float(close.iloc[base_idx][p["symbol"]])
                for h in lookaheads:
                    j = min(base_idx + int(h), len(close) - 1)
                    px = float(close.iloc[j][p["symbol"]])
                    rec[f"ret_{h}d"] = round(px / base_px - 1.0, 4) if base_px > 0 else np.nan
            rows.append(rec)
    hist = pd.DataFrame(rows)
    if not hist.empty and "buy_date" in hist.columns:
        hist = hist.sort_values(["buy_date", "rank"], na_position="last").reset_index(drop=True)
    return hist


def selection_stats(hist: pd.DataFrame, horizons: Sequence[int] = (5, 20, 60),
                    benchmark: Optional[pd.Series] = None) -> pd.DataFrame:
    """历史选股的胜率与平均收益（按买入日期统计）。"""
    if hist is None or hist.empty or "symbol" not in hist.columns:
        return pd.DataFrame()
    df = hist[hist["symbol"].notna()].copy()
    if df.empty:
        return pd.DataFrame()
    rows = []
    for h in horizons:
        col = f"ret_{h}d"
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        # 等权组合：同一买入日期的选股等权平均
        by_date = df.groupby("buy_date")[col].mean().dropna()
        rows.append({
            "持有期": f"{h} 个交易日", "样本数": int(len(s)),
            "个股平均收益": float(s.mean()), "个股胜率": float((s > 0).mean()),
            "组合平均收益": float(by_date.mean()) if len(by_date) else np.nan,
            "组合胜率": float((by_date > 0).mean()) if len(by_date) else np.nan,
            "最好": float(s.max()), "最差": float(s.min()),
            "选股期数": int(len(by_date)),
        })
    return pd.DataFrame(rows)









