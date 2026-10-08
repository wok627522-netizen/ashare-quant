"""数据层：统一数据结构 + 四种数据来源。

1. ``synthetic_bundle``   离线合成数据（无需联网，用于演示、教学与单元测试）
2. ``load_csv_bundle``    本地 CSV（自己导出的行情，支持宽表/长表）
3. ``fetch_akshare_bundle``  AkShare 在线抓取（免费，需 pip install akshare）
4. ``fetch_tushare_bundle``  TuShare Pro（需 token，见环境变量 TUSHARE_TOKEN）

统一输出 ``DataBundle``，所有下游模块（策略/回测/界面）只依赖它。
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .config import DATA_DIR, DEFAULT_POOL
from .rules import board_limit, get_board, limit_prices

OHLCV = ["open", "high", "low", "close", "volume", "amount"]
CN_HOLIDAYS = {(1, 1), (1, 2), (1, 3), (5, 1), (5, 2), (5, 3),
               (10, 1), (10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7)}

__all__ = [
    "DataBundle", "trading_calendar", "synthetic_bundle", "load_csv_bundle",
    "fetch_akshare_bundle", "fetch_tushare_bundle", "fetch_symbol_list",
    "save_bundle", "load_bundle", "normalize_price_df", "ensure_bundle",
]


# --------------------------------------------------------------------------- #
# 交易日历
# --------------------------------------------------------------------------- #
def trading_calendar(start: Union[str, date, datetime], end: Union[str, date, datetime]) -> pd.DatetimeIndex:
    """近似交易日历：工作日剔除主要法定节假日（离线场景使用）。

    接入 AkShare 后可用官方交易日历替换，见 ``fetch_trade_calendar``。
    """
    days = pd.bdate_range(pd.Timestamp(start), pd.Timestamp(end))
    keep = [d for d in days if (d.month, d.day) not in CN_HOLIDAYS]
    return pd.DatetimeIndex(keep)


# --------------------------------------------------------------------------- #
# 核心数据结构
# --------------------------------------------------------------------------- #
@dataclass
class DataBundle:
    """多标的行情容器。

    Attributes
    ----------
    prices : Dict[str, DataFrame]
        每只股票的日线表，索引为 DatetimeIndex，列至少包含
        open/high/low/close/volume，推荐附带 amount/prev_close/limit_up/limit_down/suspended。
    calendar : DatetimeIndex
        统一交易日序列（所有标的按它对齐，停牌日 ffill）。
    benchmark : Series
        基准指数收盘价（用于超额收益、Alpha/Beta）。
    meta : DataFrame
        标的静态信息：name / industry / board / is_st。
    """

    prices: Dict[str, pd.DataFrame]
    calendar: pd.DatetimeIndex
    benchmark: Optional[pd.Series] = None
    benchmark_name: str = "沪深300"
    meta: Optional[pd.DataFrame] = None
    freq: str = "1d"

    def __post_init__(self) -> None:
        self.calendar = pd.DatetimeIndex(pd.to_datetime(self.calendar)).sort_values().unique()
        self.calendar = pd.DatetimeIndex(self.calendar)
        self.prices = {str(k): normalize_price_df(v, str(k)) for k, v in self.prices.items()}
        for sym, df in self.prices.items():
            self.prices[sym] = df.reindex(self.calendar).ffill()
            if "suspended" not in self.prices[sym]:
                self.prices[sym]["suspended"] = False
            self.prices[sym]["suspended"] = self.prices[sym]["suspended"].fillna(False).astype(bool)
            if "volume" in self.prices[sym]:
                zero_vol = self.prices[sym]["volume"].fillna(0.0) <= 0
                self.prices[sym]["suspended"] = self.prices[sym]["suspended"] | zero_vol
        if self.benchmark is not None:
            b = pd.Series(self.benchmark).astype(float)
            b.index = pd.DatetimeIndex(pd.to_datetime(b.index))
            self.benchmark = b.reindex(self.calendar).ffill().dropna()
        if self.meta is None:
            self.meta = pd.DataFrame(
                {"name": list(self.prices.keys())}, index=list(self.prices.keys())
            )
        else:
            self.meta = self.meta.copy()
            self.meta.index = self.meta.index.astype(str)
            if "name" not in self.meta.columns:
                self.meta["name"] = self.meta.index

    # ---------------- 基础访问 ----------------
    @property
    def symbols(self) -> List[str]:
        return sorted(self.prices.keys())

    def __len__(self) -> int:
        return len(self.calendar)

    def __contains__(self, symbol: str) -> bool:
        return str(symbol) in self.prices

    def get(self, symbol: str) -> pd.DataFrame:
        return self.prices[str(symbol)]

    def name_of(self, symbol: str) -> str:
        try:
            return str(self.meta.loc[str(symbol), "name"])
        except Exception:
            return str(symbol)

    def industry_of(self, symbol: str) -> str:
        try:
            return str(self.meta.loc[str(symbol), "industry"])
        except Exception:
            return "未知"

    def is_st(self, symbol: str) -> bool:
        try:
            return bool(self.meta.loc[str(symbol), "is_st"])
        except Exception:
            return False

    # ---------------- 矩阵视图（策略层最常用） ----------------
    def _field_matrix(self, col: str) -> pd.DataFrame:
        data = {}
        for sym in self.symbols:
            df = self.prices[sym]
            data[sym] = df[col] if col in df.columns else pd.Series(np.nan, index=df.index)
        return pd.DataFrame(data).reindex(self.calendar)

    def close_matrix(self) -> pd.DataFrame:
        return self._field_matrix("close")

    def open_matrix(self) -> pd.DataFrame:
        return self._field_matrix("open")

    def high_matrix(self) -> pd.DataFrame:
        return self._field_matrix("high")

    def low_matrix(self) -> pd.DataFrame:
        return self._field_matrix("low")

    def volume_matrix(self) -> pd.DataFrame:
        return self._field_matrix("volume")

    def amount_matrix(self) -> pd.DataFrame:
        return self._field_matrix("amount")

    def returns_matrix(self) -> pd.DataFrame:
        return self.close_matrix().pct_change()

    def suspended_matrix(self) -> pd.DataFrame:
        return self._field_matrix("suspended").fillna(False).astype(bool)

    # ---------------- 变换 ----------------
    def slice(self, start=None, end=None) -> "DataBundle":
        cal = self.calendar
        if start is not None:
            cal = cal[cal >= pd.Timestamp(start)]
        if end is not None:
            cal = cal[cal <= pd.Timestamp(end)]
        bm = self.benchmark
        if bm is not None:
            bm = bm[(bm.index >= cal[0]) & (bm.index <= cal[-1])] if len(cal) else bm
        return DataBundle(
            prices={s: self.prices[s].reindex(cal).copy() for s in self.symbols},
            calendar=cal, benchmark=bm, benchmark_name=self.benchmark_name,
            meta=self.meta.copy(), freq=self.freq,
        )

    def filter(self, symbols: Sequence[str]) -> "DataBundle":
        keep = [s for s in symbols if s in self.prices]
        sub = DataBundle(prices={s: self.prices[s] for s in keep}, calendar=self.calendar,
                         benchmark=self.benchmark, benchmark_name=self.benchmark_name,
                         meta=self.meta.loc[[s for s in self.meta.index if s in keep]]
                         if self.meta is not None else None, freq=self.freq)
        return sub

    def describe(self) -> pd.DataFrame:
        rows = []
        for sym in self.symbols:
            df = self.prices[sym]
            close = df["close"].dropna()
            ret = close.pct_change().dropna()
            rows.append({
                "symbol": sym, "name": self.name_of(sym), "industry": self.industry_of(sym),
                "board": get_board(sym), "is_st": self.is_st(sym),
                "start": df.index.min().date() if len(df) else None,
                "end": df.index.max().date() if len(df) else None,
                "bars": int(len(df)), "last_close": round(float(close.iloc[-1]), 2) if len(close) else np.nan,
                "annual_vol": round(float(ret.std(ddof=0) * np.sqrt(252)), 4) if len(ret) > 5 else np.nan,
                "period_return": round(float(close.iloc[-1] / close.iloc[0] - 1), 4) if len(close) > 1 else np.nan,
            })
        return pd.DataFrame(rows)

    def to_long(self) -> pd.DataFrame:
        frames = []
        for sym in self.symbols:
            df = self.prices[sym][[c for c in OHLCV if c in self.prices[sym].columns]].copy()
            df["symbol"] = sym
            frames.append(df.reset_index().rename(columns={"index": "date"}))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def coverage(self) -> pd.DataFrame:
        return self.close_matrix().notna()


# --------------------------------------------------------------------------- #
# 行情表标准化
# --------------------------------------------------------------------------- #
_COL_ALIAS = {
    "日期": "date", "时间": "date", "trade_date": "date", "datetime": "date",
    "开盘": "open", "今开": "open", "最高": "high", "最低": "low", "收盘": "close",
    "最新价": "close", "成交量": "volume", "成交额": "amount", "成交金额": "amount",
    "涨跌幅": "pct_chg", "换手率": "turnover", "prev_close": "prev_close", "昨收": "prev_close",
}


def normalize_price_df(df: pd.DataFrame, symbol: str = "", is_st: bool = False) -> pd.DataFrame:
    """把任意来源的行情表规范成统一格式，并补算涨跌停价与停牌标记。"""
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex) is False and "date" not in out.columns:
        out = out.reset_index()
    out = out.rename(columns={k: v for k, v in _COL_ALIAS.items() if k in out.columns})
    out.columns = [str(c).strip().lower() for c in out.columns]
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.dropna(subset=["date"]).set_index("date")
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index)).normalize()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    for c in ("open", "high", "low", "close", "volume", "amount"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if "close" not in out.columns:
        raise ValueError(f"{symbol}: 行情数据缺少 close 列")
    for c in ("open", "high", "low"):
        if c not in out.columns:
            out[c] = out["close"]
    if "volume" not in out.columns:
        out["volume"] = np.nan
    if "amount" not in out.columns:
        out["amount"] = out.get("volume", 0) * out["close"]
    if "prev_close" not in out.columns:
        out["prev_close"] = out["close"].shift(1)
    # 用昨收与板块限制推导涨跌停价（若数据源已提供则保留）
    if symbol:
        pct = board_limit(symbol, is_st)
        prev = out["prev_close"]
        if "limit_up" not in out.columns:
            out["limit_up"] = np.round(prev * (1 + pct), 2)
        if "limit_down" not in out.columns:
            out["limit_down"] = np.round(prev * (1 - pct), 2)
    else:
        prev = out["prev_close"]
        out.setdefault("limit_up", np.round(prev * 1.10, 2))
        out.setdefault("limit_down", np.round(prev * 0.90, 2))
    out["limit_up"] = pd.to_numeric(out.get("limit_up"), errors="coerce")
    out["limit_down"] = pd.to_numeric(out.get("limit_down"), errors="coerce")
    if "suspended" not in out.columns:
        out["suspended"] = out["volume"].fillna(0) <= 0
    out["suspended"] = out["suspended"].astype(bool)
    return out[[c for c in OHLCV + ["prev_close", "limit_up", "limit_down", "suspended"]
                if c in out.columns]]


# --------------------------------------------------------------------------- #
# 离线合成数据
# --------------------------------------------------------------------------- #
def synthetic_bundle(n_stocks: int = 12, start: str = "2020-01-01", end: str = "2024-12-31",
                     seed: int = 42, with_st: bool = True, benchmark_name: str = "沪深300(合成)",
                     bull_bias: float = 0.0, market_drift: float = 0.04) -> DataBundle:
    """生成带 A 股特征（涨跌停、停牌、T+1、板块差异、行业因子、牛熊切换）的合成行情。

    用途：无网络环境下的演示、教学、策略开发与单元测试；
    它**不是**真实行情，不能据此做任何投资决策。
    """
    rng = np.random.default_rng(int(seed))
    cal = trading_calendar(start, end)
    n = len(cal)
    if n < 60:
        raise ValueError("时间区间过短，至少需要 60 个交易日")

    pool = list(DEFAULT_POOL)
    while len(pool) < n_stocks:
        i = len(pool)
        pool.append({"symbol": f"{600000 + i * 7:06d}", "name": f"示例标的{i:02d}", "industry": "综合"})
    pool = pool[:n_stocks]
    industries = sorted({p["industry"] for p in pool})

    # ---- 市场状态：3 状态马尔可夫（牛市 / 震荡 / 熊市）----
    drifts = np.array([0.00085, 0.00005, -0.00085])
    vols = np.array([0.0085, 0.0115, 0.019])
    trans = np.array([[0.965, 0.030, 0.005], [0.020, 0.960, 0.020], [0.010, 0.040, 0.950]])
    state, states = 1, np.empty(n, dtype=int)
    for t in range(n):
        states[t] = state
        state = int(rng.choice(3, p=trans[state]))
    # 对数漂移补偿（+σ²/2）消除波动率拖累；再做样本内去均值，
    # 使长期趋势由 market_drift 参数显式控制（默认年化 +4%，温和上行），
    # 避免马尔可夫状态的稳态分布不均导致合成市场意外系统性偏空。
    market = (rng.normal(drifts[states] + 0.5 * vols[states] ** 2, vols[states])
              + 0.00005 * np.sin(np.arange(n) / 21.0))
    market = market - market.mean() + (float(market_drift) + float(bull_bias)) / 252.0
    # 极少数系统性跳空（政策/外盘冲击）
    shock_idx = rng.choice(n, size=max(1, n // 250), replace=False)
    market[shock_idx] += rng.choice([-1, 1], size=len(shock_idx)) * rng.uniform(0.02, 0.045, len(shock_idx))

    ind_factor = {ind: rng.normal(0.0001, 0.006, n) for ind in industries}

    prices: Dict[str, pd.DataFrame] = {}
    meta_rows = []
    betas = rng.uniform(0.75, 1.35, len(pool))
    for i, item in enumerate(pool):
        sym, ind = item["symbol"], item["industry"]
        is_st = bool(with_st and i == len(pool) - 1 and len(pool) >= 5)
        pct = board_limit(sym, is_st)
        idio = rng.normal(0.0, rng.uniform(0.010, 0.018), n)
        load = rng.uniform(0.4, 0.9)
        ret = betas[i] * market + load * ind_factor[ind] + idio
        # 涨跌停截断
        ret = np.clip(ret, -pct, pct)
        # 一字板 / T 字板事件
        lock_up = rng.random(n) < 0.0035
        lock_dn = rng.random(n) < 0.0025
        ret = np.where(lock_up, pct, ret)
        ret = np.where(lock_dn, -pct, ret)
        ret[0] = 0.0

        star = float(rng.uniform(6.0, 60.0))
        close = star * np.cumprod(1.0 + ret)
        prev_close = np.concatenate([[star], close[:-1]])
        # 跳空开盘
        gap = 0.35 * ret + rng.normal(0, 0.0035, n)
        open_ = np.clip(prev_close * (1 + gap), prev_close * (1 - pct), prev_close * (1 + pct))
        open_[0] = star
        hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
        lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
        hi = np.clip(hi, prev_close * (1 - pct), prev_close * (1 + pct))
        lo = np.clip(lo, prev_close * (1 - pct), prev_close * (1 + pct))
        hi = np.maximum.reduce([hi, open_, close])
        lo = np.minimum.reduce([lo, open_, close])

        suspended = rng.random(n) < 0.0025
        base_vol = rng.uniform(5e6, 6e7)
        vol = base_vol * (1 + 4 * np.abs(ret)) * rng.lognormal(0, 0.45, n)
        vol = np.where(suspended, 0.0, vol)
        close = np.where(suspended, prev_close, close)
        open_ = np.where(suspended, prev_close, open_)
        hi = np.where(suspended, prev_close, hi)
        lo = np.where(suspended, prev_close, lo)
        amount = vol * (open_ + close) / 2.0

        df = pd.DataFrame({
            "open": np.round(open_, 2), "high": np.round(hi, 2), "low": np.round(lo, 2),
            "close": np.round(close, 2), "volume": np.round(vol).astype(float),
            "amount": np.round(amount, 2), "prev_close": np.round(prev_close, 2),
            "suspended": suspended,
        }, index=cal)
        df["limit_up"] = np.round(df["prev_close"] * (1 + pct), 2)
        df["limit_down"] = np.round(df["prev_close"] * (1 - pct), 2)
        prices[sym] = df
        meta_rows.append({"symbol": sym, "name": item["name"], "industry": ind,
                          "board": get_board(sym), "is_st": is_st, "beta": round(float(betas[i]), 2)})

    bench = 3000.0 * np.cumprod(1.0 + market + rng.normal(0, 0.0012, n))
    benchmark = pd.Series(bench, index=cal, name=benchmark_name)
    meta = pd.DataFrame(meta_rows).set_index("symbol")
    return DataBundle(prices=prices, calendar=cal, benchmark=benchmark,
                      benchmark_name=benchmark_name, meta=meta)


# --------------------------------------------------------------------------- #
# 本地 CSV
# --------------------------------------------------------------------------- #
def load_csv_bundle(path: Union[str, Path], calendar: Optional[pd.DatetimeIndex] = None,
                    long_format: bool = False, benchmark: Optional[pd.Series] = None) -> DataBundle:
    """读取本地行情。

    - 目录模式：每个文件一只股票，文件名（不含扩展名）= 代码，列为 date/open/high/low/close/volume
    - 单文件长表模式：``long_format=True``，需包含 symbol/date/OHLCV 列
    """
    p = Path(path)
    if p.is_dir():
        frames: Dict[str, pd.DataFrame] = {}
        for f in sorted(list(p.glob("*.csv")) + list(p.glob("*.CSV"))):
            sym = f.stem.split("_")[0]
            frames[sym] = normalize_price_df(pd.read_csv(f), sym)
        if not frames:
            raise FileNotFoundError(f"{p} 下没有找到 CSV 文件")
    else:
        raw = pd.read_csv(p)
        raw.columns = [str(c).lower() for c in raw.columns]
        if not long_format and "symbol" not in raw.columns:
            frames = {p.stem: normalize_price_df(raw, p.stem)}
        else:
            if "symbol" not in raw.columns:
                raise ValueError("长表模式必须包含 symbol 列")
            frames = {str(s): normalize_price_df(g, str(s))
                      for s, g in raw.groupby("symbol", sort=False)}
    cal = calendar
    if cal is None:
        idx = pd.DatetimeIndex(sorted(set().union(*[f.index for f in frames.values()])))
        cal = idx
    return DataBundle(prices=frames, calendar=pd.DatetimeIndex(cal),
                      benchmark=benchmark, meta=None)


# --------------------------------------------------------------------------- #
# 在线数据源
# --------------------------------------------------------------------------- #
def fetch_akshare_bundle(symbols: Iterable[str], start: str = "2018-01-01", end: str = "2024-12-31",
                         adjust: str = "qfq", benchmark: str = "000300",
                         calendar: Optional[pd.DatetimeIndex] = None) -> DataBundle:
    """通过 AkShare 抓取 A 股日线（前复权）与指数基准。

    依赖：``pip install akshare``。首次抓取较慢，建议先用 ``save_bundle`` 缓存。
    """
    try:
        import akshare as ak
    except ImportError as exc:  # pragma: no cover
        raise ImportError("需要先安装 akshare：pip install akshare") from exc

    s = pd.Timestamp(start).strftime("%Y%m%d")
    e = pd.Timestamp(end).strftime("%Y%m%d")
    frames: Dict[str, pd.DataFrame] = {}
    meta_rows = []
    errors = []
    for sym in symbols:
        code = "".join(ch for ch in str(sym) if ch.isdigit())[:6]
        try:
            raw = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=s, end_date=e,
                                     adjust=adjust)
            df = normalize_price_df(raw, code)
            frames[code] = df
            meta_rows.append({"symbol": code, "name": code, "industry": "未分类",
                              "board": get_board(code), "is_st": False})
        except Exception as exc:
            errors.append(f"{code}: {exc}")
    if not frames:
        raise RuntimeError("AkShare 未返回任何数据；" + "; ".join(errors[:3]))

    bench_series = None
    try:
        braw = ak.index_zh_a_hist(symbol=benchmark, period="daily", start_date=s, end_date=e)
        braw = braw.rename(columns={"日期": "date", "收盘": "close"})
        braw["date"] = pd.to_datetime(braw["date"])
        bench_series = braw.set_index("date")["close"].astype(float)
        bench_series.name = benchmark
    except Exception:
        pass

    if calendar is None:
        cal = pd.DatetimeIndex(sorted(set().union(*[f.index for f in frames.values()])))
    else:
        cal = pd.DatetimeIndex(calendar)
    meta = pd.DataFrame(meta_rows).set_index("symbol") if meta_rows else None
    return DataBundle(prices=frames, calendar=cal, benchmark=bench_series,
                      benchmark_name=benchmark, meta=meta)


def fetch_tushare_bundle(symbols: Iterable[str], start: str = "2018-01-01", end: str = "2024-12-31",
                         token: Optional[str] = None, benchmark: str = "000300",
                         adj: str = "qfq") -> DataBundle:
    """通过 TuShare Pro 抓取（需 token，或设置环境变量 TUSHARE_TOKEN）。"""
    try:
        import tushare as ts
    except ImportError as exc:  # pragma: no cover
        raise ImportError("需要先安装 tushare：pip install tushare") from exc
    token = token or os.environ.get("TUSHARE_TOKEN", "")
    if not token:
        raise ValueError("缺少 TuShare token（参数 token 或环境变量 TUSHARE_TOKEN）")
    pro = ts.pro_api(token)
    codes = []
    for sym in symbols:
        c = "".join(ch for ch in str(sym) if ch.isdigit())[:6]
        codes.append(f"{c}.SH" if c.startswith(("6", "9")) else f"{c}.SZ")
    frames, meta_rows = {}, []
    for code in codes:
        raw = ts.pro_bar(ts_code=code, adj=adj, start_date=pd.Timestamp(start).strftime("%Y%m%d"),
                         end_date=pd.Timestamp(end).strftime("%Y%m%d"), freq="D")
        if raw is None or raw.empty:
            continue
        raw = raw.rename(columns={"trade_date": "date", "vol": "volume"})
        raw["amount"] = raw["amount"] * 1000.0
        raw["volume"] = raw["volume"] * 100.0
        df = normalize_price_df(raw, code[:6])
        frames[code[:6]] = df
        meta_rows.append({"symbol": code[:6], "name": code[:6], "industry": "未分类",
                          "board": get_board(code[:6]), "is_st": False})
    if not frames:
        raise RuntimeError("TuShare 未返回数据，请检查 token 与权限")
    cal = pd.DatetimeIndex(sorted(set().union(*[f.index for f in frames.values()])))
    return DataBundle(prices=frames, calendar=cal, meta=pd.DataFrame(meta_rows).set_index("symbol"))


def fetch_symbol_list(board: str = "全部", exclude_st: bool = True) -> pd.DataFrame:
    """获取 A 股实时股票列表（含名称/行业/市值），用于构建选股池。"""
    try:
        import akshare as ak
    except ImportError as exc:  # pragma: no cover
        raise ImportError("需要先安装 akshare：pip install akshare") from exc
    df = ak.stock_zh_a_spot_em()
    ren = {"代码": "symbol", "名称": "name", "最新价": "close", "涨跌幅": "pct_chg",
           "成交额": "amount", "换手率": "turnover", "总市值": "total_mv", "流通市值": "float_mv"}
    df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    df["board"] = df["symbol"].map(get_board)
    if exclude_st:
        df = df[~df["name"].astype(str).str.contains("ST|退", case=False, na=False)]
    if board not in ("全部", "ALL", None):
        df = df[df["board"] == board]
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 缓存
# --------------------------------------------------------------------------- #
def save_bundle(bundle: DataBundle, path: Union[str, Path, None] = None) -> Path:
    path = Path(path) if path else DATA_DIR / f"bundle_{datetime.now():%Y%m%d_%H%M%S}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_bundle(path: Union[str, Path]) -> DataBundle:
    with open(Path(path), "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, DataBundle):
        raise TypeError("缓存文件不是 DataBundle")
    return obj


def ensure_bundle(bundle: Optional[DataBundle] = None, **kwargs) -> DataBundle:
    """便捷函数：没有数据时自动生成合成数据，保证程序永远能跑起来。"""
    if bundle is not None:
        return bundle
    return synthetic_bundle(**kwargs)