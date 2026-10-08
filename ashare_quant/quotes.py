"""行情快照源：为实盘/模拟盘提供统一定价的 Quote。

优先级（``get_quotes`` 自动降级）：

1. **券商通道**（QMT 内置行情，最准）            —— 需要券商通道可用
2. **AkShare 全市场快照**（免费，约 3 秒级延迟）  —— 需要联网 + pip install akshare
3. **本地行情最后一根 K 线**（离线兜底）          —— 永远可用，但价格滞后

返回统一的 ``Quote`` 对象，并补算涨跌停价、停牌标记。
"""

from __future__ import annotations

import time
from dataclasses import asdict
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .data import DataBundle
from .execution.base import Broker, Quote
from .rules import board_limit, round_price

__all__ = ["quotes_from_bundle", "quotes_from_akshare", "get_quotes", "quote_frame",
           "clear_cache"]

_SNAPSHOT_CACHE: Dict[str, object] = {"ts": 0.0, "df": None}
CACHE_TTL = 20.0


def quotes_from_bundle(bundle: DataBundle, symbols: Optional[Sequence[str]] = None,
                       as_of: Optional[pd.Timestamp] = None) -> Dict[str, Quote]:
    """用本地行情最后一根（或指定日期的）K 线构造 Quote（离线兜底）。"""
    syms = list(symbols) if symbols is not None else list(bundle.symbols)
    out: Dict[str, Quote] = {}
    for s in syms:
        if s not in bundle.prices:
            continue
        df = bundle.get(s)
        if as_of is not None:
            df = df.loc[:pd.Timestamp(as_of)]
        if df.empty:
            continue
        row = df.iloc[-1]
        prev = float(row.get("prev_close", np.nan))
        if not np.isfinite(prev) or prev <= 0:
            prev = float(df["close"].iloc[-2]) if len(df) > 1 else float(row["close"])
        last = float(row["close"])
        st = bundle.is_st(s)
        pct = board_limit(s, st)
        out[s] = Quote(symbol=s, last=last, prev_close=prev, open=float(row.get("open", last)),
                       high=float(row.get("high", last)), low=float(row.get("low", last)),
                       volume=float(row.get("volume", 0) or 0),
                       amount=float(row.get("amount", 0) or 0),
                       limit_up=round_price(prev * (1 + pct)), limit_down=round_price(prev * (1 - pct)),
                       suspended=bool(row.get("suspended", False)),
                       name=bundle.name_of(s), ts=datetime.now(), source="本地行情")
    return out


def _akshare_snapshot(force: bool = False) -> Optional[pd.DataFrame]:
    now = time.time()
    if not force and _SNAPSHOT_CACHE["df"] is not None and (now - float(_SNAPSHOT_CACHE["ts"])) < CACHE_TTL:
        return _SNAPSHOT_CACHE["df"]  # type: ignore[return-value]
    try:
        import akshare as ak
    except ImportError:
        return None
    try:
        df = ak.stock_zh_a_spot_em()
        df.columns = [str(c) for c in df.columns]
        if "代码" in df.columns:
            df["代码"] = df["代码"].astype(str).str.zfill(6)
        _SNAPSHOT_CACHE.update({"ts": now, "df": df})
        return df
    except Exception:
        return None


def quotes_from_akshare(symbols: Sequence[str], force: bool = False) -> Dict[str, Quote]:
    """AkShare 全市场快照 → 指定代码的 Quote（含涨跌停价与停牌判断）。"""
    df = _akshare_snapshot(force=force)
    if df is None or df.empty:
        return {}
    wanted = {str(s).zfill(6) for s in symbols}
    sub = df[df["代码"].isin(wanted)] if "代码" in df.columns else pd.DataFrame()
    out: Dict[str, Quote] = {}
    for _, r in sub.iterrows():
        sym = str(r["代码"]).zfill(6)
        def num(key, default=0.0):
            try:
                v = float(r.get(key, default))
                return v if np.isfinite(v) else default
            except Exception:
                return default
        last = num("最新价")
        prev = num("昨收")
        if last <= 0 or prev <= 0:
            continue
        pct = board_limit(sym, False)
        vol = num("成交量")
        out[sym] = Quote(symbol=sym, last=last, prev_close=prev, open=num("今开", last),
                         high=num("最高", last), low=num("最低", last), volume=vol,
                         amount=num("成交额"), limit_up=round_price(prev * (1 + pct)),
                         limit_down=round_price(prev * (1 - pct)),
                         suspended=(vol <= 0 or last <= 0),
                         name=str(r.get("名称", "")), ts=datetime.now(), source="AkShare")
    return out


def get_quotes(symbols: Sequence[str], broker: Optional[Broker] = None,
               bundle: Optional[DataBundle] = None, as_of: Optional[pd.Timestamp] = None,
               use_akshare: bool = True, verbose: bool = False) -> Dict[str, Quote]:
    """按优先级获取行情，任何一级失败自动降级。"""
    syms = [str(s).zfill(6) for s in symbols]
    info: Dict[str, str] = {}
    merged: Dict[str, Quote] = {}
    if broker is not None and getattr(broker, "supports_realtime_quote", False):
        try:
            q = broker.quotes(syms)
            if q:
                merged.update(q)
                info["券商通道"] = f"{len(q)} 只"
        except Exception as exc:
            info["券商通道"] = f"失败：{exc}"
    missing = [s for s in syms if s not in merged]
    if missing and use_akshare:
        try:
            q = quotes_from_akshare(missing)
            if q:
                merged.update(q)
                info["AkShare"] = f"{len(q)} 只"
            else:
                info["AkShare"] = "无数据"
        except Exception as exc:
            info["AkShare"] = f"失败：{exc}"
    missing = [s for s in syms if s not in merged]
    if missing and bundle is not None:
        try:
            q = quotes_from_bundle(bundle, missing, as_of=as_of)
            merged.update(q)
            info["本地行情"] = f"{len(q)} 只"
        except Exception as exc:
            info["本地行情"] = f"失败：{exc}"
    if verbose:
        for k, v in info.items():
            print(f"    行情来源 {k}: {v}")
    return merged


def quote_frame(quotes: Dict[str, Quote]) -> pd.DataFrame:
    """Quote 字典 → 表格（界面展示）。"""
    if not quotes:
        return pd.DataFrame(columns=["代码", "名称", "最新价", "涨跌幅", "昨收", "涨停价", "跌停价", "状态", "来源"])
    rows = []
    for s, q in quotes.items():
        rows.append({"代码": s, "名称": q.name or s, "最新价": round(q.last, 3),
                     "涨跌幅": q.chg, "昨收": q.prev_close, "今开": q.open,
                     "最高": q.high, "最低": q.low, "成交额(万)": round(q.amount / 1e4, 1),
                     "涨停价": q.limit_up, "跌停价": q.limit_down,
                     "状态": "停牌" if q.suspended else ("涨停" if q.at_limit_up else ("跌停" if q.at_limit_down else "正常")),
                     "来源": q.source})
    return pd.DataFrame(rows)


def clear_cache() -> None:
    _SNAPSHOT_CACHE.update({"ts": 0.0, "df": None})
