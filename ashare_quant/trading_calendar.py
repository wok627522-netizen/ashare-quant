"""交易日历：买入/卖出日期推算必须基于真实交易日（含节假日）。

优先级：AkShare 官方日历（缓存到本地） → 内置近似日历（工作日剔除法定节假日）。

所有对外函数都返回 ``pd.Timestamp``，与行情数据的 DatetimeIndex 对齐。
"""

from __future__ import annotations

from datetime import date, datetime
from functools import lru_cache
from typing import Optional, Union

import pandas as pd

from .config import DATA_DIR

__all__ = ["approx_calendar", "get_calendar", "refresh_calendar", "next_trade_date",
           "prev_trade_date", "is_trade_day", "last_trade_date", "trade_days_between",
           "calendar_source", "calendar_info"]

_CACHE_FILE = DATA_DIR / "trade_calendar.csv"
# 近似法定节假日（仅用于离线兜底）：元旦 1 天、劳动节 3 天、国庆 7 天；
# 清明/端午/中秋等按农历浮动，离线无法推算，建议联网刷新官方日历。
CN_HOLIDAYS = {(1, 1), (5, 1), (5, 2), (5, 3),
               (10, 1), (10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7)}
_CAL_START, _CAL_END = "2005-01-01", "2035-12-31"
_SOURCE = "近似日历(工作日-节假日)"


def approx_calendar(start: Union[str, date] = _CAL_START,
                    end: Union[str, date] = _CAL_END) -> pd.DatetimeIndex:
    """近似交易日历：工作日剔除主要法定节假日（离线兜底方案）。"""
    days = pd.bdate_range(pd.Timestamp(start), pd.Timestamp(end))
    keep = [d for d in days if (d.month, d.day) not in CN_HOLIDAYS]
    return pd.DatetimeIndex(keep)


def _load_cache() -> Optional[pd.DatetimeIndex]:
    if not _CACHE_FILE.exists():
        return None
    try:
        df = pd.read_csv(_CACHE_FILE)
        col = "trade_date" if "trade_date" in df.columns else df.columns[0]
        return pd.DatetimeIndex(pd.to_datetime(df[col]).dropna()).normalize()
    except Exception:
        return None


def _save_cache(cal: pd.DatetimeIndex) -> None:
    try:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"trade_date": cal}).to_csv(_CACHE_FILE, index=False)
    except Exception:
        pass


@lru_cache(maxsize=4)
def _fetch_akshare_calendar() -> Optional[pd.DatetimeIndex]:
    """从 AkShare 拉取官方交易日历（失败返回 None，不抛异常）。"""
    try:
        import akshare as ak
    except ImportError:
        return None
    try:
        df = ak.tool_trade_date_hist_sina()
        col = "trade_date" if "trade_date" in df.columns else df.columns[0]
        cal = pd.DatetimeIndex(pd.to_datetime(df[col]).dropna()).normalize().unique()
        return pd.DatetimeIndex(cal).sort_values()
    except Exception:
        return None


_FULL_CAL_CACHE: dict = {"cal": None}      # 全量日历内存缓存（避免每次都读 CSV）


def _full_calendar(use_online: bool = True, force_refresh: bool = False) -> pd.DatetimeIndex:
    global _SOURCE
    if not force_refresh and _FULL_CAL_CACHE.get("cal") is not None:
        return _FULL_CAL_CACHE["cal"]                      # type: ignore[return-value]
    approx = approx_calendar(_CAL_START, _CAL_END)
    online: Optional[pd.DatetimeIndex] = None
    if force_refresh or not _CACHE_FILE.exists():
        if use_online:
            online = _fetch_akshare_calendar()
            if online is not None and len(online) > 100:
                _save_cache(online)
    if online is None:
        online = _load_cache()
    if online is not None and len(online) > 100:
        _SOURCE = "AkShare 官方交易日历"
        extra = approx[(approx > online.max()) | (approx < online.min())]
        cal = pd.DatetimeIndex(sorted(set(online).union(set(extra))))
    else:
        _SOURCE = "近似日历(工作日-节假日)"
        cal = approx
    _FULL_CAL_CACHE["cal"] = cal
    return cal


def get_calendar(start: Optional[Union[str, date]] = None,
                 end: Optional[Union[str, date]] = None,
                 use_online: bool = True, force_refresh: bool = False) -> pd.DatetimeIndex:
    """获取交易日历（官方优先，离线兜底），可指定区间；结果在内存中缓存。"""
    cal = _full_calendar(use_online=use_online, force_refresh=force_refresh)
    if start is not None:
        cal = cal[cal >= pd.Timestamp(start)]
    if end is not None:
        cal = cal[cal <= pd.Timestamp(end)]
    return pd.DatetimeIndex(cal)


def refresh_calendar() -> dict:
    """强制刷新官方交易日历（需要 akshare 与网络）。"""
    _FULL_CAL_CACHE["cal"] = None
    cal = get_calendar(force_refresh=True)
    return {"source": _SOURCE, "count": int(len(cal)), "start": str(cal[0].date()),
            "end": str(cal[-1].date()), "cache": str(_CACHE_FILE)}


def calendar_source() -> str:
    return _SOURCE


def calendar_info() -> dict:
    cal = get_calendar()
    return {"source": _SOURCE, "count": int(len(cal)), "start": str(cal[0].date()),
            "end": str(cal[-1].date()), "cache": str(_CACHE_FILE)}


def is_trade_day(d: Union[str, date, datetime], calendar: Optional[pd.DatetimeIndex] = None) -> bool:
    ts = pd.Timestamp(d).normalize()
    cal = calendar if calendar is not None else get_calendar()
    return bool(ts in set(cal))


def next_trade_date(d: Union[str, date, datetime], n: int = 1,
                    calendar: Optional[pd.DatetimeIndex] = None) -> pd.Timestamp:
    """返回 d **之后**第 n 个交易日（d 本身不算），自动跳过周末与节假日。"""
    ts = pd.Timestamp(d).normalize()
    cal = calendar if calendar is not None else get_calendar()
    future = cal[cal > ts]
    if len(future) >= n:
        return pd.Timestamp(future[n - 1])
    guess, found = ts, 0
    while found < n:
        guess += pd.Timedelta(days=1)
        if guess.weekday() < 5 and (guess.month, guess.day) not in CN_HOLIDAYS:
            found += 1
    return pd.Timestamp(guess)


def prev_trade_date(d: Union[str, date, datetime], n: int = 1,
                    calendar: Optional[pd.DatetimeIndex] = None) -> pd.Timestamp:
    """返回 d **之前**第 n 个交易日。"""
    ts = pd.Timestamp(d).normalize()
    cal = calendar if calendar is not None else get_calendar()
    past = cal[cal < ts]
    if len(past) >= n:
        return pd.Timestamp(past[-n])
    guess, found = ts, 0
    while found < n:
        guess -= pd.Timedelta(days=1)
        if guess.weekday() < 5 and (guess.month, guess.day) not in CN_HOLIDAYS:
            found += 1
    return pd.Timestamp(guess)


def last_trade_date(ref: Optional[Union[str, date, datetime]] = None,
                    calendar: Optional[pd.DatetimeIndex] = None) -> pd.Timestamp:
    """<= ref 的最后一个交易日（ref 缺省为今天）。"""
    ts = pd.Timestamp(ref or datetime.now()).normalize()
    cal = calendar if calendar is not None else get_calendar()
    past = cal[cal <= ts]
    if len(past):
        return pd.Timestamp(past[-1])
    return prev_trade_date(ts, 1, cal)


def trade_days_between(start: Union[str, date, datetime], end: Union[str, date, datetime],
                       calendar: Optional[pd.DatetimeIndex] = None) -> int:
    """两个日期之间的交易日数量（含首尾）。"""
    cal = calendar if calendar is not None else get_calendar()
    a, b = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if a > b:
        a, b = b, a
    return int(((cal >= a) & (cal <= b)).sum())


