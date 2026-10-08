"""实时行情数据源（东方财富直连，绕过系统代理）。

为什么不用 AkShare 的现成接口？
--------------------------------
* AkShare 会走系统代理，某些环境下会直接失败；本模块用 ``session.trust_env = False``
  强制直连，实测可用；
* 只需要两个接口就能覆盖实时选股的全部需求：**批量快照**（最新价/涨跌幅/成交额/时间戳）
  与 **日线 K 线**（前复权，供因子计算）；
* 不依赖 akshare，安装更轻、请求更快（批量取价一次请求可拿多只）。

接口
----
* 批量快照：``push2.eastmoney.com/api/qt/ulist.np/get``
* 日线 K 线：``push2his.eastmoney.com/api/qt/stock/kline/get``（fqt=1 前复权）

实时性说明
----------
``Quote.ts`` 使用东财返回的时间戳（``f86``），盘中为**当前时刻**，休市为**最后成交时间**。
``market_status()`` 会告诉你现在是"连续竞价（实时）"还是"已收盘/休市"，
选股结果的买入日期会据此给出 **今日盘中** 或 **下一交易日开盘**。
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, time as dtime
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import DATA_DIR, DEFAULT_POOL
from .data import DataBundle
from .execution.base import Quote
from .http_util import MarketDataClient
from .rules import board_limit, round_price
from .trading_calendar import get_calendar, next_trade_date, prev_trade_date

__all__ = ["secid", "fetch_spot", "fetch_spot_frame", "fetch_daily", "fetch_daily_many",
           "fetch_index_daily", "fetch_realtime_bundle", "market_status", "RealtimeStatus",
           "is_market_open", "SNAPSHOT_FIELDS"]

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120 Safari/537.36")
# 同一接口准备 http/https 两个入口：部分网络环境下代理只允许其中一种
SPOT_URLS = ("https://push2.eastmoney.com/api/qt/ulist.np/get",
             "http://push2.eastmoney.com/api/qt/ulist.np/get")
KLINE_URLS = ("http://push2his.eastmoney.com/api/qt/stock/kline/get",
              "https://push2his.eastmoney.com/api/qt/stock/kline/get")
CLIST_URLS = ("https://push2.eastmoney.com/api/qt/clist/get",
              "http://push2.eastmoney.com/api/qt/clist/get")
SPOT_URL = SPOT_URLS[0]
KLINE_URL = KLINE_URLS[0]
KLINE_UT = "fa5fd1943c7b386f172d6893dbfba10b"
SINA_KLINE = "https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData"


_URL_PREF: Dict[str, int] = {}          # 记录每个接口最近可用的入口，避免每次都先试坏的那个


def _request_any(client: "MarketDataClient", urls, params: dict, timeout: float,
                 key: Optional[str] = None):
    """依次尝试多个 URL（http/https），返回第一个成功的 JSON，并记住可用入口。"""
    key = key or urls[0]
    order = list(range(len(urls)))
    pref = _URL_PREF.get(key)
    if pref in order:
        order.remove(pref)
        order.insert(0, pref)
    last = None
    for pos, idx in enumerate(order):
        try:
            js = client.get_json(urls[idx], params=params,
                                 timeout=timeout if pos == 0 else max(6.0, timeout * 0.5))
            _URL_PREF[key] = idx
            return js
        except Exception as exc:
            last = exc
    raise RuntimeError(f"所有行情入口均失败：{last}")
SNAPSHOT_FIELDS = "f12,f14,f2,f3,f4,f5,f6,f15,f16,f17,f18,f86"


def _client(timeout: float = 15.0, retry: int = 2) -> MarketDataClient:
    """自适应代理的行情客户端：代理优先（Clash 等），失败回退直连。"""
    return MarketDataClient(timeout=timeout, retry=retry)


# 指数代码 → 东财 secid（指数的市场前缀与个股不同，必须显式映射）
INDEX_SECIDS = {
    "000001": "1.000001",   # 上证指数
    "000300": "1.000300",   # 沪深300
    "000905": "1.000905",   # 中证500
    "000852": "1.000852",   # 中证1000
    "000016": "1.000016",   # 上证50
    "399006": "0.399006",   # 创业板指
    "399001": "0.399001",   # 深证成指
}


def secid(symbol: str, is_index: bool = False) -> str:
    """6 位代码 → 东财 secid：600519 → 1.600519，000001(个股) → 0.000001。

    ``is_index=True`` 时按指数映射（例如沪深300 = 1.000300）。
    """
    s = "".join(ch for ch in str(symbol) if ch.isdigit())[:6]
    if not s:
        raise ValueError(f"非法代码：{symbol}")
    if is_index:
        return INDEX_SECIDS.get(s, f"1.{s}")
    return f"1.{s}" if s.startswith(("6", "9")) else f"0.{s}"


# --------------------------------------------------------------------------- #
# 市场状态
# --------------------------------------------------------------------------- #
@dataclass
class RealtimeStatus:
    """当前市场状态与行情时效说明。"""

    now: datetime
    is_trade_day: bool
    is_open: bool                 # 是否处于连续竞价（可实时交易）
    session: str                  # 连续竞价（实时）/ 集合竞价 / 午间休市 / 已收盘 / 休市
    last_trade_date: pd.Timestamp
    next_trade_date: pd.Timestamp
    live: bool                    # 报价是否为实时（盘中）
    note: str = ""

    def to_dict(self) -> dict:
        return {"现在": self.now.strftime("%Y-%m-%d %H:%M:%S"),
                "交易日": "是" if self.is_trade_day else "否",
                "是否盘中": "是" if self.is_open else "否", "时段": self.session,
                "最近交易日": str(self.last_trade_date.date()),
                "下一交易日": str(self.next_trade_date.date()),
                "行情时效": "实时" if self.live else "最近收盘", "说明": self.note}


def market_status(now: Optional[datetime] = None,
                  calendar: Optional[pd.DatetimeIndex] = None) -> RealtimeStatus:
    """判断当前 A 股市场状态与行情时效。"""
    now = now or datetime.now()
    cal = calendar if calendar is not None else get_calendar()
    today = pd.Timestamp(now.date())
    is_td = today in set(cal)
    t = now.time()
    is_open, session = False, "休市（非交易日）"
    if is_td:
        if dtime(9, 30) <= t <= dtime(11, 30) or dtime(13, 0) <= t <= dtime(15, 0):
            session, is_open = "连续竞价（实时）", True
        elif dtime(9, 15) <= t < dtime(9, 25):
            session = "开盘集合竞价"
        elif dtime(11, 30) < t < dtime(13, 0):
            session = "午间休市"
        elif t < dtime(9, 15):
            session = "未开盘"
        elif dtime(14, 57) <= t <= dtime(15, 0):
            session = "收盘集合竞价"
        else:
            session = "已收盘"
    if is_td and (is_open or t >= dtime(9, 15)):
        ltd = today
    else:
        ltd = prev_trade_date(today, 1, cal)
    if is_td and is_open:
        ntd = next_trade_date(today, 1, cal)
    else:
        ntd = next_trade_date(max(today, pd.Timestamp(ltd)), 1, cal)
    if not is_td:
        note = f"今天休市，行情为最近交易日 {ltd.date()} 的收盘数据；下一个交易日 {ntd.date()}"
    elif not is_open:
        note = f"当前{session}，行情为最近成交数据；下一个交易日 {ntd.date()}"
    else:
        note = "行情为盘中实时数据，可在交易时段直接执行买入（买入日期=今天）"
    return RealtimeStatus(now=now, is_trade_day=is_td, is_open=is_open, session=session,
                          last_trade_date=pd.Timestamp(ltd), next_trade_date=pd.Timestamp(ntd),
                          live=bool(is_open), note=note)


def is_market_open(now: Optional[datetime] = None) -> bool:
    return market_status(now).is_open


# --------------------------------------------------------------------------- #
# 批量实时快照
# --------------------------------------------------------------------------- #
def _all_symbols(sess) -> List[str]:
    """拉取全市场代码列表（沪深京）。"""
    codes: List[str] = []
    fs = "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048"
    params = {"pn": "1", "pz": "6000", "po": "1", "np": "1",
              "ut": "bd1d9ddb04089700cf9c27f6f7426281", "fltt": "2", "invt": "2",
              "fid": "f12", "fs": fs, "fields": "f12,f14"}
    try:
        js = _request_any(sess, CLIST_URLS, params, 25, key="clist")
        for it in ((js or {}).get("data") or {}).get("diff") or []:
            code = str(it.get("f12", "")).zfill(6)
            if code.isdigit():
                codes.append(code)
    except Exception:
        pass
    return sorted(set(codes))


def _quote_from_item(it: dict) -> Optional[Quote]:
    """东财字段 → Quote。"""
    try:
        sym = str(it.get("f12", "")).zfill(6)
        last = float(it.get("f2")) if it.get("f2") not in (None, "-") else 0.0
        prev = float(it.get("f18")) if it.get("f18") not in (None, "-") else 0.0

        def num(key, default=0.0):
            v = it.get(key)
            if v in (None, "-", ""):
                return default
            try:
                return float(v)
            except Exception:
                return default

        if not sym.isdigit() or last <= 0:
            return None
        pct = board_limit(sym, False)
        ts = None
        raw_ts = it.get("f86")
        try:                                  # f86 为行情时间戳；无效(0/缺失)时用当前时间
            v = int(raw_ts)
            if 1_000_000_000 < v < 4_000_000_000:
                ts = datetime.fromtimestamp(v)
        except Exception:
            ts = None
        vol = num("f5")                      # 单位：手
        return Quote(symbol=sym, last=last, prev_close=prev, open=num("f17", last),
                     high=num("f15", last), low=num("f16", last), volume=vol * 100.0,
                     amount=num("f6"), limit_up=round_price(prev * (1 + pct)) if prev else None,
                     limit_down=round_price(prev * (1 - pct)) if prev else None,
                     suspended=(vol <= 0), name=str(it.get("f14", "")),
                     ts=ts or datetime.now(), source="东财实时")
    except Exception:
        return None


_SPOT_CACHE: Dict[str, object] = {"ts": 0.0, "data": {}, "key": None}
SPOT_TTL = 15.0


def fetch_spot(symbols: Optional[Iterable[str]] = None, batch: int = 400,
               timeout: float = 15.0, retry: int = 2,
               progress_cb: Optional[Callable[[float], None]] = None,
               use_cache: bool = True) -> Dict[str, Quote]:
    """批量获取实时快照；``symbols=None`` 时取全市场（分页请求）。

    15 秒内存缓存，避免界面自动刷新时把行情站打爆。
    """
    cache_key = "ALL" if symbols is None else ",".join(sorted(str(c).zfill(6) for c in symbols))
    now = time.time()
    if use_cache and _SPOT_CACHE["key"] == cache_key and \
            (now - float(_SPOT_CACHE["ts"])) < SPOT_TTL:
        return dict(_SPOT_CACHE["data"])          # type: ignore[arg-type]
    s = MarketDataClient(timeout=timeout, retry=retry)
    if symbols is None:
        codes = _all_symbols(s)
    else:
        codes = [str(c).zfill(6) for c in symbols]
    result: Dict[str, Quote] = {}
    total = max(len(codes), 1)
    for i in range(0, len(codes), batch):
        chunk = codes[i:i + batch]
        params = {"fltt": "2", "invt": "2", "fields": SNAPSHOT_FIELDS,
                  "secids": ",".join(secid(c) for c in chunk), "pn": "1", "pz": str(batch),
                  "_": str(int(time.time() * 1000))}
        data = None
        for attempt in range(retry + 1):
            try:
                js = _request_any(s, SPOT_URLS, params, timeout, key="spot")
                data = (js or {}).get("data") or {}
                break
            except Exception:
                if attempt >= retry:
                    data = None
                time.sleep(0.4 * (attempt + 1))
        if not data:
            continue
        for it in (data.get("diff") or []):
            q = _quote_from_item(it)
            if q is not None:
                result[q.symbol] = q
        if progress_cb:
            progress_cb(min(1.0, (i + len(chunk)) / total))
    if use_cache and result:
        _SPOT_CACHE.update({"ts": now, "data": dict(result), "key": cache_key})
    return result


def fetch_spot_frame(symbols: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """实时快照 → 表格。"""
    from .quotes import quote_frame
    return quote_frame(fetch_spot(symbols))


# --------------------------------------------------------------------------- #
# 日线（前复权）
# --------------------------------------------------------------------------- #
CACHE_DIR = DATA_DIR / "realtime_cache"


def _cache_file(symbol: str, days: int, adjust: int):
    return CACHE_DIR / f"{symbol}_{days}_{adjust}.pkl"


_MARKET_OPEN_CACHE: dict = {"ts": 0.0, "val": False}


def _market_open_cached(ttl: float = 30.0) -> bool:
    """是否盘中（30 秒内存缓存，避免批量读取缓存文件时反复解析交易日历）。"""
    now = time.time()
    if now - float(_MARKET_OPEN_CACHE["ts"]) > ttl:
        try:
            _MARKET_OPEN_CACHE["val"] = bool(is_market_open())
        except Exception:
            _MARKET_OPEN_CACHE["val"] = False
        _MARKET_OPEN_CACHE["ts"] = now
    return bool(_MARKET_OPEN_CACHE["val"])


def _cache_valid(path, ttl: Optional[float] = None) -> bool:
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    if ttl is None:
        ttl = 300.0 if _market_open_cached() else 6 * 3600.0   # 盘中 5 分钟，收盘后 6 小时
    return age < ttl



# ---------------- 东财接口熔断（避免慢接口拖垮全市场抓取） ----------------
_EM_FAILS = 0
_EM_DOWN_UNTIL = 0.0
_EM_FAIL_LIMIT = 3
_EM_COOLDOWN = 600.0


def _em_available() -> bool:
    return time.time() >= _EM_DOWN_UNTIL


def _em_note(success: bool) -> None:
    global _EM_FAILS, _EM_DOWN_UNTIL
    if success:
        _EM_FAILS = 0
        return
    _EM_FAILS += 1
    if _EM_FAILS >= _EM_FAIL_LIMIT:
        _EM_DOWN_UNTIL = time.time() + _EM_COOLDOWN
        _EM_FAILS = 0


def fetch_daily(symbol: str, days: int = 500, adjust: int = 1, timeout: float = 20.0,
                retry: int = 2, session=None, is_index: bool = False,
                use_cache: bool = True) -> pd.DataFrame:
    """取单只股票日线（默认前复权），索引为日期，列为 OHLCV + amount。

    ``adjust``：1=前复权，2=后复权，0=不复权。
    结果按"盘中 5 分钟 / 收盘后 6 小时"缓存到 ``data_cache/realtime_cache``，
    既提速也降低被行情站限流的概率。
    """
    cp = _cache_file(symbol, days, adjust)
    if use_cache and _cache_valid(cp):
        try:
            return pd.read_pickle(cp)
        except Exception:
            pass
    s = session if isinstance(session, MarketDataClient) else MarketDataClient(timeout=timeout, retry=retry)
    begin = (pd.Timestamp.today() - pd.Timedelta(days=int(days * 1.7) + 60)).strftime("%Y%m%d")
    params = {"secid": secid(symbol, is_index=is_index), "ut": KLINE_UT,
              "fields1": "f1,f2,f3,f4,f5,f6",
              "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
              "klt": "101", "fqt": str(adjust), "beg": begin, "end": "20500101",
              "_": str(int(time.time() * 1000))}
    kl: List[str] = []
    for attempt in range(retry + 1):
        try:
            js = _request_any(s, KLINE_URLS, params, timeout, key="kline")
            kl = ((js or {}).get("data") or {}).get("klines") or []
            break
        except Exception:
            if attempt >= retry:
                kl = []
                break
            time.sleep(0.4 * (attempt + 1))
    if not kl:
        # 东财全部入口失败 → 用新浪日线兜底（不复权）
        out = _fetch_daily_sina(symbol, days=days, client=s, timeout=timeout, is_index=is_index)
        if not out.empty and use_cache:
            try:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                out.to_pickle(cp)
            except Exception:
                pass
        return out
    rows = [x.split(",") for x in kl]
    df = pd.DataFrame([r[:11] for r in rows],
                      columns=["date", "open", "close", "high", "low", "volume",
                               "amount", "amplitude", "pct_chg", "chg", "turnover"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["volume"] = df["volume"] * 100.0          # 手 → 股
    df["prev_close"] = df["close"].shift(1)
    if len(df):
        df.loc[df.index[0], "prev_close"] = df["open"].iloc[0]
    df["suspended"] = df["volume"].fillna(0) <= 0
    out = df.tail(int(days)).copy()
    pct = board_limit(symbol, False)
    out["limit_up"] = np.round(out["prev_close"] * (1 + pct), 2)
    out["limit_down"] = np.round(out["prev_close"] * (1 - pct), 2)
    if use_cache:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            out.to_pickle(cp)
        except Exception:
            pass
    return out


def _fetch_daily_sina(symbol: str, days: int = 500, client=None, timeout: float = 20.0,
                      is_index: bool = False) -> pd.DataFrame:
    """新浪日线兜底（不复权）。字段：day/open/high/low/close/volume。"""
    s = client if isinstance(client, MarketDataClient) else MarketDataClient(timeout=timeout)
    s6 = "".join(ch for ch in str(symbol) if ch.isdigit())[:6]
    if is_index:
        prefix = "sh" if INDEX_SECIDS.get(s6, "1.").startswith("1.") else "sz"
    else:
        prefix = "sh" if s6.startswith(("6", "9")) else ("bj" if s6.startswith(("4", "8", "92")) else "sz")
    try:
        js = s.get_json(SINA_KLINE, params={"symbol": f"{prefix}{s6}", "scale": "240",
                                            "ma": "no", "datalen": str(min(int(days), 1023))},
                        timeout=timeout,
                        headers={"Referer": "https://finance.sina.com.cn"})
    except Exception:
        return pd.DataFrame()
    if not isinstance(js, list) or not js:
        return pd.DataFrame()
    df = pd.DataFrame(js)
    if "day" not in df.columns:
        return pd.DataFrame()
    df = df.rename(columns={"day": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    for c in ("open", "high", "low", "close", "volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "close" not in df.columns or df["close"].dropna().empty:
        return pd.DataFrame()
    df["amount"] = df.get("volume", 0) * df["close"]
    df = df[[c for c in ["open", "high", "low", "close", "volume", "amount"] if c in df.columns]]
    df["prev_close"] = df["close"].shift(1)
    if len(df):
        df.loc[df.index[0], "prev_close"] = df["open"].iloc[0]
    df["suspended"] = df["volume"].fillna(0) <= 0
    pct = board_limit(s6, False)
    df["limit_up"] = np.round(df["prev_close"] * (1 + pct), 2)
    df["limit_down"] = np.round(df["prev_close"] * (1 - pct), 2)
    return df.tail(int(days)).copy()


def fetch_daily_many(symbols: Sequence[str], days: int = 500, workers: int = 4,
                     progress_cb: Optional[Callable[[float], None]] = None,
                     warmup: bool = True,
                     kline_source: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """并发抓取多只股票日线。

    注意：行情站对"同域并发"较敏感，且首次请求需要先探明 http/https 哪个入口可用，
    因此这里默认先用**单只热身请求**确定入口，再以较低的并发度抓取（实测比高并发快得多）。
    """
    out: Dict[str, pd.DataFrame] = {}
    syms = [str(s).zfill(6) for s in symbols]
    total = max(len(syms), 1)
    if warmup and syms and _URL_PREF.get("kline") is None:
        try:
            first = fetch_daily(syms[0], days=days)
            if not first.empty:
                out[syms[0]] = first
        except Exception:
            pass
        if progress_cb:
            progress_cb(1.0 / total)
    rest = [s for s in syms if s not in out]
    n_sess = max(1, min(int(workers), 6))
    clients = [MarketDataClient() for _ in range(n_sess)]

    def work(pair):
        i, sym = pair
        return sym, fetch_daily(sym, days=days, session=clients[i % n_sess],
                                source=kline_source)
    with ThreadPoolExecutor(max_workers=n_sess) as ex:
        futs = {ex.submit(work, (i, s)): s for i, s in enumerate(rest)}
        done0 = len(out)
        for done, fut in enumerate(as_completed(futs), start=1):
            done = done + done0
            try:
                sym, df = fut.result()
                if df is not None and not df.empty:
                    out[sym] = df
            except Exception:
                pass
            if progress_cb:
                progress_cb(done / total)
    return out


def fetch_index_daily(code: str = "000300", days: int = 500) -> pd.Series:
    """取指数日线收盘价（用于大盘择时）。"""
    df = fetch_daily(code, days=days, adjust=0, is_index=True)
    return df["close"] if not df.empty else pd.Series(dtype=float)


# --------------------------------------------------------------------------- #
# 组装实时 DataBundle
# --------------------------------------------------------------------------- #
def fetch_realtime_bundle(symbols: Optional[Sequence[str]] = None, days: int = 500,
                          benchmark: str = "000300", workers: int = 8,
                          max_symbols: int = 5000, include_benchmark: bool = True,
                          progress_cb: Optional[Callable[[float], None]] = None,
                          pool: Optional[str] = None, pool_limit: Optional[int] = None,
                          exclude_st: bool = True, kline_source: Optional[str] = None
                          ) -> Tuple[DataBundle, RealtimeStatus, Dict[str, Quote]]:
    """构建**实时** DataBundle：历史日线（前复权）+ 实时快照价。

    实时价会覆盖"当前有效交易日"那根 K 线的 open/close/high/low，
    因此所有因子（动量、波动、趋势）都按**当前价格**计算 —— 这是实时选股的数据基础。

    Returns
    -------
    (bundle, status, quotes)
    """
    status = market_status()
    if symbols is None and pool:
        # 指定股票池类型：main_board（全部沪深主板）/ active（主板活跃前 N）/ all_a ...
        from .universe import build_pool
        info = build_pool(pool, limit=pool_limit, exclude_st=exclude_st,
                          progress_cb=(lambda p: progress_cb(p * 0.1)) if progress_cb else None)
        symbols = list(info.symbols)
    if symbols is None:
        # 未指定：取"成交额最大"的活跃股票池
        symbols = resolve_pool(None, limit=max_symbols)
    symbols = [str(s).zfill(6) for s in symbols][:max_symbols]
    spot = fetch_spot(symbols)
    if not spot:
        raise RuntimeError(
            "未能获取实时行情。请检查网络与代理设置；"
            "可运行 network_diagnose() 诊断，或用环境变量 ASHARE_PROXY_MODE=proxy|direct 固定方式")
    symbols = [s for s in symbols if s in spot]
    if not symbols:
        raise RuntimeError("实时行情未返回任何请求的代码")

    def cb(p):
        if progress_cb:
            progress_cb(0.1 + min(0.88, float(p) * 0.88))

    n_workers = int(workers)
    if len(symbols) > 500:
        n_workers = max(n_workers, 12)
    daily = fetch_daily_many(symbols, days=days, workers=n_workers, progress_cb=cb,
                             kline_source=kline_source)
    if not daily:
        raise RuntimeError("未能获取历史日线数据")

    effective_day = pd.Timestamp(status.now.date()) if status.is_open else pd.Timestamp(status.last_trade_date)
    prices: Dict[str, pd.DataFrame] = {}
    meta_rows = []
    for sym, df in daily.items():
        q = spot.get(sym)
        if q is None or df.empty:
            continue
        d = df.copy()
        px = float(q.last)
        prev = float(q.prev_close) if q.prev_close else float(d["close"].iloc[-1])
        last_day = pd.Timestamp(d.index[-1])
        if last_day <= effective_day:
            target = effective_day
            if target not in d.index:
                d.loc[target] = d.iloc[-1].copy()
                d = d.sort_index()
            i = d.index.get_loc(target)
            d.iloc[i, d.columns.get_loc("close")] = px
            d.iloc[i, d.columns.get_loc("high")] = max(float(q.high or px), px)
            d.iloc[i, d.columns.get_loc("low")] = min(float(q.low or px), px)
            d.iloc[i, d.columns.get_loc("open")] = float(q.open or px)
            if "amount" in d.columns and q.amount:
                d.iloc[i, d.columns.get_loc("amount")] = float(q.amount)
            d["prev_close"] = d["close"].shift(1)
            d.iloc[i, d.columns.get_loc("prev_close")] = prev
            pct = board_limit(sym, False)
            d["limit_up"] = np.round(d["prev_close"] * (1 + pct), 2)
            d["limit_down"] = np.round(d["prev_close"] * (1 - pct), 2)
        prices[sym] = d
        meta_rows.append({"symbol": sym, "name": q.name or sym, "industry": "未分类",
                          "board": "MAIN", "is_st": "ST" in str(q.name).upper(),
                          "last_price": px, "quote_time": q.ts})

    if not prices:
        raise RuntimeError("组装实时行情失败")
    cal = pd.DatetimeIndex(sorted(set().union(*[df.index for df in prices.values()])))
    bench_series = None
    if include_benchmark:
        try:
            bench_series = fetch_index_daily(benchmark, days=days)
            bq = fetch_spot([benchmark], indices=True).get(benchmark)
            if bench_series is not None and len(bench_series) and bq is not None:
                bench_series.loc[effective_day] = bq.last
                bench_series = bench_series.sort_index()
        except Exception:
            bench_series = None
    meta = pd.DataFrame(meta_rows).set_index("symbol") if meta_rows else None
    bundle = DataBundle(prices=prices, calendar=cal, benchmark=bench_series,
                        benchmark_name="沪深300(实时)", meta=meta)
    # 行情时间：盘中 = 抓取时刻；休市/收盘 = 最后成交日 15:00（价格所属时间）
    q_ts = spot[next(iter(spot))].ts
    if not status.is_open:
        q_ts = pd.Timestamp(effective_day) + pd.Timedelta(hours=15)
    bundle.quote_time = q_ts                                # type: ignore[attr-defined]
    bundle.quote_source = "东财实时"                          # type: ignore[attr-defined]
    bundle.realtime_status = status                         # type: ignore[attr-defined]
    return bundle, status, spot










# --------------------------------------------------------------------------- #
# 数据源优先级：新浪优先（快），东财兜底/前复权
# --------------------------------------------------------------------------- #
# 实测：新浪日线 ~0.4s/只（8 只并发 0.6s），东财日线较慢且易被限流（502）。
# 因此默认用新浪，东财仅在需要前复权或新浪失败时使用。
# 可用环境变量 ASHARE_KLINE_SOURCE 固定：sina / eastmoney / auto
# 日线数据源策略（实测：新浪 0.5 秒/只，东财 6.4 秒/只但支持前复权）：
#   - 默认 sina：全市场 3000+ 只必须用它，否则要 20 分钟；
#   - 做因子研究 / IC 分析时用 eastmoney（前复权，消除分红送股假跳空），
#     通过 fetch_realtime_bundle(..., kline_source="eastmoney") 显式指定；
#   - 环境变量 ASHARE_KLINE_SOURCE 可强制覆盖。
KLINE_SOURCE = str(os.environ.get("ASHARE_KLINE_SOURCE", "sina")).strip().lower()
_EM_FETCH_DAILY = fetch_daily          # 保存原东财实现


def _cache_file_src(symbol: str, days: int, adjust: int, src: str):
    return CACHE_DIR / f"{symbol}_{days}_{adjust}_{src}.pkl"


def fetch_daily(symbol: str, days: int = 500, adjust: int = 1, timeout: float = 20.0,
                retry: int = 2, session=None, is_index: bool = False,
                use_cache: bool = True, source: Optional[str] = None) -> pd.DataFrame:
    """日线行情（统一入口）。

    * 默认 ``source='sina'``：新浪日线，**不复权**，速度最快，适合实时选股；
    * ``source='eastmoney'``：东财日线，支持前复权（``adjust=1``），较慢；
    * ``source='auto'``：先新浪，失败再用东财。
    """
    src = str(source or KLINE_SOURCE or "sina").lower()
    if src not in ("sina", "eastmoney", "auto"):
        src = "sina"
    order = ["sina", "eastmoney"] if src == "auto" else (
        ["sina", "eastmoney"] if src == "sina" else ["eastmoney", "sina"])
    cp = _cache_file_src(symbol, days, adjust, src)
    if use_cache and _cache_valid(cp):
        try:
            return pd.read_pickle(cp)
        except Exception:
            pass
    s = session if isinstance(session, MarketDataClient) else MarketDataClient(timeout=timeout, retry=retry)
    for name in order:
        if name == "eastmoney" and not _em_available():
            continue                      # 东财熔断中，直接用新浪
        try:
            if name == "sina":
                df = _fetch_daily_sina(symbol, days=days, client=s, timeout=timeout, is_index=is_index)
                adjusted = "none"
            else:
                df = _EM_FETCH_DAILY(symbol, days=days, adjust=adjust, timeout=timeout,
                                     retry=retry, session=s, is_index=is_index, use_cache=False)
                adjusted = "qfq" if adjust == 1 else ("hfq" if adjust == 2 else "none")
        except Exception:
            if name == "eastmoney":
                _em_note(False)
            df = pd.DataFrame()
        if df is not None and not df.empty:
            _em_note(name != "eastmoney" or True)   # 成功：重置失败计数
            try:
                df = df.copy()
                df.attrs["source"] = name
                df.attrs["adjusted"] = adjusted
            except Exception:
                pass
            if use_cache:
                try:
                    CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    df.to_pickle(cp)
                except Exception:
                    pass
            return df
    return pd.DataFrame()



# --------------------------------------------------------------------------- #
# 新浪实时快照（hq.sinajs.cn）：批量、极快，作为快照首选；东财为兜底
# --------------------------------------------------------------------------- #
SINA_HQ = "https://hq.sinajs.cn/list="
SPOT_SOURCE = str(os.environ.get("ASHARE_SPOT_SOURCE", "sina")).strip().lower()
_EM_FETCH_SPOT = fetch_spot


def _sina_code(symbol: str, is_index: bool = False) -> str:
    """新浪代码前缀：sh/sz/bj。

    ⚠️ 只有显式 is_index=True 时才按指数映射 —— 因为 000001 既是上证指数、
    也是平安银行的代码，混用会把个股取成指数价（历史 bug）。
    """
    s6 = "".join(ch for ch in str(symbol) if ch.isdigit())[:6]
    if is_index:
        return ("sh" if INDEX_SECIDS.get(s6, "1.").startswith("1.") else "sz") + s6
    if s6.startswith(("6", "9")):
        return "sh" + s6
    if s6.startswith(("4", "8", "92")):
        return "bj" + s6
    return "sz" + s6


def fetch_spot_sina(symbols: Sequence[str], timeout: float = 10.0, batch: int = 400,
                    client=None, indices: bool = False) -> Dict[str, Quote]:
    """新浪批量快照：``hq.sinajs.cn``，一次请求可取数百只，速度极快。"""
    import re
    s = client if isinstance(client, MarketDataClient) else MarketDataClient(timeout=timeout, retry=1)
    syms = [str(x).zfill(6) for x in symbols]
    out: Dict[str, Quote] = {}
    # 指数（000300/000001/399006…）走指数代码映射，避免拼成 sz000300 这类错误代码
    for i in range(0, len(syms), batch):
        chunk = syms[i:i + batch]
        url = SINA_HQ + ",".join(_sina_code(c, is_index=indices) for c in chunk)
        try:
            text = s.get_text(url, timeout=timeout, encoding="gbk",
                              headers={"Referer": "https://finance.sina.com.cn",
                                       "User-Agent": "Mozilla/5.0"})
        except Exception:
            continue
        for line in str(text).splitlines():
            m = re.search(r'hq_str_([a-z]{2})(\d{6})="([^"]*)"', line)
            if not m:
                continue
            code = m.group(2)
            parts = m.group(3).split(",")
            if len(parts) < 32:
                continue

            def num(idx, default=0.0):
                try:
                    v = float(parts[idx])
                    return v if np.isfinite(v) else default
                except Exception:
                    return default

            name = parts[0]
            open_, prev, last = num(1), num(2), num(3)
            high, low = num(4), num(5)
            vol, amount = num(8), num(9)
            suspended = False
            if last <= 0:                     # 停牌时新浪现价为 0
                last, suspended = (prev if prev > 0 else float("nan")), True
            if not np.isfinite(last) or last <= 0:
                continue
            ts = None
            try:
                ts = pd.Timestamp(f"{parts[30]} {parts[31]}").to_pydatetime()
            except Exception:
                ts = None
            pct = board_limit(code, False)
            out[code] = Quote(symbol=code, last=float(last), prev_close=float(prev), open=float(open_),
                              high=float(high), low=float(low), volume=float(vol), amount=float(amount),
                              limit_up=round_price(prev * (1 + pct)) if prev else None,
                              limit_down=round_price(prev * (1 - pct)) if prev else None,
                              suspended=bool(suspended or vol <= 0), name=name,
                              ts=ts or datetime.now(), source="新浪实时")
    return out


def fetch_spot(symbols, batch: int = 400, timeout: float = 15.0, retry: int = 2,
               progress_cb=None, use_cache: bool = True, source: Optional[str] = None,
               indices: bool = False) -> Dict[str, Quote]:
    """实时快照统一入口：默认新浪（快），失败回退东财。"""
    src = str(source or SPOT_SOURCE or "sina").lower()
    cache_key = ("ALL" if symbols is None else ",".join(sorted(str(c).zfill(6) for c in symbols)))
    now = time.time()
    if use_cache and _SPOT_CACHE.get("key") == cache_key and \
            (now - float(_SPOT_CACHE.get("ts") or 0)) < SPOT_TTL:
        return dict(_SPOT_CACHE.get("data") or {})
    result: Dict[str, Quote] = {}
    if symbols is not None:
        if src in ("sina", "auto"):
            try:
                result = fetch_spot_sina(list(symbols), timeout=timeout, indices=indices)
            except Exception:
                result = {}
        if not result:
            # 东财兜底：限制超时与重试，避免网络抖动时长时间卡住
            result = _EM_FETCH_SPOT(symbols, batch=batch, timeout=min(float(timeout), 8.0),
                                    retry=1, progress_cb=progress_cb, use_cache=False)
    else:
        result = _EM_FETCH_SPOT(None, batch=batch, timeout=min(float(timeout), 10.0), retry=1,
                                progress_cb=progress_cb, use_cache=False)
    if use_cache and result:
        _SPOT_CACHE.update({"ts": now, "data": dict(result), "key": cache_key})
    return result




# --------------------------------------------------------------------------- #
# 股票池：新浪行情中心（按成交额排序 → 活跃股票池）
# --------------------------------------------------------------------------- #
SINA_LIST = ("https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
             "Market_Center.getHQNodeData")
DEFAULT_RT_POOL = tuple(str(p["symbol"]).zfill(6) for p in DEFAULT_POOL)   # noqa: F821


def fetch_all_symbols(limit: int = 300, sort: str = "amount", client=None,
                      timeout: float = 12.0) -> List[str]:
    """获取 A 股代码列表（新浪行情中心）。

    ``sort='amount'`` 按成交额降序 → 得到**流动性最好的活跃股票池**（推荐用于选股）；
    ``sort='changepercent'`` 则按涨幅排序（用于观察异动）。
    每页 100 条，取满 ``limit`` 为止。
    """
    s = client if isinstance(client, MarketDataClient) else MarketDataClient(timeout=timeout, retry=1)
    codes: List[str] = []
    pages = max(1, int(np.ceil(int(limit) / 100.0)))
    for page in range(1, pages + 1):
        try:
            js = s.get_json(SINA_LIST, timeout=timeout,
                            params={"page": page, "num": 100, "sort": sort, "asc": 0,
                                    "node": "hs_a", "symbol": "", "_s_r_a": "page"},
                            headers={"Referer": "https://vip.stock.finance.sina.com.cn/"})
        except Exception:
            break
        if not isinstance(js, list) or not js:
            break
        for it in js:
            sym = str(it.get("symbol", ""))
            code = sym[-6:]
            if code.isdigit() and code not in codes:
                codes.append(code)
        if len(codes) >= limit:
            break
    return codes[:limit]


def resolve_pool(symbols=None, limit: int = 120) -> List[str]:
    """确定股票池：显式给出 → 用之；否则取活跃股票池（失败则用内置默认池）。"""
    if symbols:
        return [str(s).zfill(6) for s in symbols]
    try:
        codes = fetch_all_symbols(limit=limit)
        if codes:
            return codes
    except Exception:
        pass
    return list(DEFAULT_RT_POOL)




