"""股票池（Universe）构建：把"全市场"变成可用的选股范围。

支持范围
--------
======================  ==========================================  ============
``kind``                含义                                        数量级
======================  ==========================================  ============
``main_board``          **沪深主板**（沪 600/601/603/605 + 深 000/001/002/003）  约 3500 只
``all_a``               全部 A 股（含创业板/科创板/北交所）          约 5500 只
``gem``                 创业板（300/301）                          约 1400 只
``star``                科创板（688/689）                          约 600 只
``active``              活跃股票池（按成交额排序取前 N）             自定义
``default``             内置示例池（15 只，最快）                    15 只
``custom``              用户自定义代码                                —
======================  ==========================================  ============

列表来源：东方财富行情列表（分页，每页 100 条，约 0.1 秒/页），
带 1 天磁盘缓存；失败时回退到新浪行情中心。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, field
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import DATA_DIR, DEFAULT_POOL
from .http_util import MarketDataClient

__all__ = ["POOL_KINDS", "UniverseInfo", "fetch_symbol_list", "build_pool", "pool_summary",
           "MAIN_BOARD_FS", "POOL_LABELS", "is_main_board"]

CACHE_DIR = DATA_DIR / "universe"
MAIN_BOARD_FS = "m:0+t:6,m:1+t:2"        # 深市主板 + 沪市主板
ALL_A_FS = "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23"
GEM_FS = "m:0+t:80"
STAR_FS = "m:1+t:23"
CLIST_URLS = ("https://push2.eastmoney.com/api/qt/clist/get",
              "http://push2.eastmoney.com/api/qt/clist/get")
UT = "bd1d9ddb04089700cf9c27f6f7426281"

POOL_KINDS = ["main_board", "active", "all_a", "gem", "star", "default", "custom"]
POOL_LABELS: Dict[str, str] = {
    "main_board": "沪深主板（全部）",
    "active": "沪深主板活跃股（按成交额前 N）",
    "all_a": "全部 A 股（含创业板/科创板/北交所）",
    "gem": "创业板",
    "star": "科创板",
    "default": "内置示例池（15 只，最快）",
    "custom": "自定义代码",
}


@dataclass
class UniverseInfo:
    """股票池信息。"""

    kind: str
    label: str
    symbols: List[str]
    industries: Dict[str, str] = field(default_factory=dict)
    names: Dict[str, str] = field(default_factory=dict)
    total: int = 0
    source: str = ""
    cached: bool = False
    elapsed: float = 0.0

    def to_dict(self) -> dict:
        return {"类型": self.label, "数量": len(self.symbols), "市场总数": self.total,
                "来源": self.source, "缓存": "是" if self.cached else "否",
                "耗时秒": round(self.elapsed, 2)}


def _code_of(x) -> str:
    s = "".join(ch for ch in str(x) if ch.isdigit())
    return s[-6:].zfill(6) if s else ""


def is_main_board(symbol: str) -> bool:
    """是否沪深主板（沪 600/601/603/605，深 000/001/002/003）。"""
    c = _code_of(symbol)
    return c.startswith(("600", "601", "603", "605", "000", "001", "002", "003"))



def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """把 price/pct_chg/amount/market 强制转成数值。

    东财接口对停牌/无数据的股票会返回 "-"，直接参与排序会抛
    TypeError: '<' not supported between instances of 'str' and 'float'。
    """
    for c in ("price", "pct_chg", "amount", "market"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _name_bad(name: str) -> bool:
    """需要剔除的名称：ST/退市/未上市等。"""
    n = str(name).upper()
    for kw in ("ST", "退", "N ", "C "):
        if kw in n:
            return True
    return False


def _cache_file(kind: str):
    return CACHE_DIR / f"universe_{kind}.csv"


def fetch_symbol_list(kind: str = "main_board", limit: Optional[int] = None,
                      exclude_st: bool = True, use_cache: bool = True,
                      cache_hours: float = 12.0, timeout: float = 15.0,
                      progress_cb=None) -> tuple:
    """抓取股票列表（东财分页 → 新浪兜底）。

    Returns
    -------
    (DataFrame, source, cached)
        列：symbol / name / price / pct_chg / amount / market
    """
    fs = {"main_board": MAIN_BOARD_FS, "active": MAIN_BOARD_FS, "all_a": ALL_A_FS,
          "gem": GEM_FS, "star": STAR_FS}.get(kind, MAIN_BOARD_FS)
    cp = _cache_file(kind)
    if use_cache and cp.exists() and (time.time() - cp.stat().st_mtime) < cache_hours * 3600:
        try:
            df = _coerce_numeric(pd.read_csv(cp, dtype={"symbol": str}))
            if not df.empty:
                return df, "本地缓存", True
        except Exception:
            pass

    client = MarketDataClient(timeout=timeout, retry=1)
    rows: List[dict] = []
    page, total, pages = 1, None, None
    while True:
        params = {"pn": page, "pz": 100, "po": 1, "np": 1, "ut": UT, "fltt": 2, "invt": 2,
                  "fid": "f12", "fs": fs, "fields": "f12,f14,f2,f3,f6,f13,f100"}
        js = None
        for _attempt in range(3):           # 接口偶发限流，重试 3 次
            for url in CLIST_URLS:
                try:
                    js = client.get_json(url, params=params, timeout=timeout)
                    break
                except Exception:
                    js = None
            if js is not None:
                break
            time.sleep(0.8 * (_attempt + 1))
        data = (js or {}).get("data") or {}
        diff = data.get("diff") or []
        if total is None:
            total = int(data.get("total") or 0)
            pages = max(1, int(np.ceil(total / 100.0))) if total else 1
        if not diff:
            break
        for it in diff:
            code = _code_of(it.get("f12"))
            if not code:
                continue
            rows.append({"symbol": code, "name": str(it.get("f14", "")),
                         "price": it.get("f2"), "pct_chg": it.get("f3"),
                         "amount": it.get("f6"), "market": int(it.get("f13") or 0),
                         "industry": str(it.get("f100") or "未分类")})
        if progress_cb:
            progress_cb(min(1.0, page / float(pages or 1)))
        page += 1
        if page > (pages or 1) or page > 80:
            break

    if not rows:
        # 新浪兜底：沪市 + 深市
        rows = _fetch_list_sina(client, progress_cb=progress_cb)
    df = _coerce_numeric(pd.DataFrame(rows))
    if df.empty:
        return df, "无数据", False
    df = df.drop_duplicates(subset=["symbol"]).reset_index(drop=True)
    # 沪深主板（含"活跃股"）必须过滤掉创业板/科创板/北交所 ——
    # 之前只有 main_board 走了这个过滤，导致"活跃股"池混入 300/688 等非主板股票。
    if kind in ("main_board", "active"):
        _before = len(df)
        df = df[df["symbol"].map(is_main_board)].reset_index(drop=True)
        if _before != len(df):
            pass  # 已剔除非主板标的
    if exclude_st:
        df = df[~df["name"].map(_name_bad)].reset_index(drop=True)
    if limit:
        df = df.head(int(limit)).reset_index(drop=True)
    # 分页抓取可能被限流中断：若明显不完整（< 总数的 90%），不要覆盖已有缓存，
    # 否则会把 3000 只的列表悄悄替换成 1800 只（历史 bug）。
    _partial = bool(total and len(df) < int(total * 0.9))
    if _partial and cp.exists():
        try:
            old = _coerce_numeric(pd.read_csv(cp, dtype={"symbol": str}))
            if len(old) > len(df):
                return old, "本地缓存(新抓取不完整)", True
        except Exception:
            pass
    if not _partial:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(cp, index=False, encoding="utf-8-sig")
        except Exception:
            pass
    return df, "东财列表" + ("(不完整)" if _partial else ""), False


def _fetch_list_sina(client, progress_cb=None) -> List[dict]:
    """新浪行情中心兜底（沪市/深市主板）。"""
    url = ("https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
           "Market_Center.getHQNodeData")
    rows: List[dict] = []
    for node in ("sh_a", "sz_a"):
        for page in range(1, 40):
            try:
                js = client.get_json(url, timeout=12,
                                     params={"page": page, "num": 100, "sort": "amount",
                                             "asc": 0, "node": node, "symbol": "",
                                             "_s_r_a": "page"},
                                     headers={"Referer": "https://vip.stock.finance.sina.com.cn/"})
            except Exception:
                break
            if not isinstance(js, list) or not js:
                break
            for it in js:
                code = _code_of(it.get("symbol"))
                if code:
                    rows.append({"symbol": code, "name": str(it.get("name", "")),
                                 "price": it.get("trade"), "pct_chg": it.get("changepercent"),
                                 "amount": it.get("amount"), "market": 1 if node == "sh_a" else 0})
            if progress_cb:
                progress_cb(0.5 if node == "sh_a" else 1.0)
    return rows


def build_pool(kind: str = "main_board", limit: Optional[int] = None,
               custom: Optional[Sequence[str]] = None, exclude_st: bool = True,
               use_cache: bool = True, progress_cb=None) -> UniverseInfo:
    """构建股票池。

    * ``main_board``／``all_a``／``gem``／``star``：取相应板块全部股票；
    * ``active``：沪深主板按成交额降序取前 ``limit`` 只（默认 300）；
    * ``default``：内置 15 只；
    * ``custom``：使用 ``custom`` 传入的代码。
    """
    t0 = time.time()
    kind = str(kind or "main_board").lower()
    label = POOL_LABELS.get(kind, kind)
    if kind == "default":
        syms = [str(p["symbol"]).zfill(6) for p in DEFAULT_POOL]
        return UniverseInfo(kind, label, syms, total=len(syms), source="内置",
                            elapsed=time.time() - t0)
    if kind == "custom":
        syms = [_code_of(s) for s in (custom or []) if _code_of(s)]
        return UniverseInfo(kind, label, syms, total=len(syms), source="自定义",
                            elapsed=time.time() - t0)
    want = int(limit) if limit else (300 if kind == "active" else None)
    df, source, cached = fetch_symbol_list(kind, limit=None, exclude_st=exclude_st,
                                           use_cache=use_cache, progress_cb=progress_cb)
    total = int(len(df))
    if df.empty:
        syms = [str(p["symbol"]).zfill(6) for p in DEFAULT_POOL]
        return UniverseInfo(kind, label + "（回退默认池）", syms, total=0, source="回退",
                            elapsed=time.time() - t0)
    if kind == "active" and want:
        df = _coerce_numeric(df).sort_values("amount", ascending=False,
                                           na_position="last").head(want)
    df = df.copy()
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    syms = df["symbol"].tolist()
    industries = dict(zip(df["symbol"], df["industry"])) if "industry" in df.columns else {}
    names = dict(zip(df["symbol"], df["name"])) if "name" in df.columns else {}
    return UniverseInfo(kind, label, syms, industries=industries, names=names, total=total,
                        source=source, cached=cached, elapsed=time.time() - t0)


def pool_summary(info: UniverseInfo) -> pd.DataFrame:
    """股票池信息表格（界面展示）。"""
    return pd.DataFrame(list(info.to_dict().items()), columns=["项目", "数值"])



