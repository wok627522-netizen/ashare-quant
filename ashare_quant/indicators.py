"""技术指标库（纯 pandas/numpy 实现，无 TA-Lib 依赖）。

所有函数都接受 ``pd.Series``/``pd.DataFrame``，返回同索引对象，
方便与策略层、回测层直接对齐。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "sma", "ema", "wma", "macd", "rsi", "kdj", "boll", "atr", "cci", "obv",
    "donchian", "momentum", "roc", "volatility", "zscore", "ma_slope", "adx",
    "true_range", "turnover_rate", "add_all",
]


def sma(s: pd.Series, n: int = 20) -> pd.Series:
    """简单移动平均。"""
    return s.rolling(int(n), min_periods=max(2, int(n) // 2)).mean()


def ema(s: pd.Series, n: int = 20) -> pd.Series:
    """指数移动平均。"""
    return s.ewm(span=int(n), adjust=False, min_periods=max(2, int(n) // 2)).mean()


def wma(s: pd.Series, n: int = 20) -> pd.Series:
    """加权移动平均。"""
    w = np.arange(1, int(n) + 1, dtype=float)
    return s.rolling(int(n)).apply(lambda x: float(np.dot(x, w) / w.sum()), raw=True)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD：返回 (dif, dea, hist)。"""
    dif = ema(close, fast) - ema(close, slow)
    dea = dif.ewm(span=int(signal), adjust=False).mean()
    hist = (dif - dea) * 2     # 国内软件惯例，柱 = 2×(DIF-DEA)
    return dif, dea, hist


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """相对强弱指标 RSI（Wilder 平滑）。"""
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    n = int(n)
    roll_up = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    roll_down = down.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = roll_up / roll_down.replace(0.0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.fillna(100.0).where(close.notna())


def kdj(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 9,
        m1: int = 3, m2: int = 3):
    """KDJ 随机指标。"""
    n = int(n)
    ln = low.rolling(n, min_periods=1).min()
    hn = high.rolling(n, min_periods=1).max()
    rsv = (close - ln) / (hn - ln).replace(0.0, np.nan) * 100
    rsv = rsv.fillna(50.0)
    k = rsv.ewm(com=int(m1) - 1, adjust=False).mean()
    d = k.ewm(com=int(m2) - 1, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


def boll(close: pd.Series, n: int = 20, k: float = 2.0):
    """布林带：返回 (中轨, 上轨, 下轨, %B, 带宽)。"""
    mid = sma(close, n)
    sd = close.rolling(int(n), min_periods=max(2, int(n) // 2)).std(ddof=0)
    up, dn = mid + k * sd, mid - k * sd
    pctb = (close - dn) / (up - dn).replace(0.0, np.nan)
    width = (up - dn) / mid.replace(0.0, np.nan)
    return mid, up, dn, pctb, width


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(),
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """平均真实波幅 ATR（Wilder）。"""
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / int(n), adjust=False, min_periods=int(n)).mean()


def cci(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 20) -> pd.Series:
    """顺势指标 CCI。"""
    tp = (high + low + close) / 3.0
    ma = tp.rolling(int(n), min_periods=max(2, int(n) // 2)).mean()
    md = (tp - ma).abs().rolling(int(n), min_periods=max(2, int(n) // 2)).mean()
    return (tp - ma) / (0.015 * md.replace(0.0, np.nan))


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """能量潮 OBV。"""
    sign = np.sign(close.diff().fillna(0.0))
    return (sign * volume.fillna(0.0)).cumsum()


def donchian(high: pd.Series, low: pd.Series, n: int = 20, include_today: bool = False):
    """唐奇安通道：返回 (上轨, 下轨, 中轨)。``include_today=False`` 可防未来函数。"""
    win = int(n)
    if not include_today:
        up = high.rolling(win, min_periods=win).max().shift(1)
        dn = low.rolling(win, min_periods=win).min().shift(1)
    else:
        up = high.rolling(win, min_periods=1).max()
        dn = low.rolling(win, min_periods=1).min()
    return up, dn, (up + dn) / 2.0


def momentum(close: pd.Series, n: int = 20) -> pd.Series:
    """N 日动量（收益率形式）。"""
    return close / close.shift(int(n)) - 1.0


def roc(close: pd.Series, n: int = 12) -> pd.Series:
    return (close / close.shift(int(n)) - 1.0) * 100


def volatility(close: pd.Series, n: int = 20, annualize: bool = True) -> pd.Series:
    """滚动波动率（默认年化）。"""
    r = close.pct_change()
    v = r.rolling(int(n), min_periods=max(3, int(n) // 2)).std(ddof=0)
    return v * np.sqrt(252) if annualize else v


def zscore(s: pd.Series, n: int = 20) -> pd.Series:
    """滚动 Z-Score，用于均值回归与标准化因子。"""
    m = s.rolling(int(n), min_periods=max(3, int(n) // 2)).mean()
    sd = s.rolling(int(n), min_periods=max(3, int(n) // 2)).std(ddof=0)
    return (s - m) / sd.replace(0.0, np.nan)


def ma_slope(s: pd.Series, n: int = 20, lookback: int = 5) -> pd.Series:
    """均线斜率 / 归一化，衡量趋势强度。"""
    ma = sma(s, n)
    return (ma - ma.shift(int(lookback))) / ma.shift(int(lookback)).abs().replace(0.0, np.nan)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """平均趋向指标 ADX（趋势强度，不区分方向）。"""
    n = int(n)
    up_move = high.diff()
    dn_move = -low.diff()
    plus_dm = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
    tr = true_range(high, low, close)
    atr_ = tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(alpha=1 / n, adjust=False).mean() / atr_
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(alpha=1 / n, adjust=False).mean() / atr_
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan) * 100
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def turnover_rate(volume: pd.Series, float_shares: float) -> pd.Series:
    """换手率（volume 单位为股，float_shares 为流通股本）。"""
    if not float_shares or float_shares <= 0:
        return pd.Series(np.nan, index=volume.index)
    return volume / float(float_shares) * 100


def add_all(df: pd.DataFrame, close_col: str = "close", high_col: str = "high",
            low_col: str = "low", volume_col: str = "volume") -> pd.DataFrame:
    """给行情表批量追加常用指标列（便于界面里直接画图）。"""
    out = df.copy()
    c = out[close_col]
    h = out.get(high_col, c)
    l = out.get(low_col, c)
    v = out.get(volume_col, pd.Series(0.0, index=out.index))
    for n in (5, 10, 20, 60):
        out[f"ma{n}"] = sma(c, n)
    out["dif"], out["dea"], out["macd"] = macd(c)
    out["rsi6"], out["rsi14"] = rsi(c, 6), rsi(c, 14)
    out["k"], out["d"], out["j"] = kdj(h, l, c)
    out["boll_mid"], out["boll_up"], out["boll_dn"], out["boll_pctb"], out["boll_width"] = boll(c)
    out["atr14"] = atr(h, l, c, 14)
    out["vol20"] = volatility(c, 20)
    out["mom20"] = momentum(c, 20)
    out["obv"] = obv(c, v)
    out["ret"] = c.pct_change()
    return out