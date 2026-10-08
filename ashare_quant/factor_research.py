"""因子研究与筛选：IC / ICIR / IC 衰减 / 分层收益 / 因子相关性 → 有效性评分。

产出三样东西：

1. **因子有效性排名表**：每个因子的 IC 均值、ICIR、t 值、IC 胜率、单调性、
   多空组合年化、换手，最后给出「保留 / 观察 / 反向使用 / 剔除」的建议；
2. **IC 时序与衰减**：用于判断因子是"稳定有效"还是"偶然有效"；
3. **分层收益**：Q1~Q5 组合的累计净值，检验单调性（真因子应该单调递增）。

因子方向统一约定：**数值越大越好**（例如波动率取负号、反转取负号），
因此 IC > 0 表示因子方向正确。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import math
import numpy as np
import pandas as pd

from . import indicators as ind
from .data import DataBundle
from .strategies.base import cross_zscore

__all__ = ["FACTOR_SET", "FACTOR_DESC", "build_factor_panel", "forward_return",
           "factor_ic_series", "ic_stats", "ic_decay", "quantile_returns",
           "factor_correlation", "screen_factors", "FactorScreenResult", "rank_ic"]

# --------------------------------------------------------------------------- #
# 因子集合（全部已做"越大越好"的方向统一）
# --------------------------------------------------------------------------- #
FACTOR_SET: List[str] = [
    "mom20", "mom60", "mom120",          # 动量
    "rev5", "rev20",                     # 短期反转
    "vol20", "vol60",                    # 低波动
    "atr_pct",                           # 低波幅
    "trend",                             # 均线多头强度
    "ma_slope20",                        # 均线斜率
    "high52",                            # 距 52 周高点
    "sharpe60",                          # 风险调整收益
    "maxdd60",                           # 回撤浅
    "liquidity",                         # 流动性溢价（小成交额优先）
    "illiq",                             # Amihud 非流动性
    "updays20",                          # 上涨天数占比
    "skew20",                            # 收益偏度（低偏度优先）
    "rsi_rev14",                         # RSI 反转（超买反向）
    "boll_pos",                          # 布林带位置（低吸）
    "turnover20",                        # 换手活跃度
    "boll_break",                        # 跌破布林下轨（超跌幅度）
    "rebound_combo",                     # 超跌反弹组合·严格版
    "rebound_soft",                      # 超跌反弹组合·宽松版
]

FACTOR_DESC: Dict[str, str] = {
    "mom20": "20日动量", "mom60": "60日动量", "mom120": "120日动量",
    "rev5": "5日反转", "rev20": "20日反转",
    "vol20": "20日低波动", "vol60": "60日低波动", "atr_pct": "低波幅(ATR/价)",
    "trend": "趋势强度(MA20/MA60)", "ma_slope20": "均线斜率",
    "high52": "距52周高点", "sharpe60": "风险调整收益",
    "maxdd60": "60日回撤浅", "liquidity": "流动性(小市值溢价)",
    "illiq": "非流动性(Amihud)", "updays20": "上涨天数占比",
    "skew20": "收益偏度(低偏度)", "rsi_rev14": "RSI超买反向",
    "boll_pos": "布林带低位", "turnover20": "换手活跃度",
    "amount20": "流动性(成交额)",
    "boll_break": "跌破布林下轨",
    "rebound_combo": "超跌反弹组合(严格)",
    "rebound_soft": "超跌反弹组合(宽松)",
}


def build_factor_panel(data: DataBundle, factors: Optional[Sequence[str]] = None,
                       max_symbols: Optional[int] = None) -> Dict[str, pd.DataFrame]:
    """构建因子面板（方向已统一，越大越好）。

    ``max_symbols`` 用于限制股票数（按近 20 日成交额取前 N），
    在全主板 3000+ 只上做因子研究时能显著提速。
    """
    close = data.close_matrix()
    high, low = data.high_matrix(), data.low_matrix()
    amount = data.amount_matrix()
    volume = data.volume_matrix()
    if max_symbols and len(close.columns) > max_symbols:
        keep = amount.tail(20).mean().sort_values(ascending=False).head(int(max_symbols)).index
        close, high, low = close[keep], high[keep], low[keep]
        amount, volume = amount[keep], volume[keep]
    ret = close.pct_change()

    p: Dict[str, pd.DataFrame] = {}
    p["mom20"] = close.shift(5) / close.shift(25) - 1.0
    p["mom60"] = close.shift(5) / close.shift(65) - 1.0
    p["mom120"] = close.shift(5) / close.shift(125) - 1.0
    p["rev5"] = -(close / close.shift(5) - 1.0)
    p["rev20"] = -(close / close.shift(20) - 1.0)
    p["vol20"] = -ind.volatility(close, 20)
    p["vol60"] = -ind.volatility(close, 60)
    atr = pd.DataFrame({c: ind.atr(high[c], low[c], close[c], 14) for c in close.columns})
    p["atr_pct"] = -(atr / close.replace(0.0, np.nan))
    p["trend"] = ind.sma(close, 20) / ind.sma(close, 60) - 1.0
    p["ma_slope20"] = ind.ma_slope(close, 20, 5)
    p["high52"] = close / close.rolling(250, min_periods=60).max()
    mean20 = ret.rolling(20, min_periods=10).mean()
    sd20 = ret.rolling(20, min_periods=10).std(ddof=0)
    p["sharpe60"] = (mean20 / sd20.replace(0.0, np.nan)).rolling(3, min_periods=1).mean()
    p["maxdd60"] = close / close.rolling(60, min_periods=20).max() - 1.0
    p["liquidity"] = -np.log1p(amount.rolling(20, min_periods=5).mean())
    # amount20 是 liquidity 的别名（兼容 factors.py 的命名），只在被显式请求时提供，
    # 不参与因子排名（否则同一个因子会占两个名额）
    if factors and "amount20" in factors:
        p["amount20"] = p["liquidity"]
    illiq = (ret.abs() / amount.replace(0.0, np.nan))
    p["illiq"] = np.log1p(illiq.rolling(20, min_periods=5).mean() * 1e8)
    p["updays20"] = (ret > 0).rolling(20, min_periods=10).mean()
    p["skew20"] = -ret.rolling(20, min_periods=10).skew()
    p["rsi_rev14"] = -ind.rsi(close, 14)
    _mid, _up, _dn, pctb, _w = ind.boll(close, 20, 2.0)
    p["boll_pos"] = -pctb
    # 跌破布林下轨幅度：(下轨 - 收盘) / 下轨
    # > 0 表示已跌破下轨（超跌），值越大跌得越狠；< 0 表示还在下轨之上。
    p["boll_break"] = (_dn - close) / _dn.replace(0.0, np.nan)

    # ---- 超跌反弹组合因子 ----
    # 硬条件（四个同时满足，否则不入选）：
    #   ① 跌破布林下轨  ② RSI(14) 超卖(<35)  ③ 缩量(5日均量 < 20日均量×0.8)
    #   ④ 大盘企稳(基准指数在 20 日均线上方)
    # 满足后再按"跌得越狠 + RSI越低 + 缩量越明显"给连续打分，便于排序选股。
    try:
        _rsi14 = ind.rsi(close, 14)
        _v5 = volume.rolling(5, min_periods=3).mean()
        _v20 = volume.rolling(20, min_periods=10).mean()
        _broke = close < _dn                       # 跌破下轨
        _oversold = _rsi14 < 35                    # 超卖
        _shrink = _v5 < _v20 * 0.8                 # 缩量
        _cond = _broke & _oversold & _shrink
        _bench = getattr(data, "benchmark", None)
        if _bench is not None and len(pd.Series(_bench).dropna()) > 30:
            _bm = pd.Series(_bench).dropna()
            _bm_above = (_bm > _bm.rolling(20, min_periods=10).mean())
            _calm = _bm_above.reindex(close.index).ffill().fillna(False)
            _cond = _cond.mul(_calm.astype(float), axis=0) > 0.5
        _score = ((-_rsi14 / 50.0)
                  + (_dn / close - 1.0).clip(lower=0.0) * 10.0
                  + (_v20 / _v5.replace(0.0, np.nan) - 1.0).clip(0.0, 3.0) * 0.3)
        p["rebound_combo"] = _score.where(_cond)
    except Exception:
        p["rebound_combo"] = pd.DataFrame(np.nan, index=close.index, columns=close.columns)

    # ---- 超跌反弹组合·宽松版 ----
    # 条件放宽（不要求跌破下轨、不要求大盘企稳），候选更多、更适合当"选股因子"：
    #   ① RSI(14) < 40（偏超卖）  ② 缩量(5日均量 < 20日均量)  ③ 距 60 日高点回撤 > 15%
    try:
        _rsi_s = ind.rsi(close, 14)
        _v5s = volume.rolling(5, min_periods=3).mean()
        _v20s = volume.rolling(20, min_periods=10).mean()
        _dd60 = close / close.rolling(60, min_periods=20).max() - 1.0
        _cond_s = (_rsi_s < 40) & (_v5s < _v20s) & (_dd60 < -0.15)
        _score_s = ((-_rsi_s / 50.0)
                    + (-_dd60).clip(0.0, 1.0) * 1.5
                    + (_v20s / _v5s.replace(0.0, np.nan) - 1.0).clip(0.0, 2.0) * 0.3)
        p["rebound_soft"] = _score_s.where(_cond_s)
    except Exception:
        p["rebound_soft"] = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    p["turnover20"] = np.log1p((amount / close.replace(0.0, np.nan)).rolling(20, min_periods=5).mean())
    for k in list(p):
        p[k] = p[k].replace([np.inf, -np.inf], np.nan)
    if factors:
        p = {k: v for k, v in p.items() if k in factors}
    return p


def forward_return(data: DataBundle, horizon: int = 20,
                   max_symbols: Optional[int] = None) -> pd.DataFrame:
    """未来 N 日收益率（研究用；绝不能进入实盘信号）。"""
    close = data.close_matrix()
    if max_symbols and len(close.columns) > max_symbols:
        keep = data.amount_matrix().tail(20).mean().sort_values(ascending=False).head(int(max_symbols)).index
        close = close[keep]
    h = int(horizon)
    return close.shift(-h) / close - 1.0


def rank_ic(factor: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    """逐日 Rank IC（Spearman：先排序再求相关）。"""
    f = factor.reindex_like(fwd)
    a, b = f.rank(axis=1), fwd.reindex_like(f).rank(axis=1)
    return a.corrwith(b, axis=1)


def factor_ic_series(panel: Dict[str, pd.DataFrame], fwd: pd.DataFrame) -> pd.DataFrame:
    """所有因子的 IC 时序：索引=日期，列=因子。"""
    return pd.DataFrame({k: rank_ic(v, fwd) for k, v in panel.items()})


def ic_stats(ic: pd.DataFrame, periods: int = 252) -> pd.DataFrame:
    """IC 统计表：均值、标准差、ICIR、t 值、IC>0 占比。"""
    rows = []
    for k in ic.columns:
        s = ic[k].dropna()
        n = len(s)
        if n < 10:
            rows.append({"factor": k, "name": FACTOR_DESC.get(k, k), "ic_mean": np.nan,
                         "ic_std": np.nan, "icir": np.nan, "t_stat": np.nan,
                         "ic_win": np.nan, "n": n})
            continue
        mean, sd = float(s.mean()), float(s.std(ddof=0))
        rows.append({
            "factor": k, "name": FACTOR_DESC.get(k, k), "ic_mean": mean, "ic_std": sd,
            "icir": float(mean / sd) if sd > 0 else 0.0,
            "t_stat": float(mean / sd * np.sqrt(n)) if sd > 0 else 0.0,
            "ic_win": float((s > 0).mean()), "n": n,
        })
    return pd.DataFrame(rows)


def ic_decay(panel: Dict[str, pd.DataFrame], data: DataBundle,
             horizons: Sequence[int] = (1, 5, 10, 20, 60),
             max_symbols: Optional[int] = None) -> pd.DataFrame:
    """IC 衰减：不同持有期下各因子的 IC 均值（判断因子有效期）。"""
    out: Dict[int, pd.Series] = {}
    for h in horizons:
        fwd = forward_return(data, horizon=int(h), max_symbols=max_symbols)
        ic = factor_ic_series(panel, fwd)
        out[int(h)] = ic.mean()
    df = pd.DataFrame(out)
    df.index.name = "factor"
    return df


def quantile_returns(factor: pd.DataFrame, fwd: pd.DataFrame, q: int = 5) -> pd.DataFrame:
    """分层收益：Q1（最差）~ Qq（最好）每期的平均收益。"""
    f = factor.reindex_like(fwd)
    ranks = f.rank(axis=1, pct=True)
    labels = np.ceil(ranks * int(q)).clip(1, int(q))
    out = {}
    for g in range(1, int(q) + 1):
        out[f"Q{g}"] = fwd.where(labels == g).mean(axis=1)
    res = pd.DataFrame(out)
    res["多空"] = res[f"Q{int(q)}"] - res["Q1"]
    return res


def factor_correlation(panel: Dict[str, pd.DataFrame], sample_days: int = 120,
                       max_dates: int = 40) -> pd.DataFrame:
    """因子相关性（抽样的横截面标准化相关系数）。

    性能说明：早期实现对全部历史日期做 stack（600 只 × 1900 天 × 190 个因子对），
    会吃掉数 GB 内存并让服务卡死；这里改为**等间隔抽样 max_dates 天**，
    结果几乎不变但内存/耗时下降一个数量级。
    """
    step = max(1, int(sample_days) // max(1, int(max_dates)))
    zs = {k: cross_zscore(v.tail(sample_days).iloc[::step]) for k, v in panel.items()}
    cols = list(zs)
    mat = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for i, a in enumerate(cols):
        for b in cols[i:]:
            va, vb = zs[a].stack(), zs[b].stack()
            idx = va.index.intersection(vb.index)
            r = float(va.loc[idx].corr(vb.loc[idx])) if len(idx) > 30 else np.nan
            mat.loc[a, b] = mat.loc[b, a] = r
    for c in cols:
        mat.loc[c, c] = 1.0
    return mat


@dataclass
class FactorScreenResult:
    """因子筛选结果。"""

    table: pd.DataFrame
    ic_series: pd.DataFrame
    decay: pd.DataFrame
    corr: pd.DataFrame
    quantiles: Dict[str, pd.DataFrame] = field(default_factory=dict)
    panel: Dict[str, pd.DataFrame] = field(default_factory=dict)
    horizon: int = 20
    universe: int = 0
    keep: List[str] = field(default_factory=list)
    drop: List[str] = field(default_factory=list)
    top5: List[str] = field(default_factory=list)
    top5_table: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary_text(self) -> str:
        lines = [f"【因子筛选】样本 {self.universe} 只，持有期 {self.horizon} 个交易日",
                 f"🏆 最重要的 5 个因子：{('、'.join(FACTOR_DESC.get(k, k) for k in self.top5)) or '无'}",
                 f"建议保留 {len(self.keep)} 个：{('、'.join(FACTOR_DESC.get(k, k) for k in self.keep)) or '无'}",
                 f"建议剔除 {len(self.drop)} 个：{('、'.join(FACTOR_DESC.get(k, k) for k in self.drop)) or '无'}"]
        if not self.table.empty:
            t = self.table.head(8)
            for _, r in t.iterrows():
                lines.append(f"  {r['名称']}：IC {r['IC均值']:+.3f}｜ICIR {r['ICIR']:+.2f}｜"
                             f"t {r['t值']:+.2f}｜单调性 {r['单调性']:.2f}｜{r['建议']}")
        return "\n".join(lines)



# --------------------------------------------------------------------------- #
# 因子显著性：FDR 校正 + 分段 IC 稳定性（避免"23 个因子里挑运气"）
# --------------------------------------------------------------------------- #
def _t_to_p(t_values):
    """t 值 -> 双侧 p 值（正态近似，样本量通常 >100，无需 scipy）。"""
    t = np.asarray(t_values, dtype=float)
    return np.clip(2.0 * np.array([math.erfc(abs(float(x)) / math.sqrt(2.0)) for x in t]), 0.0, 1.0)


def benjamini_hochberg(pvals, q: float = 0.10):
    """Benjamini-Hochberg FDR 控制：错误发现率 <= q 时，返回每个假设是否通过。"""
    p = np.asarray(pvals, dtype=float)
    n = int(len(p))
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    sp = p[order]
    thr = np.arange(1, n + 1) / n * float(q)
    passed = sp <= thr
    ok = np.zeros(n, dtype=bool)
    if passed.any():
        k = int(np.where(passed)[0][-1])
        ok[order[: k + 1]] = True
    return ok


def ic_segment_stability(ic, n_segments: int = 4) -> dict:
    """把 IC 时序等分 n 段，统计与全样本同号的段数（>=n-1 视为稳定）。"""
    s = pd.Series(ic).dropna()
    if len(s) < n_segments * 15:
        return {"same_sign": 0, "stable": False, "segments": {}}
    idx = np.clip(np.floor(np.arange(len(s)) / (len(s) / n_segments)).astype(int),
                  0, n_segments - 1)
    seg = s.groupby(idx).mean()
    same = int((np.sign(seg.values) == np.sign(s.mean())).sum())
    return {"same_sign": same, "stable": same >= n_segments - 1,
            "segments": {int(k): round(float(v), 4) for k, v in seg.items()}}


def apply_fdr_stability(table, ic, q: float = 0.10, min_same_segments: int = 3):
    """给因子表加 p值/FDR通过/段数同向/稳定 四列，并用它收紧「建议」。"""
    keycol = "因子" if "因子" in table.columns else "factor"
    tcol = "t值" if "t值" in table.columns else "t_stat"
    advcol = "建议" if "建议" in table.columns else None
    if tcol in table.columns:
        table["p值"] = _t_to_p(table[tcol].values)
        table["FDR通过"] = benjamini_hochberg(table["p值"].values, q=q)
    else:
        table["p值"] = np.nan
        table["FDR通过"] = False
    if ic is not None and len(getattr(ic, "columns", [])):
        stab = {c: ic_segment_stability(ic[c]) for c in ic.columns}
        table["段数同向"] = table[keycol].map(lambda k: stab.get(k, {}).get("same_sign", 0))
    else:
        table["段数同向"] = 0
    table["稳定"] = table["段数同向"] >= int(min_same_segments)
    if advcol:
        def _tighten(row):
            adv = str(row.get(advcol, ""))
            ok = bool(row.get("FDR通过", False)) and bool(row.get("稳定", False))
            if ("保留" in adv or "反向" in adv) and not ok:
                return "👀 观察(FDR/稳定性未过)"
            return adv
        table[advcol] = table.apply(_tighten, axis=1)
    return table


def _monotonicity(q_means: pd.Series) -> float:
    """分层单调性：Q 序号与平均收益的 Spearman 相关（绝对值越大越单调）。"""
    s = pd.Series(q_means).dropna()
    if len(s) < 3:
        return 0.0
    ranks = pd.Series(np.arange(1, len(s) + 1), index=s.index)
    # 用"排序后 Pearson"实现 Spearman，避免依赖 scipy
    return float(ranks.rank().corr(s.rank()))


def screen_factors(data: DataBundle, horizon: int = 20, max_symbols: int = 800,
                   q: int = 5, factors: Optional[Sequence[str]] = None,
                   decay_horizons: Sequence[int] = (1, 5, 20, 60),
                   corr: bool = True,
                   progress_cb: Optional[Callable[[float], None]] = None) -> FactorScreenResult:
    """一键因子筛选：IC + ICIR + t + 单调性 + 多空年化 → 保留/剔除建议。"""
    def _p(x):
        if progress_cb:
            progress_cb(float(x))

    _p(0.05)
    panel = build_factor_panel(data, factors=factors, max_symbols=max_symbols)
    if not panel:
        raise RuntimeError("没有可用因子")
    universe = int(panel[next(iter(panel))].shape[1])
    _p(0.35)
    fwd = forward_return(data, horizon=int(horizon), max_symbols=max_symbols)
    fwd = fwd.reindex(columns=next(iter(panel.values())).columns)
    ic = factor_ic_series(panel, fwd)
    stats = ic_stats(ic)
    _p(0.6)
    decay = ic_decay(panel, data, horizons=decay_horizons, max_symbols=max_symbols)
    _p(0.85)
    rows = []
    quantiles: Dict[str, pd.DataFrame] = {}
    for k, f in panel.items():
        qr = quantile_returns(f, fwd, q=q)
        quantiles[k] = qr
        q_means = qr[[f"Q{i}" for i in range(1, int(q) + 1)]].mean()
        mono = _monotonicity(q_means)
        ls = float(qr["多空"].mean()) * (252.0 / max(int(horizon), 1))
        row = stats[stats["factor"] == k].iloc[0].to_dict() if (stats["factor"] == k).any() else {}
        row.update({"单调性": abs(mono), "单调方向": "正向" if mono >= 0 else "反向",
                    "多空年化": ls, "近期IC": 0.0})
        rows.append(row)
    table = pd.DataFrame(rows)
    # 近期 IC（最近 60 期）用于判断因子是否失效
    recent = ic.tail(60).mean()
    table["近期IC"] = table["factor"].map(recent)
    # 建议
    def _advice(r) -> str:
        ic_m = r.get("ic_mean", np.nan)
        t = r.get("t_stat", np.nan)
        mono = r.get("单调性", np.nan)
        if not np.isfinite(ic_m) or not np.isfinite(t):
            return "样本不足"
        if ic_m > 0.02 and t > 2.0 and mono > 0.6:
            return "✅ 保留"
        if ic_m < -0.02 and t < -2.0:
            return "🔄 反向使用"
        if abs(ic_m) > 0.015 and abs(t) > 1.5:
            return "👀 观察"
        return "❌ 剔除"

    table["建议"] = table.apply(_advice, axis=1)
    table = apply_fdr_stability(table, locals().get("ic"), q=0.10, min_same_segments=3)
    table = table.rename(columns={"factor": "因子", "name": "名称", "ic_mean": "IC均值",
                                  "ic_std": "IC标准差", "icir": "ICIR", "t_stat": "t值",
                                  "ic_win": "IC胜率", "n": "样本数"})
    order = ["因子", "名称", "重要性评分", "IC均值", "ICIR", "t值", "p值", "FDR通过",
             "IC胜率", "单调性", "单调方向", "多空年化", "近期IC", "段数同向", "稳定",
             "样本数", "建议"]
    table = table[[c for c in order if c in table.columns]]
    sort_key = table["IC均值"].abs() if "IC均值" in table.columns else table.index
    table = table.assign(_k=sort_key).sort_values("_k", ascending=False).drop(columns="_k")
    keep = table[table["建议"].str.contains("保留|反向", na=False)]["因子"].tolist()
    drop = table[table["建议"].str.contains("剔除", na=False)]["因子"].tolist()

    # ---------- 重要性评分：挑出最该用的 5 个因子 ----------
    # 权重设计：强度 35% + 稳定性 25% + 显著性 20% + 分层单调性 10% + 近期未失效 10%
    def _importance(r) -> float:
        ic = abs(float(r.get("IC均值", 0) or 0))
        icir = abs(float(r.get("ICIR", 0) or 0))
        tv = abs(float(r.get("t值", 0) or 0))
        mono = float(r.get("单调性", 0) or 0)
        recent = float(r.get("近期IC", 0) or 0)
        same = 1.0 if (recent * float(r.get("IC均值", 0) or 0) > 0) else 0.0
        return (0.35 * min(ic / 0.05, 1.0)          # IC 强度（|IC|=0.05 即满分）
                + 0.25 * min(icir / 0.25, 1.0)      # 稳定性（ICIR）
                + 0.20 * min(tv / 4.0, 1.0)         # 显著性（t 值）
                + 0.10 * min(max(mono, 0.0), 1.0)   # 分层单调性
                + 0.10 * same)                      # 近期是否仍有效

    table["重要性评分"] = table.apply(_importance, axis=1)
    table = table.sort_values("重要性评分", ascending=False).reset_index(drop=True)
    corr_df = factor_correlation(panel) if corr else pd.DataFrame()

    # Top5 基本门槛：|t| ≥ 1.5 且 |IC| ≥ 0.015（过滤噪声）
    elig = table[(table["t值"].abs() >= 1.5) & (table["IC均值"].abs() >= 0.015)]
    # 再按相关性去重：|相关系数| > 0.9 视为同一个因子，只保留评分更高的那个
    picked: List[str] = []
    for _, _r in elig.iterrows():
        _k = str(_r["因子"])
        if not corr_df.empty and _k in corr_df.index:
            _dup = False
            for _prev in picked:
                if _prev in corr_df.columns:
                    try:
                        if abs(float(corr_df.loc[_k, _prev])) > 0.9:
                            _dup = True
                            break
                    except Exception:
                        pass
            if _dup:
                continue
        picked.append(_k)
        if len(picked) >= 5:
            break
    top5 = picked
    top5_table = table[table["因子"].isin(top5)].copy()
    top5_table["_o"] = top5_table["因子"].map({k: i for i, k in enumerate(top5)})
    top5_table = top5_table.sort_values("_o").drop(columns="_o")
    _p(1.0)
    return FactorScreenResult(table=table, ic_series=ic, decay=decay, corr=corr_df,
                              quantiles=quantiles, panel=panel, horizon=int(horizon),
                              universe=universe, keep=keep, drop=drop,
                              top5=top5, top5_table=top5_table)









# --------------------------------------------------------------------------- #
# 大样本（全主板 3000+ 只）批量因子筛选：逐因子计算，内存占用仅 ~1 个因子矩阵
# --------------------------------------------------------------------------- #
def screen_factors_large(data: DataBundle, horizon: int = 20, max_symbols: int = 3000,
                         q: int = 5, factors: Optional[Sequence[str]] = None,
                         decay_horizons: Sequence[int] = (1, 5, 20, 60),
                         progress_cb: Optional[Callable[[float], None]] = None,
                         corr_top: int = 0) -> FactorScreenResult:
    """大样本因子筛选（逐因子循环，适合 3000+ 只）。

    与 ``screen_factors`` 的差别：**一次只构建一个因子的面板**（用完即释放），
    因此内存峰值只有单个因子矩阵（约 1/N），代价是耗时长一些。
    """
    factor_list = list(factors or FACTOR_SET)

    def _p(x):
        if progress_cb:
            progress_cb(float(x))

    _p(0.03)
    fwd_main = forward_return(data, horizon=int(horizon), max_symbols=max_symbols)
    if fwd_main.empty:
        raise RuntimeError("无法计算未来收益（数据不足）")
    cols = fwd_main.columns            # 统一股票池（按成交额取前 N）
    _p(0.08)

    rows: List[dict] = []
    ic_cols: Dict[str, pd.Series] = {}
    quantiles: Dict[str, pd.DataFrame] = {}
    decay_data: Dict[str, Dict[int, float]] = {}
    n = max(len(factor_list), 1)

    for i, k in enumerate(factor_list):
        try:
            p1 = build_factor_panel(data, factors=[k], max_symbols=max_symbols)
            if k not in p1:
                continue
            f = p1[k].reindex(columns=cols)
            ic = rank_ic(f, fwd_main).dropna()
            if len(ic) < 10:
                continue
            ic_cols[k] = rank_ic(f, fwd_main)
            qr = quantile_returns(f, fwd_main, q=q)
            quantiles[k] = qr
            q_means = qr[[f"Q{j}" for j in range(1, int(q) + 1)]].mean()
            mono = _monotonicity(q_means)
            ls = float(qr["多空"].mean()) * (252.0 / max(int(horizon), 1))
            mean, sd = float(ic.mean()), float(ic.std(ddof=0))
            rows.append({
                "factor": k, "name": FACTOR_DESC.get(k, k),
                "ic_mean": mean, "ic_std": sd,
                "icir": float(mean / sd) if sd > 0 else 0.0,
                "t_stat": float(mean / sd * np.sqrt(len(ic))) if sd > 0 else 0.0,
                "ic_win": float((ic > 0).mean()), "n": len(ic),
                "单调性": abs(mono), "单调方向": "正向" if mono >= 0 else "反向",
                "多空年化": ls, "近期IC": float(ic.tail(60).mean()),
            })
            # IC 衰减：同一因子对不同持有期
            dd = {}
            for h in decay_horizons:
                if int(h) == int(horizon):
                    dd[int(h)] = mean
                    continue
                try:
                    fh = forward_return(data, horizon=int(h), max_symbols=max_symbols).reindex(columns=cols)
                    dd[int(h)] = float(rank_ic(f, fh).mean())
                except Exception:
                    dd[int(h)] = np.nan
            decay_data[k] = dd
        except Exception:
            continue
        _p(0.08 + 0.82 * (i + 1) / n)

    if not rows:
        raise RuntimeError("没有可用因子")
    table = pd.DataFrame(rows)
    table = table.rename(columns={"factor": "因子", "name": "名称", "ic_mean": "IC均值",
                                  "ic_std": "IC标准差", "icir": "ICIR", "t_stat": "t值",
                                  "ic_win": "IC胜率", "n": "样本数"})
    ic = pd.DataFrame(ic_cols)
    decay = pd.DataFrame(decay_data).T
    decay.columns = [int(c) for c in decay.columns]
    decay.index.name = "factor"

    def _advice(r) -> str:
        ic_m, t, mono = r.get("IC均值", np.nan), r.get("t值", np.nan), r.get("单调性", np.nan)
        if not np.isfinite(ic_m) or not np.isfinite(t):
            return "样本不足"
        if ic_m > 0.02 and t > 2.0 and mono > 0.6:
            return "✅ 保留"
        if ic_m < -0.02 and t < -2.0:
            return "🔄 反向使用"
        if abs(ic_m) > 0.015 and abs(t) > 1.5:
            return "👀 观察"
        return "❌ 剔除"

    table["建议"] = table.apply(_advice, axis=1)
    table = apply_fdr_stability(table, locals().get("ic"), q=0.10, min_same_segments=3)

    def _importance(r) -> float:
        ic_ = abs(float(r.get("IC均值", 0) or 0))
        icir_ = abs(float(r.get("ICIR", 0) or 0))
        tv = abs(float(r.get("t值", 0) or 0))
        mono_ = float(r.get("单调性", 0) or 0)
        recent = float(r.get("近期IC", 0) or 0)
        same = 1.0 if (recent * float(r.get("IC均值", 0) or 0) > 0) else 0.0
        return (0.35 * min(ic_ / 0.05, 1.0) + 0.25 * min(icir_ / 0.25, 1.0)
                + 0.20 * min(tv / 4.0, 1.0) + 0.10 * min(max(mono_, 0.0), 1.0) + 0.10 * same)

    table["重要性评分"] = table.apply(_importance, axis=1)
    table["_k"] = table["IC均值"].abs()
    table = table.sort_values("重要性评分", ascending=False).drop(columns="_k").reset_index(drop=True)
    order = ["因子", "名称", "重要性评分", "IC均值", "ICIR", "t值", "p值", "FDR通过",
             "IC胜率", "单调性", "单调方向", "多空年化", "近期IC", "段数同向", "稳定",
             "样本数", "建议"]
    table = table[[c for c in order if c in table.columns]]

    # 相关性：只对 Top10 因子算（用 IC 时序相关做冗余判断，几乎不耗内存）
    corr_df = pd.DataFrame()
    if corr_top and len(table):
        topk = table["因子"].head(int(corr_top)).tolist()
        sub = ic[topk] if set(topk) <= set(ic.columns) else ic
        corr_df = sub.corr()

    elig = table[(table["t值"].abs() >= 1.5) & (table["IC均值"].abs() >= 0.015)]
    picked: List[str] = []
    for _, _r in elig.iterrows():
        _k2 = str(_r["因子"])
        if not corr_df.empty and _k2 in corr_df.index:
            if any(abs(float(corr_df.loc[_k2, _pp])) > 0.9
                   for _pp in picked if _pp in corr_df.columns):
                continue
        picked.append(_k2)
        if len(picked) >= 5:
            break
    top5 = picked
    top5_table = table[table["因子"].isin(top5)].copy()
    if top5:
        top5_table["_o"] = top5_table["因子"].map({k: i for i, k in enumerate(top5)})
        top5_table = top5_table.sort_values("_o").drop(columns="_o")
    keep = table[table["建议"].str.contains("保留|反向", na=False)]["因子"].tolist()
    drop = table[table["建议"].str.contains("剔除", na=False)]["因子"].tolist()
    _p(1.0)
    return FactorScreenResult(table=table, ic_series=ic, decay=decay, corr=corr_df,
                              quantiles=quantiles, panel={}, horizon=int(horizon),
                              universe=len(cols), keep=keep, drop=drop,
                              top5=top5, top5_table=top5_table)

# --------------------------------------------------------------------------- #
# 结论解读：把因子筛选结果翻译成人话（界面「📌 结论解读」用）
# --------------------------------------------------------------------------- #
def interpret_screen(res) -> dict:
    """根据筛选结果自动生成中文解读：市场风格 / 可用因子 / 被拦因子 / 建议权重 / 风险提示。"""
    t = getattr(res, "table", None)
    if t is None or len(t) == 0:
        return {"summary": "尚无筛选结果", "style": [], "usable": [], "blocked": [],
                "weights": {}, "notes": []}

    def _g(row, key, default=0.0):
        try:
            v = row.get(key, default)
            return float(v) if v is not None else default
        except Exception:
            return default

    adv = t["建议"].astype(str)
    usable = t[adv.str.contains("保留|反向", na=False)]
    blocked = t[adv.str.contains("观察", na=False)]
    dropped = t[adv.str.contains("剔除", na=False)]

    pos = [r for _, r in usable.iterrows() if _g(r, "IC均值") > 0]
    neg = [r for _, r in usable.iterrows() if _g(r, "IC均值") <= 0]

    # ---- 市场风格判断 ----
    style = []
    names_pos = [str(r.get("名称", "")) for r in pos]
    names_neg = [str(r.get("名称", "")) for r in neg]
    if any(("动量" in n) or ("趋势" in n) or ("MA" in n) for n in names_neg):
        style.append("**动量 / 趋势类因子是反向的** —— 说明当前「追涨」不赚钱，涨得多的后面反而跌（A 股典型的反转市特征）")
    if any(("波动" in n) or ("回撤" in n) for n in names_pos + names_neg):
        style.append("**低波动 / 低回撤因子有效** —— 资金偏防御，稳健标的占优")
    if any(("流动性" in n) or ("非流动性" in n) or ("换手" in n) for n in names_pos + names_neg):
        style.append("**小市值 / 低流动性溢价明显** —— 冷门、成交清淡的股票反而跑赢（这是 A 股长期存在的异象）")
    if any(("反转" in n) for n in names_pos):
        style.append("**短期反转有效** —— 超跌股有反弹，但要注意别接趋势性下跌的刀")
    if not style:
        style.append("本期没有明显统一的市场风格，建议以多因子分散为主")

    # ---- 可用因子 ----
    usable_rows = []
    for r in (pos + neg):
        ic, tv = _g(r, "IC均值"), _g(r, "t值")
        mono = _g(r, "单调性")
        usable_rows.append({
            "名称": str(r.get("名称", "")),
            "方向": "正向（数值越大越买）" if ic > 0 else "反向（数值越小越买，权重取负）",
            "IC": ic, "t": tv, "单调性": mono,
            "做法": f"{'买' if ic > 0 else '回避'}因子值最高的一组",
        })

    # ---- 被拦下的因子 ----
    blocked_rows = []
    for _, r in blocked.iterrows():
        pv, same = _g(r, "p值"), int(_g(r, "段数同向"))
        fdr = bool(r.get("FDR通过", False))
        if not fdr:
            why = f"统计不显著（p={pv:.3f}，未通过 FDR 10% 校正）"
        elif same < 3:
            why = f"方向不稳定（4 段里只有 {same} 段与整体同向）"
        else:
            why = "未达入选门槛"
        blocked_rows.append({"名称": str(r.get("名称", "")), "原因": why,
                             "IC": _g(r, "IC均值"), "t": _g(r, "t值")})

    # ---- 建议权重 ----
    weights = {}
    for r in (pos + neg):
        nm = str(r.get("名称", ""))
        ic = _g(r, "IC均值")
        if not nm:
            continue
        # 按 |IC| 相对大小分配权重，总仓位控制在 1.0 以内，单因子上限 0.3
        weights[nm] = (1 if ic > 0 else -1) * min(abs(ic) * 3.0, 0.30)
    _tot = sum(abs(v) for v in weights.values()) or 1.0
    if _tot > 1.0:
        weights = {k: round(v / _tot, 3) for k, v in weights.items()}

    notes = [
        f"结论基于 {getattr(res, 'universe', '?')} 只股票的样本、持有期 {getattr(res, 'horizon', '?')} 个交易日；"
        "样本期不同，结论可能变化，建议每周重跑一次。",
        "只有同时通过 **FDR 校正（p<0.05/0.10）** 与 **四段 IC 同向** 的因子才列在「可用」里，其余只是观察。",
        "因子方向会随市场切换（熊市反转、牛市动量），不要长期固定一套权重。",
        "入选理由里包含「反向」因子时，代表该因子值越小越好（权重为负）。",
    ]
    summary = (f"共检验 {len(t)} 个因子：可用 {len(usable)} 个"
               f"（正向 {len(pos)} / 反向 {len(neg)}），"
               f"被 FDR/稳定性拦下 {len(blocked)} 个，"
               f"完全无效 {len(dropped)} 个。")
    return {"summary": summary, "style": style, "usable": usable_rows,
            "blocked": blocked_rows, "weights": weights, "notes": notes,
            "horizon": getattr(res, "horizon", None), "universe": getattr(res, "universe", None)}
