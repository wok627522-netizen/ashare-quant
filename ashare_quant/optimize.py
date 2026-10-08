"""参数寻优与稳健性检验。

包含三件在实战里最重要的事：

1. **样本内/样本外切分**：所有参数都在样本内挑，样本外验证，避免"用未来数据选参数"。
2. **网格 / 随机搜索**：小参数空间用网格，大空间用随机搜索。
3. **滚动前推（Walk-Forward）**：滚动训练-测试，模拟"每年重新调参"的真实流程。

同时输出过拟合诊断（IS 与 OOS 的差距）与参数敏感性（参数平原越宽越稳）。
"""

from __future__ import annotations

import itertools
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .backtest import BacktestConfig, Backtester
from .data import DataBundle
from .strategies import build_strategy

__all__ = ["param_combinations", "grid_search", "random_search", "walk_forward",
           "sensitivity_table", "pick_stable_params"]

SCORE_KEYS = ["sharpe", "annual_return", "calmar", "total_return", "win_rate"]


def param_combinations(grid: Dict[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    """把参数网格展开成参数组合列表。"""
    keys = list(grid.keys())
    if not keys:
        return [{}]
    vals = [list(grid[k]) for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*vals)]


def _run_one(strategy_key: str, params: Dict[str, Any], data: DataBundle,
             config: Optional[BacktestConfig], metric: str) -> Optional[dict]:
    try:
        strat = build_strategy(strategy_key, **params)
        bt = Backtester(config)
        res = bt.run_strategy(data, strat)
        out = {"params": params, "metric": metric, "score": float(res.metrics.get(metric, 0.0))}
        for k in ("total_return", "annual_return", "annual_vol", "sharpe", "sortino", "calmar",
                  "max_drawdown", "win_rate", "profit_factor", "trade_count", "turnover", "exposure"):
            out[k] = float(res.metrics.get(k, 0.0))
        return out
    except Exception:
        return None


def _evaluate_split(strategy_key: str, params: Dict[str, Any], data: DataBundle,
                    config: Optional[BacktestConfig], metric: str,
                    train_ratio: float) -> Optional[dict]:
    """在样本内/样本外分别评估同一组参数。"""
    split_idx = int(len(data.calendar) * float(train_ratio))
    split_idx = max(30, min(split_idx, len(data.calendar) - 30))
    train = data.slice(end=data.calendar[split_idx - 1])
    test = data.slice(start=data.calendar[split_idx])
    a = _run_one(strategy_key, params, train, config, metric)
    b = _run_one(strategy_key, params, test, config, metric)
    if a is None:
        return None
    out: Dict[str, Any] = {"params": params}
    for k, v in (a or {}).items():
        if k not in ("params",):
            out[f"is_{k}"] = v
    for k, v in (b or {}).items():
        if k not in ("params",):
            out[f"oos_{k}"] = v
    out["is_score"] = float(a.get("score", 0.0))
    out["oos_score"] = float((b or {}).get("score", 0.0))
    out["overfit_gap"] = out["is_score"] - out["oos_score"]
    out["stable"] = bool(out["oos_score"] > 0 and out["overfit_gap"] < abs(out["is_score"]) * 0.8)
    return out


def grid_search(strategy_key: str, param_grid: Dict[str, Sequence[Any]], data: DataBundle,
                config: Optional[BacktestConfig] = None, metric: str = "sharpe",
                train_ratio: float = 0.7, n_jobs: int = 1,
                progress_cb: Optional[Callable[[float], None]] = None,
                max_combos: int = 2000) -> pd.DataFrame:
    """网格搜索 + 样本外验证。返回按样本外得分排序的结果表。"""
    combos = param_combinations(param_grid)
    if len(combos) > max_combos:
        combos = combos[:max_combos]
    return _search(strategy_key, combos, data, config, metric, train_ratio, n_jobs, progress_cb)


def random_search(strategy_key: str, param_grid: Dict[str, Sequence[Any]], data: DataBundle,
                  n_iter: int = 60, seed: int = 42, config: Optional[BacktestConfig] = None,
                  metric: str = "sharpe", train_ratio: float = 0.7, n_jobs: int = 1,
                  progress_cb: Optional[Callable[[float], None]] = None) -> pd.DataFrame:
    """随机搜索（参数空间大时比网格更高效）。"""
    rng = random.Random(int(seed))
    combos = []
    seen = set()
    pool = param_combinations(param_grid)
    if len(pool) <= n_iter:
        combos = pool
    else:
        while len(combos) < n_iter:
            c = {k: rng.choice(list(v)) for k, v in param_grid.items()}
            key = tuple(sorted(c.items(), key=lambda x: str(x[0])))
            if key not in seen:
                seen.add(key)
                combos.append(c)
    return _search(strategy_key, combos, data, config, metric, train_ratio, n_jobs, progress_cb)


def _search(strategy_key: str, combos: List[Dict[str, Any]], data: DataBundle,
            config: Optional[BacktestConfig], metric: str, train_ratio: float,
            n_jobs: int, progress_cb: Optional[Callable[[float], None]]) -> pd.DataFrame:
    rows: List[dict] = []
    total = max(len(combos), 1)

    def work(c):
        return _evaluate_split(strategy_key, c, data, config, metric, train_ratio)

    if n_jobs and n_jobs > 1:
        with ThreadPoolExecutor(max_workers=int(n_jobs)) as ex:
            futures = {ex.submit(work, c): c for c in combos}
            for i, fut in enumerate(as_completed(futures)):
                r = fut.result()
                if r:
                    rows.append(r)
                if progress_cb:
                    progress_cb((i + 1) / total)
    else:
        for i, c in enumerate(combos):
            r = work(c)
            if r:
                rows.append(r)
            if progress_cb:
                progress_cb((i + 1) / total)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    param_cols = sorted({k for r in rows for k in r["params"]})
    params_df = pd.DataFrame([r["params"] for r in rows]).reindex(columns=param_cols)
    out = pd.concat([params_df, df.drop(columns=["params"])], axis=1)
    return out.sort_values("oos_score", ascending=False).reset_index(drop=True)


def walk_forward(strategy_key: str, params: Dict[str, Any], data: DataBundle,
                 config: Optional[BacktestConfig] = None, n_splits: int = 5,
                 metric: str = "sharpe") -> pd.DataFrame:
    """滚动前推验证：把区间等分，前 k-1 段训练、第 k 段测试，拼接样本外净值。"""
    cal = data.calendar
    if len(cal) < 120:
        raise ValueError("数据太短，至少需要 120 个交易日")
    bounds = np.linspace(0, len(cal), int(n_splits) + 1).astype(int)
    rows = []
    for k in range(1, int(n_splits)):
        train = data.slice(end=cal[bounds[k] - 1])
        test = data.slice(start=cal[bounds[k]], end=cal[bounds[k + 1] - 1])
        tr = _run_one(strategy_key, params, train, config, metric)
        te = _run_one(strategy_key, params, test, config, metric)
        if tr is None or te is None:
            continue
        rows.append({
            "区间": f"{cal[bounds[k]].date()} ~ {cal[bounds[k + 1] - 1].date()}",
            "训练期": f"{cal[bounds[0]].date()} ~ {cal[bounds[k] - 1].date()}",
            "IS得分": tr.get("score", 0.0), "OOS得分": te.get("score", 0.0),
            "OOS年化": te.get("annual_return", 0.0), "OOS回撤": te.get("max_drawdown", 0.0),
            "OOS换手": te.get("turnover", 0.0), "OOS交易次数": te.get("trade_count", 0.0),
        })
    return pd.DataFrame(rows)


def sensitivity_table(results: pd.DataFrame, param: str, score_col: str = "oos_score") -> pd.DataFrame:
    """单参数敏感性：同一取值下的平均/最优样本外得分（用于找参数平原）。"""
    if results.empty or param not in results.columns:
        return pd.DataFrame()
    g = results.groupby(param)[score_col].agg(["mean", "max", "min", "count"])
    return g.sort_index().rename(columns={"mean": "平均OOS", "max": "最优OOS",
                                          "min": "最差OOS", "count": "测试组合数"})


def pick_stable_params(results: pd.DataFrame, score_col: str = "oos_score",
                       gap_penalty: float = 0.5) -> Dict[str, Any]:
    """在"样本外得分高"与"样本内外差距小"之间取平衡，返回推荐参数。"""
    if results.empty:
        return {}
    score = results[score_col] - gap_penalty * results.get("overfit_gap", 0.0).abs()
    best = results.loc[score.idxmax()]
    params = {c: best[c] for c in results.columns
              if c in best.index and not str(c).startswith(("is_", "oos_"))
              and c not in ("score", "metric", "overfit_gap", "stable")}
    return params