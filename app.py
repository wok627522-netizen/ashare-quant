# -*- coding: utf-8 -*-
"""A 股量化研究平台 · Streamlit 可视化界面

启动方式
--------
    streamlit run app.py
或 Windows 双击 ``run_app.bat``。

功能页签
--------
1. 数据总览      行情概览、K 线、相关性、覆盖度
2. 策略回测      10+ 内置策略，一键回测并查看全部绩效图表
3. 参数寻优      网格/随机搜索 + 样本外验证 + 参数敏感性
4. 策略对比      多策略同口径横向比较
5. 因子选股      多因子打分选股 + IC 分析 + 分层回测
6. 风控分析      风险体检、行业暴露、压力测试
7. 实盘信号      最新调仓清单 + 模拟盘账户
8. 规则说明      A 股交易规则、成本假设与免责声明
"""

from __future__ import annotations

import datetime as dt
import io
import json
import time
from dataclasses import replace as dc_replace
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from ashare_quant import charts as ch
from ashare_quant import indicators as ind
from ashare_quant.backtest import BacktestConfig, Backtester, compare_strategies
from ashare_quant.config import BENCHMARKS, EXPORT_DIR, TRADE_DAYS_PER_YEAR
from ashare_quant.data import (DataBundle, fetch_akshare_bundle, load_bundle,
                               load_csv_bundle, save_bundle, synthetic_bundle)
from ashare_quant.factors import (FACTOR_LABELS, composite_score, factor_ic,
                                  factor_panel, factor_table, forward_returns,
                                  ic_summary, quantile_returns, score_to_weights)
from ashare_quant.metrics import (METRIC_LABELS, format_metric, performance_summary,
                                    rolling_metrics)
from ashare_quant.optimize import (grid_search, pick_stable_params, random_search,
                                   sensitivity_table, walk_forward)
from ashare_quant.risk import (correlation_summary, industry_exposure, portfolio_risk,
                               risk_report, stress_test)
from ashare_quant.rules import CostModel, TradingRules, get_board
from ashare_quant.signals import (PaperAccount, export_signals, latest_signals,
                                  signal_summary_text)
from ashare_quant.strategies import REGISTRY, build_strategy, list_strategies
from ashare_quant.execution import (RiskGate, RiskLimits, available_brokers, create_broker,
                                    is_trading_time, session_name)
from ashare_quant.live import LiveConfig, LiveTrader
from ashare_quant.quotes import get_quotes, quote_frame
from ashare_quant.selector import SelectionConfig, select_history, select_stocks, selection_stats
from ashare_quant.trading_calendar import (calendar_info, next_trade_date, prev_trade_date,
                                           refresh_calendar)
from ashare_quant.realtime import fetch_realtime_bundle, fetch_spot, market_status
from ashare_quant.http_util import network_diagnose
from ashare_quant.labels import to_cn
from ashare_quant.param_info import param_label, param_help, describe_params
from ashare_quant.universe import POOL_LABELS, build_pool, pool_summary
from ashare_quant.factor_research import (FACTOR_DESC, FACTOR_SET, build_factor_panel,
                                          screen_factors, screen_factors_large)
from ashare_quant.auth import require_password, logout_button
from ashare_quant.selector import resolve_buy_date

st.set_page_config(page_title="A 股量化研究平台", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")

CUSTOM_CSS = """
<style>
    .block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1500px;}
    h1, h2, h3 {font-weight: 650;}
    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, #ffffff 0%, #f7f9fc 100%);
        border: 1px solid #e6e9ef; border-radius: 12px; padding: 12px 14px;
        box-shadow: 0 1px 3px rgba(16,24,40,0.04);
    }
    div[data-testid="stMetricLabel"] {font-size: 0.82rem; color: #667085;}
    div[data-testid="stMetricValue"] {font-size: 1.35rem;}
    .tag {display:inline-block; padding:2px 8px; border-radius:999px; font-size:0.72rem;
          background:#eef2ff; color:#3730a3; margin-right:6px;}
    .warnbox {background:#fff7ed; border-left:4px solid #f59e0b; padding:10px 14px;
              border-radius:8px; font-size:0.88rem; color:#7c2d12;}
    .okbox {background:#ecfdf5; border-left:4px solid #10b981; padding:10px 14px;
            border-radius:8px; font-size:0.88rem; color:#065f46;}
    .small {color:#667085; font-size:0.82rem;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================ 访问密码门禁 ============================ #
# 未登录时只渲染登录页，不加载任何行情/持仓数据（防止数据泄露）。
if not require_password():
    st.stop()

# ---------------- 运行环境检测（本地 / Streamlit Cloud 云端） ----------------
def _detect_cloud() -> bool:
    import os as _os
    if str(_os.environ.get("ASHARE_CLOUD", "")).strip() in ("1", "true", "True"):
        return True
    try:
        import streamlit as _st
        if "ASHARE_CLOUD" in _st.secrets and str(_st.secrets["ASHARE_CLOUD"]).lower() in ("1", "true"):
            return True
    except Exception:
        pass
    # Streamlit Cloud 会把仓库挂载在 /mount/src 下
    try:
        return "/mount/src" in str(Path.cwd())
    except Exception:
        return False


IS_CLOUD = _detect_cloud()

REBALANCE_OPTIONS = {"每日": 1, "每周": "W", "每两周": 10, "每月": "M", "每季度": "Q"}
REBALANCE_REVERSE = {v: k for k, v in REBALANCE_OPTIONS.items()}


# --------------------------------------------------------------------------- #
# 缓存的数据加载
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def demo_bundle(n_stocks: int, start: str, end: str, seed: int) -> DataBundle:
    return synthetic_bundle(n_stocks=int(n_stocks), start=start, end=end, seed=int(seed))


@st.cache_data(show_spinner=False)
def akshare_bundle(symbols: tuple, start: str, end: str, adjust: str) -> DataBundle:
    return fetch_akshare_bundle(list(symbols), start=start, end=end, adjust=adjust)


@st.cache_resource(show_spinner=False)
def get_backtester(_config_key: str, config: BacktestConfig) -> Backtester:
    return Backtester(config)


DEFAULT_RT_CODES = tuple(p["symbol"] for p in __import__("ashare_quant.config", fromlist=["DEFAULT_POOL"]).DEFAULT_POOL)


@st.cache_data(ttl=180, max_entries=1, show_spinner=False)
def realtime_bundle_cached(codes: tuple, days: int = 250, nonce: int = 0):
    """实时行情（3 分钟缓存）：新浪日线 + 实时快照；``nonce`` 变化可强制刷新。"""
    return fetch_realtime_bundle(list(codes), days=int(days), workers=16)


def resolve_pool_cached(kind: str, limit: int = 300, custom: tuple = ()):
    """股票池（10 分钟缓存）：main_board=全部沪深主板 / active=主板活跃前 N / …"""
    return build_pool(kind, limit=(int(limit) if limit else None),
                      custom=list(custom) if custom else None)


def load_realtime(codes, days: int = 250, quiet: bool = False,
                  pool_kind: Optional[str] = None, pool_limit: int = 300,
                  progress_cb=None, force: bool = False):
    """拉取实时行情并写入 session_state（返回 bundle 或 None）。

    ``pool_kind`` 非空时按股票池名称解析（main_board 全部沪深主板 / active 活跃前 N …）。
    """
    try:
        info = None
        if pool_kind:
            info = resolve_pool_cached(pool_kind, pool_limit)
            codes = info.symbols
            st.session_state.rt_pool_info = info
        if not codes:
            raise RuntimeError("股票池为空")
        nonce = int(time.time()) if force else int(st.session_state.get("rt_nonce", 0) or 0)
        if force:
            st.session_state.rt_nonce = nonce
        b, status, quotes = None, None, None
        last_err = None
        for attempt in range(3):
            try:
                b, status, quotes = realtime_bundle_cached(tuple(codes), int(days), nonce)
                if b is not None and len(b.symbols) > 0:
                    break
            except Exception as exc:
                last_err = exc
            time.sleep(1.5 * (attempt + 1))
        if b is None or len(b.symbols) == 0:
            raise RuntimeError(f"realtime fetch failed after 3 retries: {last_err}")
        if info is not None and getattr(info, "industries", None):
            # 用股票池的行业/名称补全元信息（选股表里显示真实行业）
            idx = b.meta.index
            b.meta["industry"] = [info.industries.get(s, "未分类") for s in idx]
            if getattr(info, "names", None):
                b.meta["name"] = [info.names.get(s, s) for s in idx]
        # 每次运行都刷新"当前市场状态"，并挂到 bundle 上供选股器使用
        now_status = market_status()
        b.realtime_status = now_status
        b.quote_time = getattr(b, "quote_time", None)
        st.session_state.bundle = b
        st.session_state.rt_status = now_status
        st.session_state.rt_quotes = quotes
        st.session_state.rt_loaded_at = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not quiet:
            st.success(f"实时行情已加载：{len(b.symbols)} 只标的，行情时间 {b.quote_time}")
        return b
    except Exception as exc:
        if not quiet:
            st.error(f"实时行情获取失败：{exc}")
        return None


def init_state() -> None:
    if "bundle" not in st.session_state:
        # 先用轻量示例数据把界面渲染出来（秒开），真正的实时行情在下面异步加载，
        # 这样用户在等待"全主板 3000+ 只"行情时看到的是进度条，而不是白屏。
        st.session_state.bundle = demo_bundle(12, "2021-01-01", "2024-12-31", 42)
        st.session_state.rt_status = None
        st.session_state.rt_pending = True
    if "result" not in st.session_state:
        st.session_state.result = None
    if "opt_results" not in st.session_state:
        st.session_state.opt_results = None
    if "multi_results" not in st.session_state:
        st.session_state.multi_results = {}
    if "factor_result" not in st.session_state:
        st.session_state.factor_result = None


def current_config() -> BacktestConfig:
    """从侧边栏读取交易成本与风控参数，构造回测配置。"""
    s = st.session_state
    return BacktestConfig(
        initial_cash=float(s.get("cash", 1_000_000.0)),
        cost=CostModel(commission_rate=float(s.get("commission", 0.00025)),
                       commission_min=float(s.get("commission_min", 5.0)),
                       stamp_tax_rate=float(s.get("stamp", 0.0005)),
                       transfer_fee_rate=float(s.get("transfer", 0.00001)),
                       slippage_bps=float(s.get("slippage", 5.0))),
        rules=TradingRules(t_plus_1=bool(s.get("t1", True)),
                           enforce_price_limit=bool(s.get("limit", True)),
                           enforce_suspension=bool(s.get("susp", True)),
                           lot_size=int(s.get("lot", 100))),
        max_position_weight=float(s.get("max_pos", 0.30)),
        max_industry_weight=float(s.get("max_ind", 0.50)),
        min_trade_amount=float(s.get("min_amount", 2000.0)),
        rebalance_band=float(s.get("band", 0.01)),
        stop_loss_atr=float(s.get("atr_stop", 0.0)),
        trailing_stop=float(s.get("trail", 0.0)),
        max_drawdown_stop=float(s.get("dd_stop", 0.0)),
        circuit_breaker_cooldown=int(s.get("cooldown", 20)),
    )


# --------------------------------------------------------------------------- #
# 侧边栏
# --------------------------------------------------------------------------- #
def sidebar() -> None:
    sb = st.sidebar
    sb.markdown("## 📈 A 股量化研究平台")
    sb.caption("策略研究 · 回测验证 · 因子选股 · 实盘信号")

    sb.markdown("### 1️⃣ 数据源")
    source = sb.radio("选择数据来源",
                      ["🟢 实时行情（A股）", "示例数据（离线合成）", "本地 CSV 文件夹", "AkShare 在线抓取"],
                      label_visibility="collapsed")

    pool = None
    if source == "🟢 实时行情（A股）":
        pool_kind = sb.selectbox("股票池范围", list(POOL_LABELS.keys()),
                                 format_func=lambda k: POOL_LABELS.get(k, k),
                                 index=(list(POOL_LABELS.keys()).index("active") if IS_CLOUD
                                 else list(POOL_LABELS.keys()).index("main_board")),
                                 key="rt_pool")
        rt_limit = 300
        rt_codes_text = ""
        if pool_kind == "active":
            rt_limit = int(sb.number_input("取成交额前 N 只", 50, 3000, 300, 50, key="rt_limit"))
        elif pool_kind == "custom":
            rt_codes_text = sb.text_area("自定义代码（逗号/换行分隔）",
                                         ",".join(DEFAULT_RT_CODES), height=80, key="rt_codes")
        rt_days = sb.slider("历史窗口（交易日）", 120, 600, 250, 20, key="rt_days")
        rt_codes = tuple(x.strip() for x in rt_codes_text.replace("\n", ",").split(",") if x.strip())
        start, end = dt.date(2000, 1, 1), dt.date.today()
        sb.caption("数据源：新浪（日线，不复权）+ 新浪/东财实时快照；缓存 60 秒 / 日线 6 小时。")
        if pool_kind == "main_board":
            sb.caption("⚠️ 沪深主板约 3100 只，**首次抓取约 90 秒**，之后走缓存会快很多。")
        rt_status_now = st.session_state.get("rt_status")
        if rt_status_now is not None:
            info_now = st.session_state.get("rt_pool_info")
            n_now = len(info_now.symbols) if info_now is not None else len(st.session_state.get("bundle").symbols)
            sb.info(f"{rt_status_now.session}　下一交易日 {rt_status_now.next_trade_date.date()}　"
                    f"当前股票池 {n_now} 只")
    elif source == "示例数据（离线合成）":
        c1, c2 = sb.columns(2)
        n_stocks = c1.number_input("股票数", 4, 40, 12, 1)
        seed = c2.number_input("随机种子", 1, 99999, 42, 1)
        start = sb.date_input("开始日期", dt.date(2021, 1, 1))
        end = sb.date_input("结束日期", dt.date(2024, 12, 31))
        sb.caption("合成数据包含涨跌停、停牌、行业因子与牛熊切换，仅用于演示。")
    elif source == "本地 CSV 文件夹":
        csv_dir = sb.text_input("CSV 目录路径", str(Path.cwd() / "data"))
        start = sb.date_input("开始日期", dt.date(2020, 1, 1))
        end = sb.date_input("结束日期", dt.date(2024, 12, 31))
        sb.caption("每个文件一只股票，文件名即代码，列名支持中文或英文。")
    else:
        default_codes = "600519,000858,601318,600036,300750,002594,600276,000333,601899,600900"
        pool_text = sb.text_area("股票代码（逗号或换行分隔）", default_codes, height=90)
        pool = tuple(t.strip() for t in pool_text.replace("\n", ",").split(",") if t.strip())
        adjust = sb.selectbox("复权方式", ["qfq", "hfq", ""], format_func=lambda x: {"qfq": "前复权", "hfq": "后复权", "": "不复权"}[x])
        start = sb.date_input("开始日期", dt.date(2021, 1, 1))
        end = sb.date_input("结束日期", dt.date(2024, 12, 31))
        sb.caption("AkShare 为免费数据源，首次抓取较慢，建议抓取后保存缓存。")

    if sb.button("🔄 加载 / 刷新实时数据", width="stretch", type="primary"):
        with st.spinner("正在准备数据…"):
            try:
                if source == "🟢 实时行情（A股）":
                    prog = st.progress(0.0, text="正在获取股票列表与实时行情（首次约 90 秒）…")
                    bundle = load_realtime(rt_codes, int(rt_days), pool_kind=pool_kind,
                                           pool_limit=int(rt_limit), force=True,
                                           progress_cb=lambda p: prog.progress(min(float(p), 0.99)))
                    prog.progress(1.0, text="实时行情加载完成")
                    if bundle is None:
                        raise RuntimeError("实时行情获取失败，请检查网络后重试")
                elif source == "示例数据（离线合成）":
                    bundle = demo_bundle(int(n_stocks), str(start), str(end), int(seed))
                elif source == "本地 CSV 文件夹":
                    bundle = load_csv_bundle(Path(csv_dir))
                    bundle = bundle.slice(start=start, end=end)
                else:
                    bundle = akshare_bundle(pool, str(start), str(end), adjust)
                st.session_state.bundle = bundle
                st.session_state.result = None
                st.session_state.opt_results = None
                st.session_state.multi_results = {}
                st.success(f"已加载 {len(bundle.symbols)} 只标的，{len(bundle)} 个交易日")
            except Exception as exc:
                st.error(f"数据加载失败：{exc}")

    bundle: DataBundle = st.session_state.bundle
    sb.markdown("### 2️⃣ 交易成本与规则")
    with sb.expander("资金与费率", expanded=False):
        st.number_input("初始资金（元）", 10_000.0, 1e10, 1_000_000.0, 10_000.0, key="cash")
        c1, c2 = st.columns(2)
        c1.number_input("佣金率", 0.0, 0.01, 0.00025, 0.00005, format="%.5f", key="commission")
        c2.number_input("最低佣金", 0.0, 100.0, 5.0, 1.0, key="commission_min")
        c1.number_input("印花税（卖出）", 0.0, 0.01, 0.0005, 0.0001, format="%.4f", key="stamp")
        c2.number_input("过户费（双边）", 0.0, 0.001, 0.00001, 0.00001, format="%.5f", key="transfer")
        st.slider("滑点（bp）", 0.0, 50.0, 5.0, 0.5, key="slippage")
        st.number_input("每手股数", 1, 1000, 100, 100, key="lot")
    with sb.expander("制度约束", expanded=False):
        st.checkbox("T+1（当日买入次日可卖）", True, key="t1")
        st.checkbox("涨跌停不可成交", True, key="limit")
        st.checkbox("停牌不可交易", True, key="susp")
        st.slider("单票权重上限", 0.05, 1.0, 0.30, 0.05, key="max_pos")
        st.slider("行业权重上限", 0.1, 1.0, 0.50, 0.05, key="max_ind")
        st.number_input("最小成交金额（元）", 0.0, 100000.0, 2000.0, 500.0, key="min_amount")
        st.slider("调仓阈值（权重偏离）", 0.0, 0.10, 0.01, 0.005, key="band")
    with sb.expander("风控参数", expanded=False):
        st.slider("ATR 止损倍数（0=关闭）", 0.0, 6.0, 0.0, 0.5, key="atr_stop")
        st.slider("移动止损（0=关闭）", 0.0, 0.5, 0.0, 0.05, key="trail")
        st.slider("组合回撤熔断（0=关闭）", 0.0, 0.6, 0.0, 0.05, key="dd_stop")
        st.number_input("熔断冷却交易日", 1, 120, 20, 1, key="cooldown")

    with sb.expander("数据缓存"):
        st.caption(f"当前：{len(bundle.symbols)} 只标的 / {len(bundle)} 交易日")
        st.caption(f"区间：{bundle.calendar[0].date()} ~ {bundle.calendar[-1].date()}")
        if st.button("💾 保存到本地缓存", width="stretch"):
            p = save_bundle(bundle)
            st.success(f"已保存：{p.name}")
        up = st.file_uploader("载入缓存文件(.pkl)", type=["pkl"])
        if up is not None:
            try:
                tmp = EXPORT_DIR / "_uploaded_bundle.pkl"
                tmp.write_bytes(up.getbuffer())
                st.session_state.bundle = load_bundle(tmp)
                st.success("缓存载入成功，请切换到任意页签查看")
            except Exception as exc:
                st.error(f"载入失败：{exc}")
    sb.markdown("---")
    try:
        _login_at = st.session_state.get("_login_time")
        if _login_at:
            sb.caption(f"🔓 已登录（{_login_at}）")
        logout_button()
    except Exception:
        pass
    sb.caption("⚠️ 本工具仅用于研究与教学，不构成投资建议，不提供自动下单。")


# --------------------------------------------------------------------------- #
# 通用组件
# --------------------------------------------------------------------------- #
def strategy_picker(label: str = "选择策略", key: str = "strategy", **kwargs) -> str:
    table = list_strategies()
    options = table["key"].tolist()
    names = {r["key"]: f"{r['name']}｜{r['category']}" for _, r in table.iterrows()}
    return st.selectbox(label, options, format_func=lambda k: names.get(k, k), key=key, **kwargs)


def param_widgets(cls, prefix: str, defaults: dict | None = None) -> dict:
    """按策略的 default_params 自动生成参数控件（全中文标签 + 悬浮说明）。"""
    params: dict = {}
    cols = st.columns(3)
    for i, (name, default) in enumerate(cls.default_params.items()):
        base = (defaults or {}).get(name, default)
        container = cols[i % 3]
        key = f"{prefix}_{name}"
        label = param_label(name)          # 中文标签
        tip = param_help(name)             # 悬浮说明
        if isinstance(default, bool):
            params[name] = container.checkbox(label, bool(base), key=key, help=tip)
        elif isinstance(default, int):
            lo, hi, step = 0, max(int(base) * 5, 60), 1
            if name in ("top_k", "per_industry"):
                lo, hi, step = 1, 20, 1
            elif name in ("lookback", "mom_window", "vol_window", "entry", "exit", "ref_window"):
                lo, hi, step = 3, 300, 1
            elif name in ("max_grids",):
                lo, hi, step = 1, 10, 1
            elif name == "trend_ma":
                lo, hi, step = 0, 250, 5
            elif name == "rebalance":
                params[name] = container.selectbox(label, list(REBALANCE_OPTIONS.keys()),
                                                   index=2, key=key, help=tip)
                params[name] = REBALANCE_OPTIONS[params[name]]
                continue
            params[name] = container.number_input(label, lo, hi, int(base), step, key=key, help=tip)
        elif isinstance(default, float):
            lo, hi = 0.0, max(abs(float(base)) * 4, 1.0)
            if name.startswith("w_"):
                lo, hi, step = 0.0, 1.0, 0.05
            elif "pctb" in name or name in ("trailing", "max_width"):
                lo, hi, step = 0.0, 1.0, 0.05
            else:
                step = 0.1
            params[name] = container.number_input(label, float(lo), float(hi), float(base),
                                                  float(step), key=key, help=tip)
        elif isinstance(default, str):
            if name == "rebalance":
                params[name] = container.selectbox(
                    label, list(REBALANCE_OPTIONS.keys()),
                    index=list(REBALANCE_OPTIONS.values()).index(base) if base in REBALANCE_OPTIONS.values() else 3,
                    key=key, help=tip)
                params[name] = REBALANCE_OPTIONS[params[name]]
            elif name == "mode":
                params[name] = container.selectbox(label, ["switch", "scale"], key=key, help=tip)
            else:
                params[name] = container.text_input(label, str(base), key=key, help=tip)
        elif isinstance(default, (list, tuple)):
            raw = container.text_input(label, ",".join(map(str, base)) if not isinstance(base, str) else base,
                                       key=key, help=tip)
            params[name] = [v.strip() for v in str(raw).split(",") if v.strip()]
    return params
def metric_cards(metrics: dict, keys: list[str], n_cols: int = 6) -> None:
    cols = st.columns(n_cols)
    for i, k in enumerate(keys):
        cols[i % n_cols].metric(METRIC_LABELS.get(k, k), format_metric(k, metrics.get(k, 0.0)))


def download_row(df: pd.DataFrame, filename: str, key: str, label: str = "⬇️ 下载 CSV") -> None:
    buf = io.BytesIO()
    df.to_csv(buf, index=True, encoding="utf-8-sig")
    st.download_button(label, buf.getvalue(), file_name=filename, mime="text/csv", key=key)


init_state()
sidebar()
bundle: DataBundle = st.session_state.bundle
config = current_config()

# ---------------- 首次自动加载实时行情（可见进度，不白屏） ----------------
if st.session_state.pop("rt_pending", False):
    _ph = st.empty()
    _pool_kind0 = st.session_state.get("rt_pool", "main_board")
    _pool_limit0 = int(st.session_state.get("rt_limit", 300) or 300)
    _ph.info(f"🔄 正在加载实时行情（{POOL_LABELS.get(_pool_kind0, _pool_kind0)}）："
             "首次约 1~2 分钟，之后只需几秒…")
    _bar = st.progress(0.0)
    try:
        _b = load_realtime(None, 250, quiet=True, pool_kind=_pool_kind0, pool_limit=_pool_limit0,
                           progress_cb=lambda p: _bar.progress(min(float(p), 0.99)))
        _bar.progress(1.0)
        if _b is not None:
            _ph.success(f"✅ 实时行情就绪：{len(_b.symbols)} 只标的，行情时间 {_b.quote_time}")
        else:
            _ph.warning("实时行情加载失败，当前使用离线示例数据；可在左侧点「加载 / 刷新实时数据」重试。")
    except Exception as _exc:
        _bar.empty()
        _ph.warning(f"实时行情加载失败：{_exc}　已切换为离线示例数据。")
    time.sleep(0.4)
    st.rerun()

# ---------------- 实时行情状态条 ----------------
_rt = st.session_state.get("rt_status")
if _rt is not None:
    _qtime = getattr(bundle, "quote_time", None)
    _buy_d, _buy_tip = resolve_buy_date(pd.Timestamp(bundle.calendar[-1]), bundle.calendar)
    st.markdown(
        f"<div class='okbox'>🟢 <b>实时行情已连接</b>　"
        f"行情时间：<b>{_qtime}</b>　|　当前时段：{_rt.session}　|　"
        f"最近交易日：{_rt.last_trade_date.date()}　|　<b>下一买入日期：{_buy_d.date()}</b>"
        f"（{_buy_tip}）　|　标的数：{len(bundle.symbols)}"
        f"{'　|　加载于 ' + str(st.session_state.get('rt_loaded_at')) if st.session_state.get('rt_loaded_at') else ''}"
        f"</div>", unsafe_allow_html=True)
else:
    st.error("⚠️ **当前是离线示例数据，不是实时行情！**（只有 12 只模拟股票，日期停在 2024-12-31，"
             "价格是程序生成的假数据）—— 说明实时行情这次没加载成功。点右边按钮重新加载：")
    _bc1, _bc2 = st.columns([1, 3])
    if _bc1.button("🔄 立即加载实时行情（沪深主板）", type="primary", key="banner_load_rt"):
        _bp = st.progress(0.0, text="正在加载实时行情（约 20 秒）…")
        _bb = load_realtime(None, 250, quiet=True,
                            pool_kind=st.session_state.get("rt_pool", "main_board"),
                            pool_limit=int(st.session_state.get("rt_limit", 300) or 300),
                            force=True,
                            progress_cb=lambda p: _bp.progress(min(float(p), 0.99)))
        if _bb is not None and len(_bb.symbols) > 0:
            _bp.progress(1.0, text="完成")
            st.rerun()
        else:
            _bp.empty()
            st.error("仍然失败：请检查网络/代理（本机如有 Clash 等代理，确保它已开启），"
                     "或先切到「主板活跃前 300」再试。")
    _bc2.caption("实时数据源：新浪（日线+快照）/ 东财（兜底）。若多次失败，可先把股票池改成"
                 "「沪深主板活跃股」减小数据量再加载。")
import plotly.graph_objects as go

# 用「单选页面」代替 st.tabs：st.tabs 会把 11 个页签的内容**全部渲染**，
# 导致手机第一屏就要下载 Plotly(4.5MB) 等大文件；改成单选后只加载当前页面。
PAGES = ["📊 数据总览", "🧪 策略回测", "🔍 参数寻优", "⚖️ 策略对比",
         "🧬 因子选股", "🛡️ 风控分析", "📡 策略信号", "🎯 智能选股",
         "🤖 实盘交易", "📊 IC 分析", "📚 规则说明"]
_sel = st.radio("功能页面", PAGES, horizontal=True, label_visibility="collapsed",
                key="page_selector")
PAGE = _sel


# =========================================================================== #
# Tab 1 · 数据总览
# =========================================================================== #
if PAGE == PAGES[0]:
    st.markdown("### 数据总览")
    _desc_all = bundle.describe()
    desc = _desc_all
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("标的数量", f"{len(bundle.symbols)}")
    c2.metric("交易日数", f"{len(bundle)}")
    c3.metric("区间", f"{bundle.calendar[0]:%Y-%m-%d} ~ {bundle.calendar[-1]:%Y-%m-%d}")
    c4.metric("基准", bundle.benchmark_name if bundle.benchmark is not None else "无")

    _show_charts = st.checkbox("📈 显示图表（K线 / 相关性热力图 · 首次加载较慢，会下载约 4.5MB 前端资源）",
                               value=False, key="data_show_charts")
    left, right = st.columns([3, 2])
    with left:
        sym = st.selectbox("查看标的", bundle.symbols,
                           format_func=lambda s: f"{s} {bundle.name_of(s)}（{bundle.industry_of(s)}）",
                           key="data_symbol")
        df = ind.add_all(bundle.get(sym))
        span = st.radio("显示区间", ["近 6 个月", "近 1 年", "近 3 年", "全部"], index=2,
                        horizontal=True, key="data_span")
        n_map = {"近 6 个月": 120, "近 1 年": 250, "近 3 年": 750, "全部": len(df)}
        if _show_charts:
            st.plotly_chart(ch.price_chart(df.tail(n_map[span]), title=f"{sym} {bundle.name_of(sym)}"),
                            width="stretch", config=ch.PLOTLY_CONFIG)
        else:
            st.dataframe(to_cn(df.tail(30).reset_index()), width="stretch", height=300)
            st.caption("↑ 勾选上方「显示图表」可查看 K 线图（手机建议按需开启）")
    with right:
        st.markdown(f"#### 标的概览（共 {len(desc)} 只，按成交额取前 200 只展示）")
        _top = desc.sort_values("amount20_wan" if "amount20_wan" in desc.columns else "last_close",
                                ascending=False).head(200) if len(desc) > 200 else desc
        show = to_cn(_top.copy())
        for col in ("年化波动", "区间收益"):
            if col in show.columns:
                show[col] = show[col].map(lambda v: f"{v:.1%}" if isinstance(v, (int, float)) and pd.notna(v) else "--")
        if "是否ST" in show.columns:
            show["是否ST"] = show["是否ST"].map({True: "是", False: "否"})
        st.dataframe(show, width="stretch", height=360)
        st.markdown("#### 收益相关性（成交额前 30 只）")
        _amt = bundle.amount_matrix().tail(20).mean().sort_values(ascending=False)
        _top30 = _amt.head(30).index.tolist()
        rets = bundle.returns_matrix()[_top30].dropna(how="all").fillna(0.0)
        if _show_charts:
            st.plotly_chart(ch.correlation_heatmap(correlation_summary(rets), height=380),
                            width="stretch", config=ch.PLOTLY_CONFIG)
        else:
            st.caption("（相关性热力图按需加载）")
    st.markdown("#### 数据导出")
    c1, c2 = st.columns([1, 3])
    if c1.button("生成 CSV", key="exp_data"):
        st.session_state["_long"] = bundle.to_long()
    if "_long" in st.session_state:
        download_row(st.session_state["_long"].set_index("date"), "ashare_prices.csv", "dl_data")
    st.caption("提示：本地 CSV 支持列名 open/high/low/close/volume/amount 或中文列名，"
               "指数基准需自行提供（列名 close、索引日期）。")


# =========================================================================== #
# Tab 2 · 策略回测
# =========================================================================== #
if PAGE == PAGES[1]:
    st.markdown("### 策略回测")
    c1, c2 = st.columns([2, 3])
    with c1:
        bt_key = strategy_picker("策略", key="bt_strategy_key")
        cls = REGISTRY[bt_key]
        st.markdown(f"<div class='small'>{cls.description}</div>", unsafe_allow_html=True)
        st.markdown("<div class='small'>提示：信号在 T 日收盘生成，T+1 开盘成交，已含"
                    "涨跌停/停牌/T+1/整手/费用/滑点约束。</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("##### 参数设置（鼠标放在控件上会有解释）")
        bt_params = param_widgets(cls, prefix=f"bt_{bt_key}")
        with st.expander("📖 这些参数是什么意思？（点开看完整说明表）", expanded=False):
            st.dataframe(pd.DataFrame(describe_params(cls)), width="stretch", height=300)
    run_col, _ = st.columns([1, 4])
    if run_col.button("🚀 开始回测", type="primary", width="stretch", key="run_bt"):
        try:
            strat = build_strategy(bt_key, **bt_params)
            prog = st.progress(0.0, text="回测中…")
            bt = Backtester(config)
            res = bt.run_strategy(bundle, strat, name=f"{cls.display_name}",
                                  progress_cb=lambda p: prog.progress(min(float(p), 0.99)))
            prog.progress(1.0, text="完成")
            st.session_state.result = res
            st.session_state.result_key = bt_key
        except Exception as exc:
            st.error(f"回测失败：{exc}")

    res = st.session_state.get("result")
    if res is None:
        st.info("在上方选择策略与参数后点击「开始回测」。")
    else:
        st.markdown(f"#### 绩效概览 · {res.name}")
        metric_cards(res.metrics, ["total_return", "annual_return", "annual_vol", "sharpe",
                                   "max_drawdown", "calmar", "win_rate", "profit_factor",
                                   "trade_count", "turnover", "exposure", "excess_return"])
        c1, c2 = st.columns([3, 2])
        with c1:
            st.plotly_chart(ch.equity_chart(res.equity, res.benchmark, title="策略净值（归一化）"),
                            width="stretch", config=ch.PLOTLY_CONFIG)
        with c2:
            st.plotly_chart(ch.yearly_bar(res.yearly_returns(), height=460),
                            width="stretch", config=ch.PLOTLY_CONFIG)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(ch.drawdown_chart(res.equity), width="stretch",
                            config=ch.PLOTLY_CONFIG)
        with c2:
            st.plotly_chart(ch.monthly_heatmap(res.monthly_table()), width="stretch",
                            config=ch.PLOTLY_CONFIG)
        st.plotly_chart(ch.weights_area(res.weights), width="stretch", config=ch.PLOTLY_CONFIG)

        st.markdown("#### 交易明细与买卖点")
        traded = sorted(res.trades["symbol"].unique().tolist()) if len(res.trades) else []
        if traded:
            c1, c2 = st.columns([1, 3])
            pick = c1.selectbox("查看标的", traded,
                                format_func=lambda s: f"{s} {bundle.name_of(s)}", key="bt_trade_sym")
            kdf = ind.add_all(bundle.get(pick))
            tsub = res.trades[res.trades["symbol"] == pick]
            c2.metric("该标的成交笔数", f"{len(tsub)}")
            st.plotly_chart(ch.price_chart(kdf, tsub, title=f"{pick} 买卖点"),
                            width="stretch", config=ch.PLOTLY_CONFIG)
            st.plotly_chart(ch.round_trip_scatter(res.round_trips), width="stretch",
                            config=ch.PLOTLY_CONFIG)
            tc1, tc2 = st.columns(2)
            with tc1:
                st.markdown("##### 成交明细")
                st.dataframe(to_cn(res.trades), width="stretch", height=320)
            with tc2:
                st.markdown("##### 回合交易")
                st.dataframe(to_cn(res.round_trips), width="stretch", height=320)
        else:
            st.info("本次回测没有成交记录（策略可能长时间空仓，或数据区间过短）。")
        c1, c2, c3 = st.columns([1, 1, 3])
        download_row(res.stats_table(), "metrics.csv", "dl_metrics", "⬇️ 绩效指标")
        download_row(res.trades.set_index("date") if len(res.trades) else res.trades,
                     "trades.csv", "dl_trades", "⬇️ 成交明细")
        if c3.button("📥 导出 Excel（多表）", key="export_xlsx"):
            try:
                p = EXPORT_DIR / f"backtest_{bt_key}_{dt.datetime.now():%Y%m%d_%H%M%S}.xlsx"
                res.to_excel(p)
                st.success(f"已导出：{p}")
            except Exception as exc:
                st.error(f"导出失败（需要 openpyxl）：{exc}")
        with st.expander("回测参数与事件日志"):
            st.dataframe(res.config["config"], width="stretch")
            if len(res.events):
                st.dataframe(to_cn(res.events), width="stretch")


# =========================================================================== #
# Tab 3 · 参数寻优
# =========================================================================== #
if PAGE == PAGES[2]:
    st.markdown("### 参数寻优（样本内 / 样本外 + 稳健性）")
    c1, c2, c3 = st.columns([2, 3, 2])
    with c1:
        opt_key = strategy_picker("策略", key="opt_strategy_key")
        opt_cls = REGISTRY[opt_key]
        st.markdown(f"<div class='small'>{opt_cls.description}</div>", unsafe_allow_html=True)
    space = opt_cls.param_space
    with c2:
        if not space:
            st.warning("该策略没有预置可优化参数。")
            chosen = []
        else:
            chosen = st.multiselect("选择要优化的参数", list(space.keys()),
                                    default=list(space.keys())[:2], key="opt_params")
            grid = {}
            for name in chosen:
                default_vals = ",".join(map(str, space[name]))
                raw = st.text_input(f"{name} 候选值（逗号分隔）", default_vals, key=f"opt_grid_{name}")
                vals = []
                for v in raw.split(","):
                    v = v.strip()
                    if not v:
                        continue
                    try:
                        vals.append(int(v))
                    except ValueError:
                        try:
                            vals.append(float(v))
                        except ValueError:
                            vals.append(v)
                if vals:
                    grid[name] = vals
    with c3:
        mode = st.radio("搜索方式", ["网格搜索", "随机搜索"], key="opt_mode")
        target_metric = st.selectbox("优化目标", ["sharpe", "annual_return", "calmar", "total_return"],
                                     format_func=lambda k: METRIC_LABELS.get(k, k), key="opt_metric")
        train_ratio = st.slider("样本内比例", 0.4, 0.85, 0.7, 0.05, key="opt_ratio")
        n_iter = st.number_input("随机搜索次数", 10, 500, 60, 10, key="opt_niter")
        n_jobs = st.number_input("并行线程", 1, 16, 4, 1, key="opt_jobs")

    if st.button("🔍 开始寻优", type="primary", key="run_opt") and space and chosen:
        try:
            prog = st.progress(0.0, text="寻优中…")
            cb = lambda p: prog.progress(min(float(p), 0.99))  # noqa: E731
            if mode == "网格搜索":
                results = grid_search(opt_key, grid, bundle, config, target_metric,
                                      train_ratio=train_ratio, n_jobs=int(n_jobs), progress_cb=cb)
            else:
                results = random_search(opt_key, grid, bundle, n_iter=int(n_iter), config=config,
                                        metric=target_metric, train_ratio=train_ratio,
                                        n_jobs=int(n_jobs), progress_cb=cb)
            prog.progress(1.0, text="完成")
            st.session_state.opt_results = results
            st.session_state.opt_key = opt_key
        except Exception as exc:
            st.error(f"寻优失败：{exc}")

    opt = st.session_state.get("opt_results")
    if opt is not None and not opt.empty:
        st.markdown("#### 寻优结果（按样本外得分排序）")
        top_row = opt.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("最优样本外得分", f"{top_row['oos_score']:.3f}")
        c2.metric("样本内得分", f"{top_row['is_score']:.3f}")
        c3.metric("过拟合差距", f"{top_row['overfit_gap']:.3f}")
        c4.metric("样本外年化", f"{top_row.get('oos_annual_return', 0):.1%}")
        st.dataframe(opt.head(50).style.format({c: "{:.3f}" for c in opt.columns
                                                if opt[c].dtype.kind in "fc"}), width="stretch", height=340)
        pcol = [c for c in opt.columns if c in space]
        if len(pcol) >= 2:
            c1, c2 = st.columns(2)
            x, y = c1.selectbox("热力图 X 轴", pcol, index=0, key="heat_x"), \
                c2.selectbox("热力图 Y 轴", pcol, index=1, key="heat_y")
            if x != y:
                piv = opt.pivot_table(index=y, columns=x, values="oos_score", aggfunc="mean")
                fig = go.Figure(go.Heatmap(z=piv.to_numpy(), x=[str(v) for v in piv.columns],
                                           y=[str(v) for v in piv.index],
                                           colorscale=[[0, ch.DOWN], [0.5, "#f5f5f5"], [1, ch.UP]],
                                           zmid=0, text=np.round(piv.to_numpy(), 2),
                                           texttemplate="%{text}", colorbar=dict(title="OOS 得分")))
                fig.update_layout(height=380, title=f"样本外得分热力图：{x} × {y}",
                                  template="plotly_white", xaxis_title=x, yaxis_title=y)
                st.plotly_chart(fig, width="stretch", config=ch.PLOTLY_CONFIG)
        c1, c2 = st.columns(2)
        with c1:
            p = c1 = st.selectbox("敏感性分析参数", pcol, key="sens_param") if pcol else None
            if p:
                st.dataframe(sensitivity_table(opt, p), width="stretch")
        with c2:
            st.markdown("##### 推荐参数（兼顾样本外收益与稳健性）")
            best = pick_stable_params(opt)
            st.json(best)
            if st.button("↩️ 用推荐参数回测", key="use_best"):
                try:
                    strat = build_strategy(opt_key, **{k: v for k, v in best.items()
                                                       if k in opt_cls.default_params})
                    st.session_state.result = Backtester(config).run_strategy(bundle, strat)
                    st.success("已回测，请到「策略回测」页签查看结果")
                except Exception as exc:
                    st.error(f"失败：{exc}")
        download_row(opt, "optimization.csv", "dl_opt", "⬇️ 下载寻优结果")
        st.markdown("#### 滚动前推（Walk-Forward）验证")
        if st.button("▶️ 运行 Walk-Forward", key="run_wf"):
            try:
                wf = walk_forward(opt_key, pick_stable_params(opt), bundle, config, n_splits=5)
                st.dataframe(wf, width="stretch")
            except Exception as exc:
                st.error(f"失败：{exc}")
    else:
        st.info("选择参数候选值后点击「开始寻优」。样本外验证可有效识别参数过拟合。")


# =========================================================================== #
# Tab 4 · 策略对比
# =========================================================================== #
if PAGE == PAGES[3]:
    st.markdown("### 多策略横向对比（同数据、同成本、同规则）")
    all_keys = list(REGISTRY.keys())
    default_pick = [k for k in ["dual_ma", "momentum_rotation", "low_vol", "turtle"] if k in all_keys]
    picked = st.multiselect("选择要对比的策略", all_keys, default=default_pick,
                            format_func=lambda k: f"{REGISTRY[k].display_name}｜{REGISTRY[k].category}",
                            key="cmp_keys")
    if st.button("⚖️ 运行对比", type="primary", key="run_cmp") and picked:
        try:
            prog = st.progress(0.0, text="对比回测中…")
            strategies = {REGISTRY[k].display_name: build_strategy(k) for k in picked}
            res_list = compare_strategies(bundle, strategies, config,
                                          progress_cb=lambda p: prog.progress(min(float(p), 0.99)))
            prog.progress(1.0, text="完成")
            st.session_state.multi_results = res_list
        except Exception as exc:
            st.error(f"失败：{exc}")
    multi = st.session_state.get("multi_results") or {}
    if multi:
        fig = go.Figure()
        for i, (name, r) in enumerate(multi.items()):
            e = r.equity / r.equity.iloc[0]
            fig.add_trace(go.Scatter(x=e.index, y=e, name=name,
                                     line=dict(width=1.8, color=ch.PALETTE[i % len(ch.PALETTE)])))
        if bundle.benchmark is not None:
            b = bundle.benchmark.reindex(bundle.calendar).ffill().dropna()
            fig.add_trace(go.Scatter(x=b.index, y=b / b.iloc[0], name="基准",
                                     line=dict(color="#9aa0a6", width=1.4, dash="dot")))
        fig.update_layout(height=480, title="各策略净值对比", template="plotly_white",
                          hovermode="x unified", yaxis_title="净值")
        st.plotly_chart(fig, width="stretch", config=ch.PLOTLY_CONFIG)

        rows = []
        for name, r in multi.items():
            m = r.metrics
            rows.append({"策略": name, "累计收益": m.get("total_return", 0),
                         "年化收益": m.get("annual_return", 0), "年化波动": m.get("annual_vol", 0),
                         "夏普": m.get("sharpe", 0), "最大回撤": m.get("max_drawdown", 0),
                         "卡玛": m.get("calmar", 0), "胜率": m.get("win_rate", 0),
                         "盈亏比": m.get("profit_factor", 0), "换手率": m.get("turnover", 0),
                         "交易次数": m.get("trade_count", 0)})
        cmp_df = pd.DataFrame(rows).sort_values("夏普", ascending=False)
        st.dataframe(cmp_df.style.format({"累计收益": "{:.1%}", "年化收益": "{:.1%}",
                                          "年化波动": "{:.1%}", "最大回撤": "{:.1%}",
                                          "胜率": "{:.1%}", "夏普": "{:.2f}", "卡玛": "{:.2f}",
                                          "盈亏比": "{:.2f}", "换手率": "{:.2f}", "交易次数": "{:.0f}"}),
                     width="stretch")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("##### 回撤对比")
            dd_fig = go.Figure()
            for i, (name, r) in enumerate(multi.items()):
                dd = r.equity / r.equity.cummax() - 1.0
                dd_fig.add_trace(go.Scatter(x=dd.index, y=dd, name=name,
                                            line=dict(width=1.3, color=ch.PALETTE[i % len(ch.PALETTE)])))
            dd_fig.update_layout(height=340, template="plotly_white", yaxis_tickformat=".1%",
                                 hovermode="x unified")
            st.plotly_chart(dd_fig, width="stretch", config=ch.PLOTLY_CONFIG)
        with c2:
            st.markdown("##### 策略相关性（日收益）")
            ret_df = pd.DataFrame({name: r.equity.pct_change() for name, r in multi.items()}).dropna()
            st.plotly_chart(ch.correlation_heatmap(ret_df.corr().round(3), height=340),
                            width="stretch", config=ch.PLOTLY_CONFIG)
        download_row(cmp_df, "compare.csv", "dl_cmp", "⬇️ 下载对比结果")
    else:
        st.info("选择 2 个以上策略后点击「运行对比」。")

# =========================================================================== #
# Tab 5 · 因子选股
# =========================================================================== #
if PAGE == PAGES[4]:
    st.markdown("### 多因子选股与因子研究")
    # 因子面板按需计算：全主板 3000+ 只计算一次约 20 秒，算完缓存在会话里，
    # 避免每次点击/刷新都重算（这是之前"卡住"的主因）。
    _panel_key = (len(bundle.symbols), str(bundle.calendar[-1]), str(bundle.calendar[0]))
    panel = st.session_state.get("_factor_panel")
    if panel is not None and st.session_state.get("_factor_panel_key") != _panel_key:
        panel = None
    pc1, pc2 = st.columns([1, 4])
    if panel is None:
        if pc1.button("🧮 计算因子面板", key="calc_panel", type="primary"):
            with st.spinner(f"正在计算 {len(bundle.symbols)} 只股票的因子（约 20 秒）…"):
                panel = factor_panel(bundle)
            st.session_state._factor_panel = panel
            st.session_state._factor_panel_key = _panel_key
            st.rerun()
        pc2.caption("⚠️ 因子面板尚未计算。上面「一键选股」不需要它；"
                    "只有本页签的因子回测 / IC 分析 / 因子明细需要先计算。")
    else:
        _cov = len(panel.get("mom60").columns) if "mom60" in panel else 0
        _pool_now = st.session_state.get("rt_pool_info")
        pc1.success(f"因子面板已就绪（覆盖 {_cov} 只）")
        pc2.caption(f"覆盖标的数取决于左侧「股票池范围」：当前 "
                    f"{_pool_now.label + f'（{len(_pool_now.symbols)} 只）' if _pool_now is not None else f'{len(bundle.symbols)} 只'}。"
                    f" 想要全沪深主板：左侧把股票池改成「沪深主板（全部）」→ 点「🔄 加载 / 刷新实时数据」。")
    c1, c2, c3 = st.columns([3, 2, 2])
    with c1:
        st.markdown("##### 因子权重")
        chosen_factors = st.multiselect("启用因子", list(FACTOR_LABELS.keys()),
                                        default=["mom60", "rev5", "vol20", "trend"],
                                        format_func=lambda k: FACTOR_LABELS.get(k, k),
                                        key="fac_keys")
        wdict = {}
        fcols = st.columns(2)
        for i, k in enumerate(chosen_factors):
            wdict[k] = fcols[i % 2].slider(FACTOR_LABELS.get(k, k), -1.0, 1.0, 0.25, 0.05,
                                           key=f"fac_w_{k}")
    with c2:
        st.markdown("##### 组合构建")
        fac_topk = st.number_input("持仓数量", 1, 30, 5, 1, key="fac_topk")
        fac_reb = st.selectbox("调仓频率", list(REBALANCE_OPTIONS.keys()), index=3, key="fac_reb")
        fac_maxw = st.slider("单票上限", 0.05, 1.0, 0.25, 0.05, key="fac_maxw")
        fac_weight_mode = st.radio("加权方式", ["等权", "打分加权"], horizontal=True, key="fac_wm")
    with c3:
        st.markdown("##### 择时与过滤")
        fac_timing = st.checkbox("大盘均线择时", True, key="fac_timing")
        fac_bench_ma = st.slider("择时均线", 10, 250, 60, 5, key="fac_bench_ma")
        st.caption("因子已做方向统一（数值越大越好）与横截面标准化。")

    if panel is None:
        st.info("请先点上面的「🧮 计算因子面板」，再使用因子回测 / IC 分析。")
    if st.button("🧬 因子组合回测", type="primary", key="run_factor") and chosen_factors and panel is not None:
        try:
            score = composite_score({k: panel[k] for k in chosen_factors if k in panel}, wdict)
            for s in score.columns:
                if bundle.is_st(s):
                    score[s] = np.nan
            score = score.where(~bundle.suspended_matrix())
            from ashare_quant.strategies.base import rebalance_mask
            from ashare_quant.strategies.trend import hold_between_rebalance
            w = score_to_weights(score, top_k=int(fac_topk), max_weight=float(fac_maxw),
                                 weight_mode="score" if fac_weight_mode == "打分加权" else "equal")
            w = hold_between_rebalance(w, rebalance_mask(w.index, REBALANCE_OPTIONS[fac_reb]))
            if fac_timing and bundle.benchmark is not None:
                ref = bundle.benchmark.dropna()
                ma = ind.sma(ref, int(fac_bench_ma))
                timing = ((ref > ma) & (ind.sma(ref, max(5, int(fac_bench_ma) // 4)) > ma)).astype(float)
                w = w.mul(timing.reindex(w.index).fillna(0.0), axis=0)
            st.session_state.factor_result = Backtester(config).run(bundle, w, name="多因子组合")
            st.success("回测完成")
        except Exception as exc:
            st.error(f"失败：{exc}")

    fr = st.session_state.get("factor_result")
    if fr is not None:
        metric_cards(fr.metrics, ["total_return", "annual_return", "annual_vol", "sharpe",
                                  "max_drawdown", "calmar", "win_rate", "turnover"])
        c1, c2 = st.columns([3, 2])
        with c1:
            st.plotly_chart(ch.equity_chart(fr.equity, fr.benchmark), width="stretch",
                            config=ch.PLOTLY_CONFIG)
        with c2:
            st.plotly_chart(ch.weights_area(fr.weights, height=440), width="stretch",
                            config=ch.PLOTLY_CONFIG)

    st.markdown("---")
    st.markdown("#### 因子有效性检验（IC / 分层）")
    c1, c2, c3 = st.columns([1, 1, 3])
    horizon = c1.number_input("未来收益窗口（交易日）", 1, 60, 20, 1, key="ic_horizon")
    q = c2.number_input("分层数", 2, 10, 5, 1, key="ic_q")
    if c3.button("📐 计算 IC 与分层收益", key="run_ic") and panel is not None:
        with st.spinner("计算中…"):
            fwd = forward_returns(bundle, int(horizon))
            ic_df = factor_ic(panel, fwd, method="spearman")
            st.session_state["ic_df"] = ic_df
            st.session_state["ic_sum"] = ic_summary(ic_df)
            st.session_state["ql"] = quantile_returns(composite_score(panel), fwd, int(q))
    if "ic_sum" in st.session_state:
        c1, c2 = st.columns([2, 3])
        with c1:
            summ = st.session_state["ic_sum"].copy()
            st.dataframe(summ.style.format({"IC均值": "{:.4f}", "IC标准差": "{:.4f}",
                                            "ICIR": "{:.2f}", "IC>0占比": "{:.1%}", "t值": "{:.2f}",
                                            "样本数": "{:.0f}"}), width="stretch", height=380)
        with c2:
            st.plotly_chart(ch.ic_bar(st.session_state["ic_sum"]), width="stretch",
                            config=ch.PLOTLY_CONFIG)
        ql = st.session_state.get("ql")
        if ql is not None and not ql.empty:
            cum = (1 + ql.fillna(0.0)).cumprod()
            fig = go.Figure()
            for i, col in enumerate(cum.columns):
                fig.add_trace(go.Scatter(x=cum.index, y=cum[col], name=col,
                                         line=dict(width=1.6, color=ch.PALETTE[i % len(ch.PALETTE)])))
            fig.update_layout(height=380, template="plotly_white", title="因子分层累计收益（Q1 最低 → Qn 最高）",
                              hovermode="x unified", yaxis_title="累计净值")
            st.plotly_chart(fig, width="stretch", config=ch.PLOTLY_CONFIG)

    with st.expander("最新截面因子明细（按综合分排序）", expanded=False):
        latest = bundle.calendar[-1]
        ft = factor_table(panel, latest) if panel is not None else pd.DataFrame()
        if not ft.empty:
            ft.index = [f"{s} {bundle.name_of(s)}" for s in ft.index]
            st.dataframe(ft.style.format(precision=3, na_rep="--"), width="stretch", height=420)


# =========================================================================== #
# Tab 6 · 风控分析
# =========================================================================== #
if PAGE == PAGES[5]:
    st.markdown("### 组合风控与压力测试")
    res = st.session_state.get("result")
    if res is None:
        st.info("请先在「策略回测」页签运行一次回测，再回到本页查看风控体检。")
    else:
        st.markdown(f"#### 风险体检 · {res.name}")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.dataframe(risk_report(res, bundle), width="stretch", height=420)
        with c2:
            w_recent = res.weights.tail(120)
            risk = portfolio_risk(w_recent, bundle.returns_matrix().reindex(w_recent.index).fillna(0.0))
            risk_df = pd.DataFrame([{"指标": k, "数值": (f"{v:.2%}" if abs(v) < 10 else f"{v:.2f}")}
                                    for k, v in risk.items()])
            st.dataframe(risk_df, width="stretch", height=420)

        st.markdown("#### 行业暴露（近 120 交易日平均）")
        exp = industry_exposure(w_recent, bundle.meta).mean().sort_values(ascending=False)
        c1, c2 = st.columns([3, 2])
        with c1:
            st.plotly_chart(ch.factor_exposure_bar(exp, height=380), width="stretch",
                            config=ch.PLOTLY_CONFIG)
        with c2:
            st.markdown("##### 压力测试")
            latest_w = res.weights.iloc[-1]
            scen = {"大盘急跌 -5%": {"beta_adjust": -0.05},
                    "大盘急跌 -8%": {"beta_adjust": -0.08},
                    "全市场 -10%": {"uniform": -0.10},
                    "全市场 +5%": {"uniform": 0.05}}
            beta = bundle.meta["beta"].reindex(latest_w.index) if "beta" in bundle.meta.columns else None
            st_df = stress_test(latest_w, bundle.returns_matrix(), scen, beta)
            if not st_df.empty:
                st.dataframe(st_df.style.format({"组合冲击": "{:.2%}"}), width="stretch")
            else:
                st.caption("暂无压力测试结果")

        st.markdown("#### 滚动绩效与相关性")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(ch.rolling_chart(rolling_metrics(res.equity, 120)),
                            width="stretch", config=ch.PLOTLY_CONFIG)
        with c2:
            held = res.weights.iloc[-1]
            held = held[held > 0].index.tolist() or res.weights.mean().nlargest(8).index.tolist()
            corr = bundle.returns_matrix()[held].tail(250).corr().round(3)
            st.plotly_chart(ch.correlation_heatmap(corr, height=380, title="持仓相关性"),
                            width="stretch", config=ch.PLOTLY_CONFIG)
        with st.expander("止损 / 熔断事件日志", expanded=False):
            if len(res.events):
                st.dataframe(to_cn(res.events), width="stretch")
            else:
                st.caption("本次回测未触发止损或熔断（可在侧边栏「风控参数」中开启）。")


# =========================================================================== #
# Tab 7 · 实盘信号
# =========================================================================== #
if PAGE == PAGES[6]:
    st.markdown("### 实盘 / 模拟盘信号")
    st.markdown("<div class='warnbox'>⚠️ 本页仅生成<strong>研究用信号与模拟记账</strong>，"
                "不接入任何券商下单接口。实盘交易请使用券商合规渠道，并自担风险。</div>",
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2, 3, 2])
    with c1:
        sig_key = strategy_picker("策略", key="sig_strategy_key")
        sig_cls = REGISTRY[sig_key]
        st.markdown(f"<div class='small'>{sig_cls.description}</div>", unsafe_allow_html=True)
    with c2:
        sig_params = param_widgets(sig_cls, prefix=f"sig_{sig_key}")
    with c3:
        sig_capital = st.number_input("账户资金（元）", 10_000.0, 1e9, 1_000_000.0, 10_000.0,
                                      key="sig_capital")
        sig_band = st.slider("信号阈值（权重变化）", 0.0, 0.05, 0.005, 0.005, key="sig_band")
        if st.button("📡 生成最新信号", type="primary", key="run_signal"):
            try:
                strat = build_strategy(sig_key, **sig_params)
                sig = latest_signals(bundle, strat, capital=float(sig_capital), band=float(sig_band))
                st.session_state.signals = sig
            except Exception as exc:
                st.error(f"失败：{exc}")
    sig = st.session_state.get("signals")
    if sig is not None and not sig.empty:
        d = pd.Timestamp(sig["date"].iloc[0]).date()
        st.markdown(f"#### 调仓清单 · 信号日 {d}（次日开盘执行）")
        summary = sig["action"].value_counts().to_dict()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("新建仓/加仓", summary.get("新建仓", 0) + summary.get("加仓", 0))
        c2.metric("减仓/清仓", summary.get("减仓", 0) + summary.get("清仓", 0))
        c3.metric("持有", summary.get("持有", 0))
        c4.metric("目标持仓数", int((sig["target_weight"] > 0).sum()))
        show = sig.copy()
        show["date"] = pd.to_datetime(show["date"]).dt.date
        st.dataframe(show.style.format({"close": "{:.2f}", "limit_up": "{:.2f}", "limit_down": "{:.2f}",
                                        "cur_weight": "{:.1%}", "target_weight": "{:.1%}",
                                        "delta_weight": "{:+.1%}", "amount": "{:,.0f}",
                                        "shares": "{:,.0f}"}, na_rep="--"),
                     width="stretch", height=420)
        st.code(signal_summary_text(sig), language="text")
        download_row(sig, f"signals_{d}.csv", "dl_signal", "⬇️ 导出信号 CSV")
        st.markdown("---")
        st.markdown("#### 模拟盘账户（按信号记账）")
        acct_path = EXPORT_DIR / "paper_account.json"
        acct = PaperAccount.load(acct_path, initial_cash=float(sig_capital))
        prices = bundle.close_matrix().ffill().iloc[-1].fillna(0.0).to_dict()
        c1, c2, c3 = st.columns([1, 1, 2])
        if c1.button("▶️ 按信号成交（按最新收盘价模拟）", key="apply_paper"):
            fills = acct.apply_signals(sig, prices)
            st.success(f"模拟成交 {len(fills)} 笔")
            if not fills.empty:
                st.dataframe(fills, width="stretch")
        if c2.button("♻️ 重置模拟盘", key="reset_paper"):
            acct = PaperAccount(initial_cash=float(sig_capital), path=str(acct_path))
            acct.save()
            st.session_state.paper_reset = True
            st.success("模拟盘已重置")
        with c3:
            st.dataframe(acct.summary(prices), width="stretch")
        holdings = acct.holding_table(prices)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### 模拟持仓")
            if holdings.empty:
                st.caption("当前空仓")
            else:
                st.dataframe(holdings.style.format({"shares": "{:,.0f}", "cost": "{:.2f}",
                                                    "price": "{:.2f}", "market_value": "{:,.0f}",
                                                    "pnl": "{:,.0f}", "pnl_pct": "{:.2%}",
                                                    "weight": "{:.1%}"}), width="stretch")
        with c2:
            st.markdown("##### 净值历史")
            if acct.history:
                hist = pd.DataFrame(acct.history)
                hist["date"] = pd.to_datetime(hist["date"])
                fig = go.Figure(go.Scatter(x=hist["date"], y=hist["equity"], mode="lines+markers",
                                           name="模拟盘净值", line=dict(color="#2f6feb", width=2)))
                fig.update_layout(height=320, template="plotly_white", yaxis_title="总资产")
                st.plotly_chart(fig, width="stretch", config=ch.PLOTLY_CONFIG)
            else:
                st.caption("暂无记录，点击「按信号成交」开始跟踪。")
    else:
        st.info("点击「生成最新信号」获取下一交易日的调仓建议。")


# =========================================================================== #
# Tab 8 · 智能选股（含买入日期）
# =========================================================================== #
if PAGE == PAGES[7]:
    st.markdown("### 🎯 数据选股（自动标注买入日期）")
    st.markdown("<div class='okbox'>流程：<b>选股日 T 收盘</b>用数据打分选出股票 → "
                "<b>买入日期 = T 的下一个交易日（自动跳过周末与节假日）</b> → 按限价区间在开盘后执行。<br>"
                "清单中的「买入日期」「买入价区间」「止损价/止盈价」「建议股数」可直接用于下单。</div>",
                unsafe_allow_html=True)

    # ---------------- 一键选股（新手推荐） ----------------
    st.markdown("#### 🚀 一键选股（不用调参数）")
    _pool_info = st.session_state.get("rt_pool_info")
    _pool_txt = (f"{_pool_info.label}｜{len(_pool_info.symbols)} 只" if _pool_info is not None
                 else f"当前数据源｜{len(bundle.symbols)} 只")
    _is_rt = st.session_state.get("rt_status") is not None
    if _is_rt:
        st.caption(f"数据源：🟢 实时行情（{_pool_txt}）　|　"
                   f"行情时间 {getattr(bundle, 'quote_time', '—')}")
    else:
        st.error("⚠️ 当前用的是**离线示例数据**（12 只模拟股票，日期停在 2024-12-31），"
                 "不是实时行情 —— 下面的选股结果没有实际参考价值。")
        if st.button("🔄 先加载实时行情，再选股", type="primary", key="pick_load_rt"):
            _pp = st.progress(0.0, text="正在加载实时行情（约 20 秒）…")
            _pb = load_realtime(None, 250, quiet=True,
                                pool_kind=st.session_state.get("rt_pool", "main_board"),
                                pool_limit=int(st.session_state.get("rt_limit", 300) or 300),
                                force=True,
                                progress_cb=lambda p: _pp.progress(min(float(p), 0.99)))
            if _pb is not None and len(_pb.symbols) > 0:
                st.rerun()
            else:
                st.error("加载失败，请检查网络/代理后重试。")
    qc1, qc2, _qc3 = st.columns([1, 1, 2])
    _quick_cfg = dict(top_k=10, min_amount_yuan=1e8, min_bars=120, trend_ma=60,
                      max_weight=0.10, max_loss_pct=0.12, take_profit_pct=0.20,
                      max_hold_days=60, buy_price_buffer=0.02)
    if qc1.button("🎯 一键选股（稳健·含大盘择时）", type="primary", key="pick_quick_safe"):
        try:
            with st.spinner("正在用实时行情选股（全主板约 20 秒）…"):
                _cfg = SelectionConfig(use_timing=True, **_quick_cfg)
                st.session_state.pick_result = select_stocks(bundle, _cfg, capital=float(config.initial_cash))
                st.session_state.pick_cfg = _cfg
                st.session_state.pick_src = "一键选股（稳健·含大盘择时）"
                st.session_state.pick_sig = None
                st.session_state.pick_at = dt.datetime.now().strftime("%H:%M:%S")
        except Exception as exc:
            st.error(f"选股失败：{exc}")
    if qc2.button("⚡ 一键选股（忽略择时·直接选）", key="pick_quick_force"):
        try:
            with st.spinner("正在用实时行情选股（全主板约 20 秒）…"):
                _cfg = SelectionConfig(use_timing=False, **_quick_cfg)
                st.session_state.pick_result = select_stocks(bundle, _cfg, capital=float(config.initial_cash))
                st.session_state.pick_cfg = _cfg
                st.session_state.pick_src = "一键选股（忽略择时·直接选）"
                st.session_state.pick_sig = None
                st.session_state.pick_at = dt.datetime.now().strftime("%H:%M:%S")
        except Exception as exc:
            st.error(f"选股失败：{exc}")
    with st.expander("📖 使用说明（3 步）· 点开看怎么用", expanded=False):
        st.markdown("""
**第 1 步**：上面两个按钮二选一
- **稳健版**：大盘在均线上方才买，空头时空仓（推荐长期使用，回撤小）
- **直接版**：不管大盘，始终选最强的前 10 只（熊市里回撤会更大）

**第 2 步**：看下面生成的「选股清单」
- **买入日期**：才是你要下单的日子（不是选股日期！）
- **买入价下限~上限**：开盘价超过上限就放弃（别追高）
- **股数**：已经按你的资金和整手规则算好
- **止损价 / 止盈价**：跌破止损卖、涨到止盈卖

**第 3 步**：下单
- 手动：按清单在券商 App 下单（最稳）
- 自动：去「🤖 实盘交易」页签，选通道 → 生成交易计划 → 执行

**想微调**：下面的「自定义参数」可以改选股数量、成交额门槛、止损止盈幅度等。
""")

    c1, c2, c3 = st.columns([3, 2, 2])
    with c1:
        pick_factors = st.multiselect("选股因子（权重可正可负）", list(FACTOR_LABELS.keys()),
                                      default=[k for k in ["mom60", "rev5", "vol20", "trend", "amount20"]
                                               if k in FACTOR_LABELS],
                                      format_func=lambda k: FACTOR_LABELS.get(k, k), key="pick_factors")
        pw: dict = {}
        fcols = st.columns(2)
        for i, k in enumerate(pick_factors):
            pw[k] = fcols[i % 2].slider(FACTOR_LABELS.get(k, k), -1.0, 1.0, 0.25, 0.05, key=f"pw_{k}")
    with c2:
        pick_top = st.number_input("选股数量 Top K", 1, 30, 5, 1, key="pick_top")
        pick_amount = st.number_input("最小日均成交额（万元）", 0.0, 1.0e7, 5000.0, 500.0,
                                      key="pick_amount")
        pick_price = st.slider("价格区间（元）", 0.0, 1000.0, (2.0, 500.0), 1.0, key="pick_price")
        pick_maxchg = st.slider("单日涨跌幅上限（剔除涨跌停）", 0.02, 0.20, 0.095, 0.005,
                                key="pick_maxchg")
    with c3:
        pick_trend = st.checkbox("需站上均线才入选", True, key="pick_trend")
        pick_trend_ma = st.slider("趋势均线", 0, 250, 60, 5, key="pick_trend_ma")
        pick_timing = st.checkbox("大盘择时（空头空仓）", True, key="pick_timing")
        pick_bench_ma = st.slider("择时均线", 10, 250, 60, 5, key="pick_bench_ma")
        pick_use_latest = st.checkbox("用最新交易日选股", True, key="pick_latest")
    if not pick_use_latest:
        pick_date = st.date_input("选股基准日", bundle.calendar[-1].date(), key="pick_date")
    else:
        pick_date = bundle.calendar[-1].date()
    c1, c2, c3, c4 = st.columns(4)
    pick_stop = c1.slider("止损幅度", 0.03, 0.30, 0.12, 0.01, key="pick_stop")
    pick_take = c2.slider("止盈幅度", 0.05, 1.00, 0.20, 0.05, key="pick_take")
    pick_hold = c3.number_input("最长持有（交易日）", 5, 250, 60, 5, key="pick_hold")
    pick_buffer = c4.slider("买入限价上浮", 0.0, 0.05, 0.02, 0.005, key="pick_buffer")
    pick_capital = st.number_input("拟投入资金（元）", 10_000.0, 1.0e9, float(config.initial_cash),
                                   10_000.0, key="pick_capital")
    pick_adv_pct = st.slider("单票占20日均成交额上限(%)", 0.0, 2.0, 0.5, 0.1,
                            key="pick_adv_pct",
                            help="容量约束：单票买入金额不超过20日均成交额的该比例，避免实盘买不进去；0=关闭")

    # 当前参数指纹：用于判断"结果是否已过期"
    _cur_sig = (int(pick_top),
                tuple(sorted((k, round(float(v), 3)) for k, v in pw.items())),
                round(float(pick_amount), 2), float(pick_price[0]), float(pick_price[1]),
                round(float(pick_maxchg), 4), bool(pick_trend), int(pick_trend_ma),
                bool(pick_timing), int(pick_bench_ma), round(float(pick_stop), 3),
                round(float(pick_take), 3), int(pick_hold), round(float(pick_buffer), 4),
                round(float(pick_capital), 2), str(pick_date), bool(pick_use_latest))

    if st.button("🎯 生成选股清单", type="primary", key="run_pick"):
        try:
            sel_cfg = SelectionConfig(
                top_k=int(pick_top), factor_weights=dict(pw) or None,
                min_amount_yuan=float(pick_amount) * 1e4,
                min_price=float(pick_price[0]), max_price=float(pick_price[1]),
                max_day_change=float(pick_maxchg),
                trend_ma=int(pick_trend_ma) if pick_trend else 0,
                use_timing=bool(pick_timing), timing_ma=int(pick_bench_ma),
                total_exposure=min(1.0, float(config.max_position_weight) * int(pick_top)),
                max_weight=float(config.max_position_weight),
                buy_price_buffer=float(pick_buffer),
                max_loss_pct=float(pick_stop), take_profit_pct=float(pick_take),
                max_hold_days=int(pick_hold),
                max_pct_of_adv=float(pick_adv_pct) / 100.0)
            sel_cfg.total_exposure = min(1.0, float(config.max_position_weight) * int(pick_top))
            with st.spinner("正在按最新数据选股…"):
                res = select_stocks(bundle, sel_cfg, capital=float(pick_capital),
                                    as_of=pd.Timestamp(pick_date))
            st.session_state.pick_result = res
            st.session_state.pick_cfg = sel_cfg
            st.session_state.pick_src = "自定义参数"
            st.session_state.pick_sig = _cur_sig
            st.session_state.pick_at = dt.datetime.now().strftime("%H:%M:%S")
        except Exception as exc:
            st.error(f"选股失败：{exc}")

    pres = st.session_state.get("pick_result")
    # 参数改了但没重新计算 → 明确提示结果已过期
    if pres is not None and st.session_state.get("pick_src") == "自定义参数" \
            and st.session_state.get("pick_sig") != _cur_sig:
        st.warning("⚠️ **你修改了选股参数，但下面的清单还是上一次的结果**（不会自动重算）。"
                   "请点上面的「🎯 生成选股清单」重新计算。")
    elif pres is not None and str(st.session_state.get("pick_src", "")).startswith("一键选股"):
        st.info("ℹ️ 下面这份清单来自「**一键选股**」，用的是**推荐默认参数**（Top10、成交额≥1亿、五因子等权）。"
                "你在下面手工调的因子**不会**影响它 —— 想让自定义因子生效，"
                "请点「🎯 生成选股清单」。")
    if pres is not None:
        _src = st.session_state.get("pick_src") or "—"
        st.markdown(f"<div class='small'>当前展示的结果来源：<b>{_src}</b>"
                    + (f"　|　生成于 {st.session_state.get('pick_at', '')}"
                       if st.session_state.get("pick_at") else "") + "</div>",
                    unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("选股日", f"{pres.signal_date:%Y-%m-%d}")
        c2.metric("👉 买入日期", f"{pres.buy_date:%Y-%m-%d}")
        c3.metric("市场状态", pres.market_state)
        c4.metric("建议总仓位", f"{pres.exposure:.0%}")
        c5.metric("入选数量", f"{pres.n_picks} / {pres.universe}")
        if pres.picks.empty:
            if pres.exposure <= 0:
                st.warning(f"本期无入选标的 · 市场状态：{pres.market_state} · 建议仓位 0%\n\n"
                           "📉 **当前大盘是空头**（指数跌破均线），「稳健版」按风控规则**空仓等待**，"
                           "所以入选数量是 0 —— 这是它在保护你，不是出错。\n\n"
                           "如果你现在就想看到具体股票，点下面这个按钮（或点上面「⚡ 直接选」）：")
                if st.button("⚡ 忽略大盘择时，立即选出股票（10 只）", type="primary",
                             key="pick_force_from_empty"):
                    try:
                        _base = st.session_state.get("pick_cfg")
                        _cfg = (dc_replace(_base, use_timing=False) if _base is not None
                                else SelectionConfig(use_timing=False, top_k=10, min_amount_yuan=1e8,
                                                     min_bars=120, trend_ma=60, max_weight=0.10,
                                                     max_loss_pct=0.12, take_profit_pct=0.20))
                        with st.spinner("正在选股（全主板约 20 秒）…"):
                            st.session_state.pick_result = select_stocks(
                                bundle, _cfg, capital=float(config.initial_cash))
                            st.session_state.pick_cfg = _cfg
                        st.rerun()
                    except Exception as exc:
                        st.error(f"选股失败：{exc}")
            else:
                st.warning(f"本期无入选标的 · 市场状态：{pres.market_state} · 建议仓位 {pres.exposure:.0%}\n\n"
                           "所有候选都被硬性条件过滤掉了：可放宽「最小日均成交额」、「需站上均线」"
                           "或调大「单日涨跌幅上限」。")
            if not pres.funnel.empty:
                st.markdown("**筛选漏斗**（看每一步剩下多少只）")
                st.dataframe(pres.funnel, width="stretch")
        else:
            show = to_cn(pres.picks.copy())
            show.insert(0, "序号", range(1, len(show) + 1))
            st.markdown("#### 选股清单（买入日期已标注）")
            cols_show = ["序号", "买入日期", "选股日期", "买入时机", "代码", "名称", "行业", "板块",
                         "排名", "收盘价", "综合分", "目标仓位", "股数", "成交额",
                         "买入价下限", "买入价上限", "止损价", "止盈价", "最长持有天数",
                         "入选理由", "风险提示"]
            show = show[[c for c in cols_show if c in show.columns]]
            st.dataframe(show.style.format({
                "综合分": "{:.3f}", "目标仓位": "{:.1%}", "股数": "{:,.0f}", "成交额": "{:,.0f}",
                "收盘价": "{:.2f}", "买入价下限": "{:.2f}", "买入价上限": "{:.2f}",
                "止损价": "{:.2f}", "止盈价": "{:.2f}"}, na_rep="--"),
                width="stretch", height=320)
            c1, c2 = st.columns([3, 2])
            with c1:
                st.code(pres.summary_text(), language="text")
            with c2:
                st.markdown("**筛选漏斗**")
                st.dataframe(pres.funnel, width="stretch")
            c1, c2, c3 = st.columns([1, 1, 3])
            out = pres.picks.copy()
            buf = io.BytesIO()
            out.to_csv(buf, index=False, encoding="utf-8-sig")
            c1.download_button("⬇️ 导出选股清单 CSV", buf.getvalue(),
                               file_name=f"picks_{pres.signal_date:%Y%m%d}_buy_{pres.buy_date:%Y%m%d}.csv",
                               mime="text/csv", key="dl_picks")
            if c2.button("➡️ 用于实盘计划", key="pick_to_live"):
                st.session_state.pick_for_live = pres
                st.success("已保存。请到「🤖 实盘交易」页签生成委托。")

        # ---------------- 历史选股 + 买入日期标注 ----------------
        st.markdown("---")
        st.markdown("#### 历史选股回看（验证选股质量 + 买入日期标注）")
        h1, h2, h3, h4 = st.columns([1, 1, 1, 2])
        hist_freq = h1.selectbox("选股频率", list(REBALANCE_OPTIONS.keys()), index=1, key="hist_freq")
        hist_periods = h2.number_input("回看期数上限", 6, 200, 24, 6, key="hist_periods")
        hist_horizon = h3.selectbox("验证持有期", [5, 10, 20, 60], index=2, key="hist_horizon")
        if h4.button("📚 回看历史选股", key="run_hist"):
            try:
                with st.spinner("正在按历史每期数据回算选股（不含未来信息）…"):
                    cfg_h = st.session_state.get("pick_cfg") or SelectionConfig(top_k=int(pick_top))
                    hist = select_history(bundle, cfg_h, capital=float(pick_capital),
                                          freq=REBALANCE_OPTIONS[hist_freq],
                                          lookaheads=(int(hist_horizon),))
                st.session_state.pick_hist = hist
            except Exception as exc:
                st.error(f"回看失败：{exc}")
        hist = st.session_state.get("pick_hist")
        if hist is not None and not hist.empty:
            shown = hist[hist["symbol"].notna()].copy()
            shown = shown.tail(int(hist_periods) * max(1, int(pick_top)))
            shown = to_cn(shown)
            keep = [c for c in ["选股日期", "买入日期", "代码", "名称", "收盘价", "综合分",
                                "目标仓位", "买入价上限", "止损价",
                                f"买入后{int(hist_horizon)}日收益"] if c in shown.columns]
            st.dataframe(shown[keep].style.format({
                "收盘价": "{:.2f}", "综合分": "{:.3f}", "目标仓位": "{:.1%}",
                "买入价上限": "{:.2f}", "止损价": "{:.2f}",
                f"买入后{int(hist_horizon)}日收益": "{:+.2%}"}, na_rep="--"),
                width="stretch", height=340)
            c1, c2 = st.columns([2, 3])
            with c1:
                st.dataframe(selection_stats(hist, horizons=(5, 20, 60)).style.format(
                    {"个股平均收益": "{:+.2%}", "个股胜率": "{:.1%}", "组合平均收益": "{:+.2%}",
                     "组合胜率": "{:.1%}", "最好": "{:+.2%}", "最差": "{:+.2%}"}, na_rep="--"),
                    width="stretch")
            with c2:
                sym_pick = st.selectbox("选择标的查看买入日期标注",
                                        sorted(hist["symbol"].dropna().unique().tolist())[:200],
                                        key="hist_symbol")
                kdf = ind.add_all(bundle.get(sym_pick))
                marks = hist[hist["symbol"] == sym_pick][["buy_date", "close"]].dropna().copy()
                marks = marks.rename(columns={"buy_date": "date", "close": "price"})
                marks["side"] = "buy"
                marks["shares"] = 100
                st.plotly_chart(ch.price_chart(kdf, marks if not marks.empty else None,
                                               title=f"{sym_pick} 历史买入日期（▲ = 买入日）",
                                               height=460),
                                width="stretch", config=ch.PLOTLY_CONFIG)
            buf = io.BytesIO()
            hist.to_csv(buf, index=False, encoding="utf-8-sig")
            st.download_button("⬇️ 导出历史选股 CSV", buf.getvalue(),
                               file_name="selection_history.csv", mime="text/csv", key="dl_hist")
    else:
        st.info("点击「🎯 生成选股清单」，程序会按最新数据打分并给出**买入日期**。")


# =========================================================================== #
# Tab 9 · 实盘交易
# =========================================================================== #
if PAGE == PAGES[8]:
    st.markdown("### 🤖 实盘交易")
    st.markdown("<div class='warnbox'><b>真实资金风险提示：</b>本页可以连接券商通道真实下单。"
                "默认 <b>dry-run</b>（只校验、不发送委托）。请务必先在「本地模拟盘」跑通，"
                "再用小资金验证，并确认所有风控参数。程序按研究用途提供，交易决策与盈亏由你自行承担。</div>",
                unsafe_allow_html=True)
    bt = available_brokers()
    bname = dict(zip(bt["key"], bt["名称"]))
    bcond = dict(zip(bt["key"], bt["前置条件"]))
    c1, c2, c3 = st.columns([2, 2, 3])
    with c1:
        _broker_keys = ["sim", "manual"] if IS_CLOUD else bt["key"].tolist()
        broker_key = st.selectbox("交易通道", _broker_keys,
                                  format_func=lambda k: bname.get(k, k), key="live_broker")
        st.caption(f"前置条件：{bcond.get(broker_key, '')}")
        broker_kwargs: dict = {}
        if broker_key == "qmt":
            broker_kwargs["userdata_path"] = st.text_input(
                "QMT userdata_mini 目录", r"D:\国金QMT交易端\userdata_mini", key="qmt_userdata")
            broker_kwargs["account_id"] = st.text_input("QMT 资金账号", "", key="qmt_account")
            st.caption("需先在本机打开并登录 QMT 客户端。")
        elif broker_key == "easytrader":
            broker_kwargs["client"] = st.selectbox("客户端类型", ["ths", "tdx", "ht", "gj"],
                                                   format_func=lambda k: {"ths": "同花顺", "tdx": "通达信",
                                                                          "ht": "华泰", "gj": "国金"}.get(k, k),
                                                   key="et_client")
            broker_kwargs["client_path"] = st.text_input(
                "下单程序路径（xiadan.exe）", r"C:\同花顺软件\同花顺\xiadan.exe", key="et_path")
            st.caption("需先手动登录客户端，并安装 easytrader。")
        elif broker_key == "manual":
            st.caption("程序导出委托单 CSV，你在券商 App 手动下单，再回填成交。最稳妥。")
        else:
            st.caption("模拟盘：用真实行情与费用模型演练，可随时重置。")
    with c2:
        live_capital = st.number_input("账户资金（元）", 10_000.0, 1.0e9, float(config.initial_cash),
                                       10_000.0, key="live_capital")
        dry_run = st.checkbox("dry-run（不真正下单）", value=(broker_key != "sim"), key="live_dry")
        auto_trade = st.checkbox("自动执行（关闭人工确认）", False, key="live_auto")
        use_latest = st.checkbox("用最新交易日生成计划", True, key="live_latest")
        if not use_latest:
            live_date = st.date_input("选股基准日", bundle.calendar[-1].date(), key="live_date")
        else:
            live_date = bundle.calendar[-1].date()
        st.caption(f"当前时间：{dt.datetime.now():%Y-%m-%d %H:%M:%S}　"
                   f"（{session_name()}）")
    with c3:
        st.markdown("**前置风控（下单前强制校验）**")
        r1, r2 = st.columns(2)
        max_order = r1.number_input("单笔最大金额（元）", 1000.0, 1.0e8, 100_000.0, 10_000.0,
                                    key="risk_max_order")
        max_daily = r2.number_input("单日累计金额（元）", 1000.0, 1.0e9, 500_000.0, 50_000.0,
                                    key="risk_max_daily")
        r1, r2 = st.columns(2)
        max_orders = r1.number_input("单日最大笔数", 1, 500, 40, 1, key="risk_max_orders")
        max_w = r2.slider("单票权重上限", 0.05, 1.0, 0.25, 0.05, key="risk_max_w")
        r1, r2 = st.columns(2)
        max_ind = r1.slider("行业权重上限", 0.1, 1.0, 0.40, 0.05, key="risk_max_ind")
        price_dev = r2.slider("限价偏离保护", 0.005, 0.10, 0.03, 0.005, key="risk_price_dev")
        r1, r2 = st.columns(2)
        day_loss = r1.slider("当日亏损熔断", 0.01, 0.20, 0.05, 0.01, key="risk_day_loss")
        allow_window = r2.checkbox("限制交易时段", True, key="risk_window")

    limits = RiskLimits(max_order_amount=float(max_order), max_daily_order_amount=float(max_daily),
                        max_daily_orders=int(max_orders), max_position_weight=float(max_w),
                        max_industry_weight=float(max_ind), max_price_deviation=float(price_dev),
                        daily_loss_limit=float(day_loss), enforce_trade_window=bool(allow_window),
                        lot_size=int(config.rules.lot_size))

    # ---- 构建/复用 trader ----
    sig_key = f"{broker_key}|{live_capital}|{dry_run}|{auto_trade}|{broker_kwargs}"
    if st.session_state.get("live_trader_key") != sig_key:
        try:
            if broker_key == "sim":
                from ashare_quant.execution import SimBroker
                broker = SimBroker(initial_cash=float(live_capital))
            elif broker_key == "manual":
                from ashare_quant.execution import ManualBroker
                broker = ManualBroker(initial_cash=float(live_capital))
            else:
                broker = create_broker(broker_key, **broker_kwargs)
            broker.connect()
            sel_cfg = st.session_state.get("pick_cfg") or SelectionConfig(
                top_k=5, min_amount_yuan=5e7, min_bars=120)
            live_cfg = LiveConfig(capital=float(live_capital), dry_run=bool(dry_run),
                                  auto_trade=bool(auto_trade), allow_gap_up=False,
                                  enforce_buy_date=True)
            trader = LiveTrader(broker, sel_cfg, limits, live_cfg,
                                industry_map={s: bundle.industry_of(s) for s in bundle.symbols})
            st.session_state.live_trader = trader
            st.session_state.live_trader_key = sig_key
            st.session_state.trading_plan = None
        except Exception as exc:
            st.error(f"通道初始化失败：{exc}")
    trader = st.session_state.get("live_trader")
    if trader is not None:
        trader.limits = limits
        trader.gate.limits = limits
        trader.cfg.capital = float(live_capital)
        trader.cfg.dry_run = bool(dry_run)
        trader.cfg.auto_trade = bool(auto_trade)
        c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
        run_plan = c1.button("① 生成交易计划", type="primary", key="live_build")
        run_exec = c2.button("② 执行计划", key="live_exec")
        run_rec = c3.button("③ 对账", key="live_rec")
        with c4:
            st.caption(f"通道：{trader.broker.display_name}　"
                       f"{'（模拟，不动真钱）' if not trader.broker.is_real_money else '⚠️ 真实资金通道'}"
                       f"　dry-run：{'开' if trader.cfg.dry_run else '关'}")

        if run_plan:
            try:
                plan = st.session_state.get("pick_for_live_plan")
                with st.spinner("生成交易计划（选股 → 目标持仓 → 风控校验）…"):
                    plan = trader.build_plan(bundle, as_of=pd.Timestamp(live_date),
                                             capital=float(live_capital),
                                             ignore_trade_window=bool(dry_run))
                st.session_state.trading_plan = plan
            except Exception as exc:
                st.error(f"生成计划失败：{exc}")

        plan = st.session_state.get("trading_plan")
        if plan is not None:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("选股日", f"{plan.signal_date:%Y-%m-%d}")
            c2.metric("👉 买入日期", f"{plan.buy_date:%Y-%m-%d}")
            c3.metric("待执行委托", f"{len(plan.approved)} 笔")
            c4.metric("被风控拦截", f"{len(plan.rejected)} 笔")
            st.markdown("#### 委托清单（含买入日期）")
            pf = plan.to_frame()
            if not pf.empty:
                st.dataframe(pf.style.format({"委托价格": "{:.2f}", "最新价": "{:.2f}",
                                              "涨停价": "{:.2f}", "跌停价": "{:.2f}",
                                              "预估金额": "{:,.0f}", "委托数量": "{:,.0f}"},
                                             na_rep="--"),
                             width="stretch", height=280)
            else:
                st.info("本次没有可执行委托。")
            if len(plan.rejected):
                st.markdown("**被拦截的委托（风控原因）**")
                st.dataframe(plan.rejected, width="stretch")
            st.code(plan.summary_text(), language="text")
            if st.button("💾 保存计划到本地", key="live_save"):
                p = trader.save_plan(plan)
                st.success(f"已保存：{p}")

        if run_exec and plan is not None:
            confirm_ok = True
            if trader.broker.is_real_money and not trader.cfg.dry_run:
                confirm_ok = st.checkbox("我已知晓这是真实资金委托，确认执行", key="live_confirm_real")
            if confirm_ok:
                try:
                    with st.spinner("执行委托…"):
                        report = trader.execute(plan, dry_run=trader.cfg.dry_run,
                                                confirm=not trader.cfg.auto_trade)
                    st.session_state.live_report = report
                except Exception as exc:
                    st.error(f"执行失败：{exc}")
            else:
                st.warning("请勾选确认后再次点击执行。")

        rep = st.session_state.get("live_report")
        if rep is not None:
            st.markdown("#### 执行结果")
            st.code(rep.summary_text(), language="text")
            rf = rep.to_frame()
            if not rf.empty:
                st.dataframe(rf, width="stretch", height=280)

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 账户")
            try:
                acct = trader.broker.account()
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("总资产", f"{acct.total:,.0f}")
                k2.metric("可用资金", f"{acct.available:,.0f}")
                k3.metric("持仓市值", f"{acct.market_value:,.0f}")
                k4.metric("仓位", f"{acct.position_weight:.1%}")
                posf = acct.to_frame()
                st.dataframe(posf.style.format({"成本价": "{:.2f}", "现价": "{:.2f}", "市值": "{:,.0f}",
                                                "盈亏": "{:,.0f}", "盈亏率": "{:+.2%}"}, na_rep="--")
                             if not posf.empty else posf, width="stretch", height=220)
            except Exception as exc:
                st.error(f"账户查询失败：{exc}")
        with c2:
            st.markdown("#### 持仓风控记录（买入日期/止损止盈）")
            mf = trader.meta_frame()
            st.dataframe(mf, width="stretch", height=220)
            if broker_key == "sim":
                if st.button("♻️ 重置模拟盘账户", key="live_reset_sim"):
                    trader.broker.reset(initial_cash=float(live_capital))
                    st.session_state.trading_plan = None
                    st.success("模拟盘已重置")

        if run_rec:
            try:
                rec = trader.reconcile(st.session_state.get("trading_plan"))
                st.markdown("#### 对账结果")
                st.dataframe(rec, width="stretch")
            except Exception as exc:
                st.error(f"对账失败：{exc}")

        with st.expander("今日委托 / 今日成交 / 运行日志"):
            try:
                st.markdown("**今日委托**")
                st.dataframe(trader.broker.orders_today(), width="stretch", height=200)
                st.markdown("**今日成交**")
                st.dataframe(trader.broker.fills_today(), width="stretch", height=200)
            except Exception as exc:
                st.caption(f"查询失败：{exc}")
            st.markdown("**日志**")
            st.code(trader.read_log(tail=60) or "（暂无日志）", language="text")

# =========================================================================== #
# Tab 10 · IC 分析（哪些因子真的有用）
# =========================================================================== #
if PAGE == PAGES[9]:
    st.markdown("### 📊 因子 IC 分析 —— 到底哪些因子有用？")
    st.markdown("<div class='okbox'><b>IC（信息系数）</b> = 因子值 与 未来收益 的秩相关："
                "IC&gt;0 表示「因子越大、未来涨得越多」；IC 越稳定（ICIR 高、t 值大）才越可信。<br>"
                "本页对 20 个因子做一次完整筛选，分成 <b>✅保留 / 🔄反向使用 / 👀观察 / ❌剔除</b>，"
                "并可直接用筛选结果去选股。</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    ic_n = c1.selectbox("研究样本（按成交额取前 N 只）", [300, 600, 1200, 5000], index=0,
                        format_func=lambda n: {300: "300 只（快）", 600: "600 只",
                                               1200: "1200 只", 5000: "全部沪深主板（慢，1~3 分钟）"}.get(n, f"{n} 只"),
                        key="fs_n")
    ic_h = c2.selectbox("持有期（交易日）", [5, 10, 20, 60], index=2, key="fs_h")
    ic_q = c3.selectbox("分层数（按因子值分几组）", [3, 5, 10], index=1, key="fs_q",
                        help="把股票按因子值从小到大分成 N 组，再看每组的未来收益。\n"
                             "若 Q1→QN 收益单调递增 = 因子正向有效；单调递减 = 反向有效；乱序 = 无效。\n"
                             "建议：股票少用 3 组、常用 5 组、3000 只以上可用 10 组。")
    ic_corr = c4.checkbox("计算因子相关性（较慢，可关闭）", False, key="fs_corr")
    if st.button("🔍 开始因子筛选（20 个因子）", type="primary", key="run_screen"):
        try:
            prog = st.progress(0.0, text="正在计算因子与 IC（约 30~60 秒）…")
            _p = lambda p: prog.progress(min(float(p), 0.99))   # noqa: E731
            with st.spinner("跑因子筛选…"):
                if int(ic_n) > 800:
                    # 大样本：逐因子计算（内存友好），最多跑全主板
                    fs = screen_factors_large(bundle, horizon=int(ic_h), max_symbols=int(ic_n),
                                              q=int(ic_q), progress_cb=_p, corr_top=10)
                else:
                    fs = screen_factors(bundle, horizon=int(ic_h), max_symbols=int(ic_n),
                                        q=int(ic_q), corr=bool(ic_corr), progress_cb=_p)
            prog.progress(1.0, text="完成")
            st.session_state.factor_screen = fs
        except Exception as exc:
            st.error(f"因子筛选失败：{exc}")

    fs = st.session_state.get("factor_screen")
    if fs is None:
        st.info("点上面的按钮开始筛选。样本 600 只约需 30 秒；样本越大结论越可信但越慢。")
    else:
        tab = fs.table
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("研究样本", f"{fs.universe} 只")
        k2.metric("持有期", f"{fs.horizon} 个交易日")
        k3.metric("建议保留", f"{len(fs.keep)} 个")
        if len(tab):
            _best = tab.iloc[0]
            k4.metric("最强因子", f"{_best.get('名称', '—')}", f"IC {_best.get('IC均值', 0):+.3f}")

        # ---------------- 🏆 最重要的 5 个因子 ----------------
        if len(getattr(fs, "top5", []) or []):
            st.markdown("#### 🏆 最重要的 5 个因子")
            st.caption("按综合评分排序：**IC 强度 35% ＋ ICIR 稳定性 25% ＋ t 值显著性 20% "
                       "＋ 分层单调性 10% ＋ 近期未失效 10%**（门槛：|t|≥1.5 且 |IC|≥0.015）")
            _t5 = fs.top5_table
            _c5 = st.columns(5)
            for _i, (_, _r) in enumerate(_t5.iterrows()):
                _dir5 = "正向" if float(_r["IC均值"]) > 0 else "反向"
                _c5[_i].metric(f"{_i + 1}. {_r['名称']}", f"评分 {_r['重要性评分']:.2f}",
                               f"IC {_r['IC均值']:+.3f}｜{_dir5}")
            _show5 = _t5[[c for c in ["名称", "重要性评分", "IC均值", "ICIR", "t值", "IC胜率",
                                      "单调性", "单调方向", "多空年化", "建议"]
                          if c in _t5.columns]]
            st.dataframe(_show5.style.format({"重要性评分": "{:.3f}", "IC均值": "{:+.4f}",
                                              "ICIR": "{:+.3f}", "t值": "{:+.2f}",
                                              "IC胜率": "{:.1%}", "单调性": "{:.2f}",
                                              "多空年化": "{:+.1%}"}, na_rep="--"),
                         width="stretch")
            with st.expander("📖 怎么用这 5 个因子？（点开）", expanded=False):
                _lines = []
                for _i, (_, _r) in enumerate(_t5.iterrows(), start=1):
                    _d = "数值越大越好（正向）" if float(_r["IC均值"]) > 0 else "数值越小越好（反向，权重取负）"
                    _lines.append(f"**{_i}. {_r['名称']}**：IC {_r['IC均值']:+.3f}、t {_r['t值']:+.2f}、"
                                  f"单调性 {_r['单调性']:.2f} → {_d}")
                st.markdown("\n\n".join(_lines))
                st.markdown("""
**怎么读这些数字？**
- **IC 均值**：因子与未来收益的秩相关，>0.03 就算不错（A 股单因子通常 0.02~0.06）
- **ICIR**：IC 均值 / IC 标准差，衡量稳定性，>0.2 算稳定
- **t 值**：|t| > 2 表示统计显著（不是运气）
- **单调性**：分层收益是否 Q1→Q5 递增，接近 1 最好
- **多空年化**：买最高分组、卖最低分组的理论年化收益

**注意**：这些结论来自当前样本期（现在是下跌市），**因子有效性会变**，建议每周重跑一次筛选。
""")
            _w5 = {str(_r["因子"]): (0.20 if float(_r["IC均值"]) > 0 else -0.20)
                   for _, _r in _t5.iterrows()}
            st.json({"这 5 个因子的权重": {FACTOR_DESC.get(k, k): v for k, v in _w5.items()}})
            _q1, _q2, _q3 = st.columns([1, 1, 3])
            if _q1.button("✅ 只用这 5 个因子选股（Top 10）", type="primary", key="pick_top5"):
                try:
                    _cfg5 = SelectionConfig(top_k=10, factor_weights=_w5, min_amount_yuan=5e7,
                                            min_bars=120, trend_ma=0, use_timing=False,
                                            max_weight=0.10, max_loss_pct=0.12,
                                            take_profit_pct=0.20)
                    with st.spinner("正在用 Top5 因子选股…"):
                        _res5 = select_stocks(bundle, _cfg5,
                                              capital=float(config.initial_cash), panel=fs.panel)
                    st.session_state.pick_result = _res5
                    st.session_state.pick_cfg = _cfg5
                    st.session_state.pick_src = "IC 分析 · Top5 因子"
                    st.session_state.pick_at = dt.datetime.now().strftime("%H:%M:%S")
                    st.success(f"完成：股票池 {_res5.universe} 只 → 入选 {len(_res5.picks)} 只")
                except Exception as exc:
                    st.error(f"选股失败：{exc}")
            _q2.caption("用这 5 个因子按方向加权（正向 +0.2 / 反向 -0.2）")

            # 选股结果**紧跟按钮显示**（之前放在页面最底部，手机上很难找）
            _pres5 = st.session_state.get("pick_result")
            if (_pres5 is not None and len(getattr(_pres5, "picks", []))
                    and str(st.session_state.get("pick_src", "")).startswith("IC 分析")):
                st.markdown("---")
                st.markdown(f"#### 📋 入选的 {len(_pres5.picks)} 只股票"
                            f"（买入日期 {_pres5.buy_date:%Y-%m-%d}）")
                _s5 = to_cn(_pres5.picks.copy())
                _k5 = [c for c in ["买入日期", "代码", "名称", "行业", "收盘价", "目标仓位",
                                   "股数", "买入价下限", "买入价上限", "止损价", "止盈价", "入选理由"]
                       if c in _s5.columns]
                st.dataframe(_s5[_k5].style.format(
                    {"目标仓位": "{:.1%}", "股数": "{:,.0f}", "收盘价": "{:.2f}",
                     "买入价下限": "{:.2f}", "买入价上限": "{:.2f}",
                     "止损价": "{:.2f}", "止盈价": "{:.2f}"}, na_rep="--"),
                    width="stretch", height=380)
                _buf5 = io.BytesIO()
                _s5.to_csv(_buf5, index=False, encoding="utf-8-sig")
                st.download_button("⬇️ 导出这 10 只（CSV）", _buf5.getvalue(),
                                   file_name=f"top5选股_{_pres5.buy_date:%Y%m%d}.csv",
                                   mime="text/csv", key="dl_top5_picks")
                st.caption("💡 也可以去「🎯 智能选股」页签查看同一份结果（含中文摘要与历史买入日期标注）")
                st.markdown("---")
        # ---------------- 📌 结论解读（自动生成，把数据结论翻译成人话） ----------------
        try:
            from ashare_quant.factor_research import interpret_screen
            _interp = interpret_screen(fs)
        except Exception:
            _interp = None
        if _interp and _interp.get("usable") is not None:
            st.markdown("---")
            st.markdown("### 📌 结论解读 · 这轮筛选到底说明了什么")
            st.info(_interp["summary"])
            st.markdown("#### 🧭 市场风格判断")
            for _s in _interp["style"]:
                st.markdown(f"- {_s}")
            if _interp["usable"]:
                st.markdown("#### ✅ 可用的因子（已通过 FDR 校正 + 四段稳定性）")
                _u = pd.DataFrame(_interp["usable"])
                st.dataframe(_u.style.format({"IC": "{:+.4f}", "t": "{:+.2f}", "单调性": "{:.2f}"},
                                             na_rep="--"),
                             width="stretch", height=min(320, 60 + 35 * len(_u)))
            if _interp["blocked"]:
                st.markdown("#### ⚠️ 被拦下的因子（IC 看着好，但不稳定/不显著）")
                _b = pd.DataFrame(_interp["blocked"])
                st.dataframe(_b.style.format({"IC": "{:+.4f}", "t": "{:+.2f}"}, na_rep="--"),
                             width="stretch", height=min(320, 60 + 35 * len(_b)))
            if _interp["weights"]:
                st.markdown("#### ⚖️ 建议权重（按 |IC| 加权；正向为正、反向为负）")
                _wdf = pd.DataFrame({"因子": list(_interp["weights"].keys()),
                                     "建议权重": list(_interp["weights"].values())})
                st.dataframe(_wdf.style.format({"建议权重": "{:+.3f}"}), width="stretch",
                             height=min(320, 60 + 35 * len(_wdf)))
                st.caption("把上面这张表的权重填到「🎯 智能选股」的因子权重里，即可用同一套逻辑选股。")
            st.markdown("#### 📌 使用提醒")
            for _n in _interp["notes"]:
                st.markdown(f"- {_n}")
            st.markdown("---")
        st.dataframe(tab.style.format({"IC均值": "{:+.4f}", "IC标准差": "{:.4f}", "ICIR": "{:+.3f}",
                                       "t值": "{:+.2f}", "IC胜率": "{:.1%}", "单调性": "{:.2f}",
                                       "多空年化": "{:+.1%}", "近期IC": "{:+.4f}", "样本数": "{:.0f}"},
                                      na_rep="--"),
                     width="stretch", height=380)
        c1, c2 = st.columns([2, 3])
        with c1:
            st.plotly_chart(ch.ic_bar(tab, height=430, title="IC 均值排名（按绝对值排序）"),
                            width="stretch", config=ch.PLOTLY_CONFIG)
        with c2:
            st.plotly_chart(ch.ic_decay_heatmap(fs.decay, height=430), width="stretch",
                            config=ch.PLOTLY_CONFIG)

        st.markdown("#### 单因子深入分析")
        fsel = st.selectbox("选择因子", tab["因子"].tolist(),
                            format_func=lambda k: FACTOR_DESC.get(k, k), key="ic_factor")
        if fsel in fs.ic_series.columns:
            a1, a2 = st.columns([3, 2])
            with a1:
                st.plotly_chart(ch.ic_series_chart(fs.ic_series[fsel],
                                                   name=FACTOR_DESC.get(fsel, fsel), height=360),
                                width="stretch", config=ch.PLOTLY_CONFIG)
            with a2:
                if fsel in fs.quantiles:
                    _qn = int(ic_q)
                    st.plotly_chart(ch.quantile_curve_chart(fs.quantiles[fsel], height=360),
                                    width="stretch", config=ch.PLOTLY_CONFIG)
                    st.caption(f"↑ 把股票按【{FACTOR_DESC.get(fsel, fsel)}】的值从小到大分成 {_qn} 组："
                               f"Q1 = 因子值最低那 {100 // _qn}%（最不看好），"
                               f"Q{_qn} = 因子值最高那 {100 // _qn}%（最看好）。"
                               f"曲线**依次向上排列**说明因子有效；"
                               f"黑色虚线「多空」= 买 Q{_qn}、卖 Q1 的累计收益。")
        if ic_corr and not fs.corr.empty:
            corr_cn = fs.corr.copy()
            corr_cn.index = [FACTOR_DESC.get(str(i), str(i)) for i in corr_cn.index]
            corr_cn.columns = [FACTOR_DESC.get(str(c), str(c)) for c in corr_cn.columns]
            st.markdown("#### 因子相关性（相关性高的因子是重复信息，可只留一个）")
            st.plotly_chart(ch.correlation_heatmap(corr_cn.round(2), height=460,
                                                   title="因子相关性矩阵"),
                            width="stretch", config=ch.PLOTLY_CONFIG)

        st.markdown("#### 用筛选结果直接选股")
        st.caption("「✅保留」的因子给正权重、「🔄反向使用」的因子给负权重，合成自适应选股模型。")
        _w = {}
        for _, r in tab.iterrows():
            adv = str(r["建议"])
            if "保留" in adv:
                _w[r["因子"]] = 0.25
            elif "反向" in adv:
                _w[r["因子"]] = -0.25
        if _w:
            st.json({"因子权重": {FACTOR_DESC.get(k, k): v for k, v in _w.items()}})
        else:
            st.warning("本次没有筛出可用因子（可换持有期或扩大样本再试）。")
        cc1, cc2, _cc3 = st.columns([1, 1, 3])
        if cc1.button("✅ 用有效因子选股（10 只）", type="primary", key="pick_with_factors") and _w:
            try:
                _cfg = SelectionConfig(top_k=10, factor_weights=_w, min_amount_yuan=1e8,
                                       min_bars=120, trend_ma=0, use_timing=False,
                                       max_weight=0.10, max_loss_pct=0.12, take_profit_pct=0.20)
                with st.spinner("正在用筛选后的因子选股…"):
                    _res = select_stocks(bundle, _cfg, capital=float(config.initial_cash),
                                         panel=fs.panel)
                st.session_state.pick_result = _res
                st.session_state.pick_cfg = _cfg
                st.success(f"选股完成：股票池 {_res.universe} 只 → 入选 {len(_res.picks)} 只")
            except Exception as exc:
                st.error(f"选股失败：{exc}")
        buf = io.BytesIO()
        tab.to_csv(buf, index=False, encoding="utf-8-sig")
        cc2.download_button("⬇️ 导出筛选结果 CSV", buf.getvalue(),
                            file_name=f"factor_screen_{int(ic_h)}d.csv", mime="text/csv",
                            key="dl_screen")
        _pres2 = st.session_state.get("pick_result")
        if _pres2 is not None and len(getattr(_pres2, "picks", [])):
            st.markdown("##### 用有效因子选出的股票")
            _s = to_cn(_pres2.picks.copy())
            _keep = [c for c in ["买入日期", "代码", "名称", "行业", "收盘价", "综合分", "目标仓位",
                                 "股数", "买入价下限", "买入价上限", "止损价", "止盈价", "入选理由"]
                     if c in _s.columns]
            st.dataframe(_s[_keep].style.format({"综合分": "{:.3f}", "目标仓位": "{:.1%}",
                                                 "股数": "{:,.0f}", "收盘价": "{:.2f}",
                                                 "买入价下限": "{:.2f}", "买入价上限": "{:.2f}",
                                                 "止损价": "{:.2f}", "止盈价": "{:.2f}"}, na_rep="--"),
                         width="stretch", height=320)


# =========================================================================== #
# Tab 11 · 规则说明
# =========================================================================== #
if PAGE == PAGES[10]:
    st.markdown("### A 股交易规则与成本假设")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
#### 交易制度
| 规则 | 说明 |
| --- | --- |
| 交易方向 | 只能做多，不提供裸卖空 / 融资融券 |
| T+1 | 当日买入的股票，次一交易日才可卖出 |
| 最小单位 | 买入 100 股整数倍；科创板单笔 ≥200 股，超过部分 1 股递增 |
| 涨跌幅 | 主板 ±10%，创业板/科创板 ±20%，北交所 ±30%，ST ±5% |
| 新股上市 | 科创板/创业板前 5 日、主板首日不设涨跌幅 |
| 停牌 | 停牌期间不可交易，回测按停牌前价格延续估值 |
| 交易时段 | 9:30-11:30 / 13:00-15:00，集合竞价 9:15-9:25、14:57-15:00 |
| 结算 | 卖出资金当日可用（可再买入），次日出金 |

#### 交易成本（默认值，可在侧边栏修改）
| 项目 | 费率 | 方向 |
| --- | --- | --- |
| 佣金 | 万 2.5，单笔最低 5 元 | 买卖双边 |
| 印花税 | 0.05% | 仅卖出 |
| 过户费 | 0.001% | 买卖双边 |
| 滑点 | 5 bp（0.05%） | 买卖双边 |
""")
    with c2:
        st.markdown("""
#### 回测执行时序（避免未来函数）
1. 第 T 日 **收盘后** 计算目标权重；
2. 第 T+1 日 **开盘** 以开盘价（含滑点）成交；
3. 受涨跌停、停牌、T+1、整手、资金约束；
4. 收盘按收盘价结算，触发止损 / 熔断则次日开盘执行。

#### 内置策略清单
""")
        st.dataframe(list_strategies()[["name", "category", "description"]]
                     .rename(columns={"name": "策略", "category": "类别", "description": "说明"}),
                     width="stretch", height=380)
    st.markdown("""
#### 常用绩效指标口径
* **年化收益**：`(期末净值/期初净值)^(252/交易日数) - 1`
* **最大回撤**：净值相对历史最高点的最大跌幅，衡量最坏持有体验
* **夏普比率**：年化超额收益 / 年化波动，>1 可接受，>2 较优
* **卡玛比率**：年化收益 / 最大回撤，衡量单位回撤收益
* **信息比率**：超额收益 / 跟踪误差，衡量主动管理能力
* **换手率**：年化双边换手，越低越省成本

#### 实盘落地的现实约束（务必了解）
1. **冲击成本**：小市值股票实际冲击成本可能远高于 5bp；
2. **涨跌停无法成交**：强势股的买入信号常常"一字板买不进"，回测已按此约束处理；
3. **停牌与退市**：长期停牌、退市的极端损失难以完全模拟；
4. **数据质量**：复权方式、停牌处理、财务数据滞后都会影响结论；
5. **过拟合**：参数寻优页签的样本外验证与 Walk-Forward 是必要但不充分的防线；
6. **容量限制**：资金规模越大，策略容量与滑点越差。

<div class='warnbox'>
<strong>风险提示：</strong>本程序仅用于量化研究与教学演示，所有数据（尤其是离线合成数据）
均不代表真实市场，任何策略回测结果都不构成投资建议。历史表现不代表未来收益。
A 股程序化交易存在合规要求，实盘请使用券商官方渠道并自行承担全部风险。
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<div class='small'>A 股量化研究平台 v1.0 · 内置 "
            f"{len(REGISTRY)} 个策略 · 数据源：{bundle.benchmark_name} · "
            "仅供研究使用</div>", unsafe_allow_html=True)


































