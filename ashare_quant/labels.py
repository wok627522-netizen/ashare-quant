"""中文标注层：把内部英文列名/取值统一翻译成中文，供界面、CLI、导出使用。

设计原则：**内部计算保持英文列名**（代码稳定、便于维护），
只在"给人看"的地方（界面表格、命令行输出、Excel/CSV 导出）通过
``to_cn()`` / ``cn_columns()`` 转成中文。

用法::

    from ashare_quant.labels import to_cn
    st.dataframe(to_cn(result.trades))

    to_cn(df, extra={"my_col": "我的列"})      # 追加自定义映射
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional

import pandas as pd

__all__ = ["COLUMN_LABELS", "VALUE_LABELS", "to_cn", "cn_columns", "cn_value", "cn_series",
           "label_of", "to_cn_excel"]

# --------------------------------------------------------------------------- #
# 列名映射
# --------------------------------------------------------------------------- #
COLUMN_LABELS: Dict[str, str] = {
    # ---- 时间 ----
    "date": "日期", "datetime": "时间", "ts": "时间", "trade_date": "交易日",
    "signal_date": "选股日期", "buy_date": "买入日期", "sell_date": "卖出日期",
    "entry_date": "买入日期", "exit_date": "卖出日期", "start": "开始日期", "end": "结束日期",
    "buy_timing": "买入时机", "created_at": "创建时间", "updated": "更新时间",
    "peak_date": "高点日期", "trough_date": "低点日期", "recover_date": "修复日期",
    "holding_days": "持有天数", "max_hold_days": "最长持有天数", "duration": "持续天数",
    # ---- 标的 ----
    "symbol": "代码", "code": "代码", "name": "名称", "industry": "行业", "board": "板块",
    "market": "市场", "is_st": "是否ST", "rank": "排名", "universe": "股票池",
    # ---- 行情 ----
    "open": "开盘价", "high": "最高价", "low": "最低价", "close": "收盘价",
    "last": "最新价", "ref_price": "参考价", "price": "价格", "prev_close": "昨收价",
    "limit_up": "涨停价", "limit_down": "跌停价", "next_limit_up": "次日涨停价",
    "limit_pct": "涨跌幅限制", "volume": "成交量", "amount": "成交额",
    "amount20": "20日均成交额", "amount20_wan": "20日均成交额(万)",
    "chg": "涨跌幅", "pct_chg": "涨跌幅", "day_chg": "当日涨跌幅", "change": "涨跌幅",
    "amplitude": "振幅", "turnover": "换手率", "suspended": "停牌", "atr": "ATR波幅",
    # ---- 选股 / 策略 ----
    "score": "综合分", "reason": "入选理由", "warnings": "风险提示",
    "target_weight": "目标仓位", "cur_weight": "当前仓位", "delta_weight": "仓位变动",
    "weight": "权重", "shares": "股数", "advice": "建议",
    "buy_price_low": "买入价下限", "buy_price_high": "买入价上限",
    "stop_loss_price": "止损价", "take_profit_price": "止盈价",
    "action": "操作建议", "note": "备注", "status": "状态",
    # ---- 交易 / 账户 ----
    "side": "方向", "side_cn": "方向", "order_id": "委托编号", "client_id": "委托号",
    "filled": "已成交数量", "filled_shares": "成交数量", "avg_price": "成交均价",
    "commission": "佣金", "stamp_tax": "印花税", "transfer_fee": "过户费",
    "total_fee": "费用合计", "fee": "费用", "cash": "现金", "available": "可用资金",
    "market_value": "市值", "frozen": "冻结", "total": "总资产", "equity": "净值",
    "total_return": "累计收益", "position_weight": "仓位", "n_holdings": "持仓数量",
    "day_return": "当日收益率", "drawdown": "回撤", "benchmark": "基准",
    "benchmark_return": "基准收益", "ret": "收益率", "pnl": "盈亏", "pnl_pct": "盈亏率",
    "cost": "成本价", "entry_price": "买入价", "exit_price": "卖出价",
    "order_type": "委托类型", "amount_cny": "金额(元)",
    # ---- 绩效 ----
    "annual_return": "年化收益", "annual_vol": "年化波动", "sharpe": "夏普比率",
    "sortino": "索提诺比率", "calmar": "卡玛比率", "max_drawdown": "最大回撤",
    "win_rate": "胜率", "profit_factor": "盈亏比", "excess_return": "超额收益",
    "information_ratio": "信息比率", "var95": "日风险价值", "cvar95": "日条件风险价值",
    # ---- 其他 ----
    "params": "参数", "metric": "指标", "score_col": "得分列", "value": "数值",
    "event": "风控事件", "detail": "说明", "symbol_list": "标的清单",
    "bars": "交易日数", "last_close": "最新价", "annual_vol": "年化波动",
    "period_return": "区间收益", "ICIR": "IC信息比", "IC均值": "IC均值",
    "行业": "行业", "步骤": "步骤", "通过数量": "通过数量", "剩余": "剩余数量",
    "样本数": "样本数", "最好": "最好", "最差": "最差", "选股期数": "选股期数",
    "持有期": "持有期", "个股平均收益": "个股平均收益", "个股胜率": "个股胜率",
    "组合平均收益": "组合平均收益", "组合胜率": "组合胜率",
    "item": "项目", "note_text": "说明", "is_score": "样本内得分", "oos_score": "样本外得分",
    "overfit_gap": "过拟合差距", "stable": "是否稳健", "count": "数量",
}

# --------------------------------------------------------------------------- #
# 取值映射
# --------------------------------------------------------------------------- #
VALUE_LABELS: Dict[str, Dict[str, str]] = {
    "side": {"buy": "买入", "sell": "卖出", "BUY": "买入", "SELL": "卖出"},
    "side_cn": {"buy": "买入", "sell": "卖出"},
    "status": {"SUBMITTED": "已报", "FILLED": "全部成交", "PARTIAL": "部分成交",
               "CANCELLED": "已撤", "REJECTED": "废单", "PENDING": "待报"},
    "order_type": {"limit": "限价", "market": "市价", "best5": "最优五档",
                   "LIMIT": "限价", "MARKET": "市价"},
    "stable": {True: "是", False: "否"},
    "suspended": {True: "停牌", False: "正常"},
    "is_st": {True: "是", False: "否"},
    "方向": {}, "状态": {},
}


def label_of(col: str) -> str:
    """单列名 → 中文（未知列原样返回；factor_ 前缀会被翻译为「因子·」）。"""
    c = str(col)
    if c in COLUMN_LABELS:
        return COLUMN_LABELS[c]
    if c.startswith("factor_"):
        return "因子·" + c[len("factor_"):]
    if c.startswith("z_"):
        return "标准化·" + c[len("z_"):]
    if c.startswith("ret_") and c.endswith("d"):
        return f"买入后{c[len('ret_'):-1]}日收益"
    if c.startswith("oos_"):
        return "样本外·" + COLUMN_LABELS.get(c[4:], c[4:])
    if c.startswith("is_"):
        return "样本内·" + COLUMN_LABELS.get(c[3:], c[3:])
    return c


def cn_columns(df: pd.DataFrame, extra: Optional[Mapping[str, str]] = None) -> pd.DataFrame:
    """把 DataFrame 的列名翻译成中文（返回副本）。"""
    out = df.copy()
    mapping = {c: label_of(c) for c in out.columns}
    if extra:
        mapping.update({k: v for k, v in extra.items() if k in out.columns})
    out = out.rename(columns=mapping)
    return out


def cn_value(col: str, value):
    """把单元格取值翻译成中文（仅对已知列）。"""
    mapping = VALUE_LABELS.get(str(col))
    if not mapping:
        return value
    try:
        return mapping.get(value, mapping.get(str(value), value))
    except Exception:
        return value


def cn_series(s: pd.Series, name: Optional[str] = None) -> pd.Series:
    """Series → 中文名 + 中文取值。"""
    out = s.copy()
    if name or out.name:
        out.name = label_of(name or out.name)
    src = str(s.name or "")
    if src in VALUE_LABELS:
        out = out.map(lambda v: cn_value(src, v))
    return out


def to_cn(df: pd.DataFrame, extra: Optional[Mapping[str, str]] = None,
          translate_values: bool = True) -> pd.DataFrame:
    """一步到位：列名中文化 + 常见取值中文化。"""
    if df is None or len(df.columns) == 0:
        return df
    src_cols = list(df.columns)
    out = df.copy()
    if translate_values:
        for c in src_cols:
            if str(c) in VALUE_LABELS and len(VALUE_LABELS[str(c)]):
                out[c] = out[c].map(lambda v, cc=c: cn_value(str(cc), v))
    out = cn_columns(out, extra=extra)
    return out


def to_cn_excel(writer, result, sheet_prefix: str = "") -> None:
    """把回测结果按中文表头写入 Excel 的多个 sheet（供 BacktestResult.to_excel 使用）。"""
    pd.DataFrame({"净值": result.equity,
                  "基准": result.benchmark if result.benchmark is not None else pd.Series(dtype=float)}
                 ).to_excel(writer, sheet_name=f"{sheet_prefix}净值")
    to_cn(result.stats_table()).to_excel(writer, sheet_name=f"{sheet_prefix}绩效指标", index=False)
    to_cn(result.daily.reset_index()).to_excel(writer, sheet_name=f"{sheet_prefix}每日明细", index=False)
    to_cn(result.weights.reset_index()).to_excel(writer, sheet_name=f"{sheet_prefix}持仓权重", index=False)
    to_cn(result.trades).to_excel(writer, sheet_name=f"{sheet_prefix}成交明细", index=False)
    to_cn(result.round_trips).to_excel(writer, sheet_name=f"{sheet_prefix}回合交易", index=False)
    to_cn(result.monthly_table()).to_excel(writer, sheet_name=f"{sheet_prefix}月度收益")
    to_cn(result.yearly_returns()).to_excel(writer, sheet_name=f"{sheet_prefix}年度收益", index=False)
    if len(getattr(result, "events", [])):
        to_cn(result.events).to_excel(writer, sheet_name=f"{sheet_prefix}风控事件", index=False)


