"""策略参数的中文标注与说明（供界面显示，避免出现 fast/slow/top_k 这类英文）。"""

from __future__ import annotations

from typing import Dict, Optional

__all__ = ["PARAM_LABELS", "PARAM_HELP", "param_label", "param_help", "describe_params"]

PARAM_LABELS: Dict[str, str] = {
    "fast": "快线周期(天)", "slow": "慢线周期(天)", "trend_ma": "趋势均线(天)",
    "signal": "MACD信号线(天)", "entry": "突破周期(天)", "exit": "离场周期(天)",
    "top_k": "持仓只数", "rebalance": "调仓频率", "max_weight": "单票上限",
    "weight_mode": "加权方式", "total_exposure": "总仓位上限",
    "vol_filter": "波动率过滤(年化)", "momentum_confirm": "动量确认(天)",
    "min_amount": "最小成交额(元)", "use_timing": "大盘择时", "bench_ma": "择时均线(天)",
    "rsi_n": "RSI周期(天)", "oversold": "超卖阈值", "exit_level": "离场阈值",
    "n": "布林周期(天)", "k": "布林倍数", "entry_pctb": "入场位置(%B)",
    "exit_pctb": "离场位置(%B)", "max_width": "带宽上限",
    "ref_window": "基准均线(天)", "grid_step": "网格间距", "max_grids": "最大网格数",
    "per_grid": "每格仓位", "lookback": "动量回看(天)", "skip": "跳过最近(天)",
    "n_industry": "行业数量", "per_industry": "行业内只数", "mom_window": "动量窗口(天)",
    "rev_window": "反转窗口(天)", "vol_window": "波动窗口(天)",
    "w_momentum": "动量权重", "w_reversal": "反转权重", "w_lowvol": "低波动权重",
    "w_trend": "趋势权重", "w_liquidity": "流动性权重",
    "mode": "模式", "per_symbol": "单只仓位", "members": "子策略",
    "hold_days": "最长持有(天)", "stop": "止损幅度", "take": "止盈幅度",
}

PARAM_HELP: Dict[str, str] = {
    "fast": "短期均线天数。越小越灵敏、交易越频繁。例：fast=5 就是 5 日均线。",
    "slow": "长期均线天数。快线上穿慢线视为买入信号（金叉）。必须大于快线。",
    "trend_ma": "长期趋势过滤线：只有股价站在该均线之上才允许买入。填 0 = 关闭过滤。",
    "signal": "MACD 的信号线天数，常用 9。数值越大信号越平滑、越滞后。",
    "entry": "价格突破过去多少天的最高价才买入（海龟法则）。",
    "exit": "价格跌破过去多少天的最低价就卖出。",
    "top_k": "每期最多持有几只股票。越多越分散、单只收益贡献越小。",
    "rebalance": "多久调一次仓（重新选股换股）。越频繁交易成本越高。",
    "max_weight": "单只股票最多占账户的比例（0.25 = 25%），控制单票风险。",
    "weight_mode": "equal = 每只等额买入；score = 分数越高买得越多。",
    "vol_filter": "年化波动率高于该值的股票不买（例 0.6 = 60%）。填 0 关闭。",
    "momentum_confirm": "要求最近 N 天涨幅为正才买。填 0 关闭。",
    "min_amount": "近 20 日平均成交额低于该值的股票不买（保证流动性）。",
    "use_timing": "开=大盘在均线下方时空仓；关=不管大盘一直选股。",
    "bench_ma": "判断大盘强弱的均线天数（沪深300 相对该均线）。",
    "rsi_n": "RSI 计算周期，常用 14 天。",
    "oversold": "RSI 低于该值视为超卖买入，常用 30。",
    "exit_level": "RSI 回到该值以上就卖出离场，常用 55。",
    "n": "布林带/统计窗口天数，常用 20。",
    "k": "布林带宽度倍数，常用 2（上下轨 = 均线 ± 2 倍标准差）。",
    "entry_pctb": "价格在布林带中的位置低于该值才买入（0.05 = 贴近下轨）。",
    "exit_pctb": "价格回到该位置以上卖出（0.55 = 回到中轨上方）。",
    "max_width": "布林带过宽（极端波动）时不买入。填 0 关闭。",
    "ref_window": "网格的基准均线天数，价格相对它上下波动时加减仓。",
    "grid_step": "每下跌多少比例加一格（0.03 = 3%）。",
    "max_grids": "最多加几格（控制最大买入次数）。",
    "per_grid": "每格买入的仓位比例。",
    "lookback": "动量排名回看天数，看过去多少天的涨幅。",
    "skip": "排名时跳过最近 N 天，规避短期反转（常用 5 天）。",
    "n_industry": "选几个最强的行业。",
    "per_industry": "每个行业内买几只股票。",
    "mom_window": "动量因子回看天数。",
    "rev_window": "反转因子回看天数（短期跌得多的优先）。",
    "vol_window": "波动率计算窗口。",
    "w_momentum": "多因子模型里「动量」的权重，可正可负。",
    "w_reversal": "多因子模型里「反转」的权重。",
    "w_lowvol": "多因子模型里「低波动」的权重。",
    "w_trend": "多因子模型里「趋势强度」的权重。",
    "w_liquidity": "多因子模型里「流动性/小市值」的权重。",
    "mode": "switch = 择时关闭时完全空仓；scale = 按强弱调整仓位。",
    "per_symbol": "每只标的的目标仓位（择时策略用）。",
    "members": "要集成的子策略列表（逗号分隔）。",
    "hold_days": "最长持有交易日，到期强制评估卖出。",
    "stop": "跌破买入价的该比例就止损卖出（0.12 = 亏 12% 止损）。",
    "take": "涨到买入价的该比例就止盈卖出（0.20 = 赚 20% 止盈）。",
}


def param_label(name: str) -> str:
    return PARAM_LABELS.get(str(name), str(name))


def param_help(name: str) -> Optional[str]:
    return PARAM_HELP.get(str(name))


def describe_params(cls) -> list:
    """返回某策略的参数说明表（中文名 / 参数名 / 默认值 / 说明）。"""
    rows = []
    for k, v in getattr(cls, "default_params", {}).items():
        rows.append({"中文名": param_label(k), "参数名": k, "默认值": v,
                     "说明": param_help(k) or ""})
    return rows
