"""ashare_quant —— 面向 A 股市场的量化研究 / 回测 / 选股 / **实盘交易** 框架。

模块总览
--------
- ``rules``             A 股交易规则（涨跌停、T+1、整手、印花税、过户费…）
- ``trading_calendar``  交易日历（官方日历优先，离线兜底）→ 买入日期推算
- ``data``              数据层（AkShare / TuShare / 本地 CSV / 离线合成数据）
- ``indicators``        技术指标库
- ``strategies``        策略库（趋势、动量、均值回归、多因子、网格…）
- ``backtest``          事件驱动回测引擎（含成本、滑点、涨跌停、停牌约束）
- ``metrics``           绩效与风险指标
- ``risk``              组合风控（止损、仓位、行业上限、回撤熔断）
- ``optimize``          参数寻优与滚动前推验证
- ``factors``           多因子打分与 IC 分析
- ``selector``          **数据选股器（输出含"买入日期"的选股清单）**
- ``signals``           策略信号与模拟记账
- ``quotes``            实时行情快照（券商 / AkShare / 本地行情三级降级）
- ``execution``         **实盘执行层**（模拟盘 / 手动执行 / QMT / easytrader + 前置风控）
- ``live``              **实盘交易引擎**（选股 → 目标持仓 → 风控 → 下单 → 对账）
"""

from .config import ROOT, DATA_DIR, EXPORT_DIR, BacktestDefaults
from .rules import CostModel, TradingRules, get_board, limit_prices, round_lot
from .data import DataBundle, synthetic_bundle
from .backtest import Backtester, BacktestConfig, BacktestResult
from .metrics import performance_summary
from .trading_calendar import next_trade_date, prev_trade_date, get_calendar
from .selector import SelectionConfig, SelectionResult, select_stocks, select_history, selection_stats

__version__ = "1.1.0"
__all__ = [
    "ROOT", "DATA_DIR", "EXPORT_DIR", "BacktestDefaults",
    "CostModel", "TradingRules", "get_board", "limit_prices", "round_lot",
    "DataBundle", "synthetic_bundle",
    "Backtester", "BacktestConfig", "BacktestResult",
    "performance_summary",
    "next_trade_date", "prev_trade_date", "get_calendar",
    "SelectionConfig", "SelectionResult", "select_stocks", "select_history", "selection_stats",
]
