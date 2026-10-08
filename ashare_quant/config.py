"""全局配置：路径、默认成本、市场常量。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List

ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = ROOT / "data_cache"
EXPORT_DIR: Path = ROOT / "exports"
for _d in (DATA_DIR, EXPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# A 股一年约 242~245 个交易日，业界习惯用 252 做年化
TRADE_DAYS_PER_YEAR: int = 252
RISK_FREE_RATE: float = 0.02

# 各板块涨跌幅限制（普通股票）
BOARD_LIMIT: Dict[str, float] = {
    "MAIN": 0.10,   # 沪深主板 10%
    "GEM": 0.20,    # 创业板 20%
    "STAR": 0.20,   # 科创板 20%
    "BSE": 0.30,    # 北交所 30%
}
ST_LIMIT: float = 0.05           # ST / *ST 股票 5%
NEW_LISTING_UNCAPPED_DAYS: Dict[str, int] = {"STAR": 5, "GEM": 5, "BSE": 1, "MAIN": 1}

# 默认演示股票池（各行业代表性标的，仅用于离线示例，不构成投资建议）
DEFAULT_POOL: List[Dict[str, str]] = [
    {"symbol": "600519", "name": "贵州茅台", "industry": "食品饮料"},
    {"symbol": "000858", "name": "五粮液", "industry": "食品饮料"},
    {"symbol": "601318", "name": "中国平安", "industry": "非银金融"},
    {"symbol": "600036", "name": "招商银行", "industry": "银行"},
    {"symbol": "000001", "name": "平安银行", "industry": "银行"},
    {"symbol": "300750", "name": "宁德时代", "industry": "电力设备"},
    {"symbol": "002594", "name": "比亚迪", "industry": "汽车"},
    {"symbol": "600276", "name": "恒瑞医药", "industry": "医药生物"},
    {"symbol": "300760", "name": "迈瑞医疗", "industry": "医药生物"},
    {"symbol": "002415", "name": "海康威视", "industry": "电子"},
    {"symbol": "600030", "name": "中信证券", "industry": "非银金融"},
    {"symbol": "601899", "name": "紫金矿业", "industry": "有色金属"},
    {"symbol": "600900", "name": "长江电力", "industry": "公用事业"},
    {"symbol": "601012", "name": "隆基绿能", "industry": "电力设备"},
    {"symbol": "000333", "name": "美的集团", "industry": "家用电器"},
]

BENCHMARKS: Dict[str, str] = {
    "000300": "沪深300",
    "000905": "中证500",
    "000852": "中证1000",
    "399006": "创业板指",
    "000001": "上证指数",
}


@dataclass
class BacktestDefaults:
    """回测默认参数（可在界面中覆盖）。"""

    initial_cash: float = 1_000_000.0     # 初始资金
    commission_rate: float = 0.00025      # 佣金：万 2.5（双边）
    commission_min: float = 5.0           # 单笔最低佣金 5 元
    stamp_tax_rate: float = 0.0005        # 印花税：卖出 0.05%
    transfer_fee_rate: float = 0.00001    # 过户费：成交额 0.001%（双边）
    slippage_bps: float = 5.0             # 滑点：5bp = 0.05%
    lot_size: int = 100                   # 1 手 = 100 股
    t_plus_1: bool = True                 # T+1 制度
    enforce_price_limit: bool = True      # 涨跌停不可成交
    enforce_suspension: bool = True       # 停牌不可交易
    max_position_weight: float = 0.25     # 单票权重上限
    max_industry_weight: float = 0.40     # 单行业权重上限
    stop_loss_atr: float = 2.5            # ATR 止损倍数（None/0 表示关闭）
    trailing_stop: float = 0.15           # 移动止损（相对持仓最高价回撤）
    max_drawdown_stop: float = 0.25       # 组合回撤熔断线（None 关闭）

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @property
    def total_cost_rate(self) -> float:
        """买入方向的显性成本率（用于快速估算）。"""
        return self.commission_rate + self.transfer_fee_rate + self.slippage_bps / 10000.0