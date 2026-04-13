# 防前视偏差回测引擎 (Realistic Backtest Engine)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Pandas](https://img.shields.io/badge/Pandas-1.5+-green.svg)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **严谨、专业的量化回测框架，防止前视偏差，确保回测结果可靠**

## 核心问题

**为什么大多数回测都是骗人的？**

| 问题 | 影响 | 本解决方案 |
|------|------|-----------|
| **前视偏差** | 使用未来信息做决策 | Point-in-Time数据访问 |
| **生存者偏差** | 只测试幸存股票 | 包含退市股票数据 |
| **过拟合** | 策略过度拟合历史 | Walk-forward验证 |
| **执行假设** |  unrealistic fill prices | 滑点/冲击成本模拟 |
| **数据修正** | 使用修正后的历史数据 | 原始as-of数据 |

## 核心特性

### 1. Point-in-Time Data (防止前视偏差)

```python
from backtest_engine.data.pit_data import PointInTimeData

# Only use data available at decision time
data = PointInTimeData(price_data, availability_delay="1d")
available_data = data.get_data_as_of(date)  # Only past data
```

**原理**：每个数据点都记录其可用时间，确保回测时不会使用未来的信息。

### 2. Walk-Forward Validation (防止过拟合)

```python
[Train Period] [Test Period] → Roll → [Train Period] [Test Period] → ...
```

真正的样本外测试，滚动窗口验证，黄金标准。

### 3. 执行仿真 (真实交易成本)

- **滑点模型**：基于成交量和市场冲击
- **延迟模拟**：订单执行延迟
- **市场冲击**：Almgren-Chriss模型
- **流动性约束**：大额订单分批执行

### 4. 回测-实盘归因分析

量化回测和实盘表现差异的原因：
- 执行成本差异
- 时序差异
- 数据质量差异
- 过拟合程度

## 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                    Realistic Backtest Engine                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Data Layer (Point-in-Time)                                    │
│  ├─ PIT Data: Only available data at time T                    │
│  ├─ Corporate Actions: As-of dates, not ex-dates              │
│  └─ Survivorship Bias Prevention: Include delisted            │
│                              ↓                                  │
│  Walk-Forward Validation                                        │
│  ├─ Train Period: Optimize strategy                           │
│  ├─ Purge: Gap to prevent leakage                             │
│  ├─ Test Period: Out-of-sample evaluation                     │
│  └─ Embargo: Gap after test                                   │
│                              ↓                                  │
│  Execution Simulation                                           │
│  ├─ Slippage: Volume-based model                              │
│  ├─ Market Impact: Almgren-Chriss                              │
│  ├─ Latency: Random delay simulation                          │
│  └─ Commission: Realistic fee structure                       │
│                              ↓                                  │
│  Attribution Analysis                                           │
│  ├─ Backtest vs Live gap decomposition                        │
│  ├─ Data snooping detection                                    │
│  └─ Overfitting probability estimation                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 关键技术

### Walk-Forward Analysis

```python
from backtest_engine.walkforward.validator import WalkForwardValidator

validator = WalkForwardValidator(
    train_window=252*2,  # 2 years
    test_window=63,      # 3 months
    step_size=63,        # Roll 3 months
    purge_length=5,      # 5-day gap
)

results = validator.validate(data, strategy_factory, metrics_fn)
```

### Execution Simulation

```python
from backtest_engine.execution.simulator import ExecutionSimulator

simulator = ExecutionSimulator(
    slippage_model=VolumeBasedSlippage(),
    impact_model=MarketImpactModel(),
    latency_model=LatencySimulator(),
)

fill = simulator.execute_order(order, market_data, timestamp)
# Returns: fill_price, slippage, commission, market_impact
```

### Attribution Analysis

```python
from backtest_engine.analytics.attribution import AttributionAnalyzer

analyzer = AttributionAnalyzer(backtest_period, live_period)
report = analyzer.full_attribution_report()

# Components:
# - Execution attribution (slippage, commission)
# - Timing attribution (entry/exit differences)
# - Unexplained portion (data quality, etc.)
```

## 快速开始

### 安装

```bash
pip install pandas numpy scipy
```

### 运行回测

```python
from backtest_engine import BacktestEngine, BacktestConfig
from backtest_engine.data.pit_data import PointInTimeData

# Configuration
config = BacktestConfig(
    initial_capital=1_000_000,
    commission_rate=0.001,
)

# Create engine
engine = BacktestEngine(config)

# Load data
engine.load_data(price_data)

# Run backtest
results = engine.run_backtest(strategy)

# Run walk-forward validation
wf_results = engine.run_walk_forward(strategy_factory)
```

### 完整示例

```bash
python examples/momentum_strategy_example.py
```

## 核心概念详解

### 1. Point-in-Time Data

**问题**：使用修订后的数据产生不切实际的回测结果

**例子**：
- 公司财报最初发布时可能有错误
- 后续会发布修正版本
- 如果用修正后的数据回测，就是使用前视偏差

**解决方案**：
```python
# Original data point
data_point = {
    'date': '2020-03-31',
    'eps': 1.23,
    'available_at': '2020-05-15',  # When data actually became available
}
```

### 2. Walk-Forward Validation

**为什么不是简单train_test_split？**

时间序列数据有自相关性，随机分割会导致数据泄漏。

**正确做法**：
```
2020-2021 (Train) → 2022 Q1 (Test)
2020-2022 Q1 (Train) → 2022 Q2 (Test)
2020-2022 Q2 (Train) → 2022 Q3 (Test)
...
```

### 3. 执行成本模型

**滑点模型**：
```
slippage = base_bps + volume_impact * participation + volatility_factor
```

**市场冲击模型**（Almgren-Chriss）：
```
Temporary Impact = η * σ * (X/VT)^0.6
Permanent Impact = γ * σ * √(X/V)
```

Where:
- X: Order size
- V: Daily volume
- T: Execution time
- σ: Volatility

## 防止的回测陷阱

| 陷阱 | 描述 | 检测方法 |
|------|------|---------|
| **Look-ahead Bias** | 使用未来信息 | `DataSnoopingDetector.check_for_lookahead_bias()` |
| **Survivorship Bias** | 只测试存活股票 | `SurvivorshipBiasFreeData` 包含退市股 |
| **Data Snooping** | 多次测试选最优 | `deflated_sharpe_ratio()` 调整 |
| **Overfitting** | 过度拟合历史 | `probability_of_backtest_overfitting()` |
| **Execution Assumptions** |  unrealistic fills |  `ExecutionSimulator` 真实成本 |

## 严谨的研究实践

### 1. 多重验证

- **Walk-forward**: 主要验证方法
- **Combinatorial CV**: 生成多条回测路径
- **Paper trading**: 实盘前验证

### 2. 统计显著性

```python
# Deflated Sharpe Ratio
adjusted_sr = deflated_sharpe(
    observed_sr=2.0,
    num_trials=100,  # Number of strategies tested
)
```

### 3. 归因分析

区分真实alpha vs 数据挖掘：
- 执行成本归因
- 时序差异归因
- 未解释部分（数据质量）

## 项目结构

```
06_realistic_backtest_engine/
├── backtest_engine/
│   ├── core/
│   │   ├── config.py           # 配置系统
│   │   └── engine.py           # 主引擎
│   ├── data/
│   │   └── pit_data.py         # Point-in-Time数据
│   ├── execution/
│   │   └── simulator.py        # 执行仿真
│   ├── walkforward/
│   │   └── validator.py        # Walk-forward验证
│   ├── analytics/
│   │   └── attribution.py      # 归因分析
│   └── __init__.py
├── examples/
│   └── momentum_strategy_example.py
└── README.md
```

## 关键指标解读

### Backtest Quality Score

综合评估回测可信度：
- Walk-forward一致性: 40%
- 执行成本合理性: 20%
- 统计显著性: 20%
- 数据质量: 20%

### Probability of Backtest Overfitting (PBO)

使用CSCV (Combinatorially Symmetric Cross-Validation) 估计过拟合概率。

## 参考论文

1. **Advances in Financial Machine Learning** (Marcos Lopez de Prado, 2018)
   - Chapter 7: Cross-Validation in Finance
   - Chapter 12: Backtesting

2. **The Deflated Sharpe Ratio: Correcting for Selection Bias** (Bailey & Lopez de Prado, 2014)

3. **Optimal Execution of Portfolio Transactions** (Almgren & Chriss, 2000)

4. **The Probability of Backtest Overfitting** (Bailey et al., 2017)

## 简历表述建议

> "开发防前视偏差回测引擎，实现Point-in-Time数据访问、Walk-forward验证框架、真实执行成本模拟（含Almgren-Chriss市场冲击模型）。设计回测-实盘归因分析系统，量化执行成本、时序差异、数据质量等因素对策略表现的影响。确保回测结果的可靠性和可复现性，符合机构级量化研究标准。"

## License

MIT License

---

**郑重声明**：这是一个严谨的研究级回测框架，不是玩具回测器。遵循机构级最佳实践，确保回测结果的真实性和可复现性。
