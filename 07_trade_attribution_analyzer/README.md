# 实盘交易归因分析系统 (Trade Attribution Analyzer)

> 一个专业级的交易归因框架，用于量化策略从"信号生成"到"实际成交"的全链路收益分解与执行质量诊断。

## 项目定位

本系统不是简历装饰，而是一个具备完整理论基础、可验证、可扩展的**机构级归因分析引擎**。核心目标：

1. **数学严谨性**：所有分解满足可加性恒等式（各分量之和严格等于总收益）
2. **工程可落地**：模块化设计，支持未来无缝接入实盘交易系统
3. **全链路覆盖**：从信号 → 指令 → 成交 → 持仓，每个环节的损耗都可量化

## 核心能力

### 1. 五维收益分解 (Return Decomposition)

将组合总收益严格分解为：
- **基准收益 (Benchmark Return)**：被动持有基准组合的收益
- **选股 Alpha (Security Selection)**：在基准配置下，个股选择带来的超额收益
- **资产配置效应 (Allocation Effect)**：权重偏离基准带来的收益
- **交互效应 (Interaction)**：选股与配置的协同效应
- **执行成本 (Execution Cost)**：理想纸面组合与实盘组合之间的收益差距
- **Beta 收益**：系统性风险暴露的市场收益
- **残差 (Residual)**：确保可加性恒等式严格成立的平衡项

实现基于经典 **Brinson 归因模型** 的扩展，支持单期分析与多期链接（算术链接 / 几何链接）。

### 2. 执行质量分析 (Transaction Cost Analysis, TCA)

对每一笔成交进行多维度成本拆解：
- **延迟成本 (Delay Cost)**：从决策到订单到达市场期间的价格变动
- **市场冲击 (Market Impact)**：订单本身对市场价格的影响
- **滑点 (Slippage)**：成交价 vs 基准价（如 VWAP）的偏离
- **显式费用 (Fees)**：佣金、税费等

支持 **Perold (1988) 实施缺口 (Implementation Shortfall)** 的portfolio级汇总。

### 3. 机会成本计算 (Opportunity Cost)

量化未成交或部分成交订单的潜在收益损失：
- 分类统计：流动性不足、主动撤销、被后续周期覆盖等
- 多时间窗口评估：T+1, T+5, T+20 等
- 与执行成本合并计算总摩擦 (Total Friction)

## 项目结构

```
07_trade_attribution_analyzer/
├── core/
│   ├── __init__.py
│   ├── models.py                  # 核心数据模型
│   ├── return_decomposition.py    # 收益五维分解
│   ├── execution_attribution.py   # TCA / 执行滑点归因
│   ├── opportunity_cost.py        # 机会成本计算
│   └── engine.py                  # 主归因引擎
├── tests/
│   ├── test_return_decomposition.py
│   ├── test_execution_attribution.py
│   ├── test_opportunity_cost.py
│   └── test_engine.py
├── docs/
│   └── THEORY.md                  # 理论设计文档（必读）
└── README.md
```

## 理论基础

详见 `docs/THEORY.md`，涵盖：
- Brinson 归因模型的数学推导
- Perold Implementation Shortfall 框架
- Almgren-Chriss 市场冲击模型的引用
- 可加性验证与多期链接方法

## 测试与验证

项目包含 **25+ 单元测试**，覆盖所有核心计算路径。最重要的不变量：

```python
# 对于任何归因结果，以下恒等式必须成立
assert result.validate()  # components sum == total_return
```

运行测试：
```bash
python -m pytest tests/ -v
```

## 使用示例

```python
from core.engine import AttributionEngine, EngineConfig
from core.models import Signal, Order, Fill, Side, OrderType, FillStatus

config = EngineConfig(opportunity_horizon_days=5)
engine = AttributionEngine(config)

engine.load_signals([...])
engine.load_orders([...])
engine.load_fills([...])
engine.load_positions([...])
engine.load_market_data({...})

result = engine.run_full_analysis(
    period_label="2024-Q1",
    benchmark_weights=benchmark_w,
    asset_returns=returns,
    evaluation_prices=eval_prices,
)

# 验证可加性
assert result["return_attribution"].validate()

# 查看执行成本
print(result["execution_summary"])

# 查看机会成本
print(result["opportunity_summary"])
```

## 与 Trading 项目的集成规划

当前 Trading 项目的实盘交易数据尚处于积累阶段，因此本项目优先完成**理论框架与核心引擎**的建设。未来集成路径：

1. **数据映射层**：将 `main_trade_execution_log.csv` 映射为 `Order` + `Fill` 模型
2. **持仓同步**：将周期持仓映射为 `Position` 快照
3. **市场数据桥接**：接入 `StockHistory` 的价格数据用于 VWAP / 到达价计算
4. **自动化报告**：为每条策略线生成周期性归因报告

## 开发状态

- [x] 理论设计文档
- [x] 核心数据模型
- [x] 收益五维分解实现 + 可加性验证
- [x] 执行归因 (TCA) 实现
- [x] 机会成本计算实现
- [x] 主引擎 (Engine) 集成
- [x] 单元测试覆盖
- [ ] 公开数据集验证示例（下一步）
- [ ] 可视化报告模块（下一步）
- [ ] Trading 项目数据桥接（待数据积累后）

## 参考文献

- Brinson, G.P., Hood, L.R., & Beebower, G.L. (1986). "Determinants of Portfolio Performance." *Financial Analysts Journal*.
- Perold, A.F. (1988). "The Implementation Shortfall: Paper versus Reality." *Journal of Portfolio Management*.
- Almgren, R., & Chriss, N. (2000). "Optimal Execution of Portfolio Transactions." *Journal of Risk*.
- Kissell, R., & Glantz, M. (2003). *Optimal Trading Strategies*. AMACOM.
