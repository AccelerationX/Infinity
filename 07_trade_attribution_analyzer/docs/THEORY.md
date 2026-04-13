# 实盘交易归因分析系统 - 理论设计文档

## 1. 项目定位

本系统旨在建立一套**机构级的交易归因分析框架**，对量化策略从"信号生成"到"实际成交"的全链路进行严谨的数学分解与诊断。系统设计遵循学术严谨性与工程可落地性并重的原则，避免简历式表面工程。

## 2. 核心概念与符号体系

### 2.1 基础定义

| 符号 | 含义 |
|------|------|
| $S_t$ | 策略在 $t$ 时刻生成的理想信号（目标持仓/权重） |
| $P_t$ | 计划交易指令（由信号转换而来的理想订单） |
| $A_t$ | 实际执行的交易（经过市场摩擦后的成交结果） |
| $R_t^{total}$ | 组合总收益（实盘收益） |
| $R_t^{paper}$ | 纸面组合收益（理想执行下的收益） |
| $B_t$ | 基准组合权重/收益 |

### 2.2 时间线定义

对于每个交易日 $t$，事件顺序为：
1. **$t_0$**: 收盘后信号计算完成（基于 $t$ 日及之前的数据）
2. **$t_1$**: 次日开盘前生成交易指令（基于 $S_t$）
3. **$t_2$**: 盘中实际执行交易 $A_t$（产生成交价、成交量、滑点）
4. **$t_3$**: 收盘后持仓更新，可计算当日收益

## 3. 五维收益分解模型

总收益 $R^{total}$ 分解为五个正交（或近似正交）的组成部分：

### 3.1 选股 Alpha（Security Selection）

在**给定资产配置权重**下，个股选择带来的超额收益。

对于单期单资产：
$$\alpha_t^{selection} = \sum_i (w_{i,t}^{actual} - w_{i,t}^{benchmark}) \times (r_{i,t} - R_t^{benchmark})$$

其中 $w_{i,t}$ 为实际权重，$r_{i,t}$ 为个股收益。

**经济学含义**：反映了策略在正确的时间选择正确的股票的能力。

### 3.2 资产配置/择时（Asset Allocation / Timing）

资产配置决策带来的收益，即在不同资产类别/行业/风格上的权重偏离基准所带来的收益：

$$\alpha_t^{allocation} = \sum_i (w_{i,t}^{actual} - w_{i,t}^{benchmark}) \times R_t^{benchmark}$$

**注**：对于纯股票多头策略，此项通常较小；对于多资产策略（股债商等），此项是主要 alpha 来源之一。

### 3.3 市场 Beta（Market Exposure）

组合承担系统性风险所获得的市场收益：

$$R_t^{beta} = \beta_t \times R_t^{market}$$

其中 $\beta_t$ 可用组合历史收益对市场收益的回归估计，或用持仓加权个股 beta 计算。

**作用**：将总收益中"靠承担市场风险获得"的部分剥离，才能客观评价 alpha 能力。

### 3.4 执行成本（Execution Cost）

理想信号到实际成交之间的摩擦损耗，包括：
- 滑点（Slippage）
- 市场冲击（Market Impact）
- 延迟成本（Delay Cost）
- 部分成交/未成交的机会成本

$$C_t^{execution} = R_t^{paper} - R_t^{total}$$

（若 $C_t > 0$，表示执行侵蚀了收益；若 $< 0$，表示执行反而增厚了收益，如 lucky fill）

### 3.5 交互项（Interaction Effect）

Brinson 归因中的经典交互项，反映"好的股票+好的时机"的协同效应：

$$\alpha_t^{interaction} = \sum_i (w_{i,t}^{actual} - w_{i,t}^{benchmark}) \times (r_{i,t} - R_t^{benchmark})$$

在五维分解中，为避免与选股 alpha 重复计算，我们将 Brinson 的三项（选股+配置+交互）重新打包为：
- **纯选股效应**：在基准配置下选股带来的收益
- **纯配置效应**：用基准股票组合按实际权重配置带来的收益
- **执行效应**：信号到成交的损耗
- **Beta 效应**：市场风险暴露
- **残差/交互**：无法用上述解释的收益波动

更严谨的表达（基于持仓法）：

$$R_t^{total} = \underbrace{\sum_i w_{i,t}^{benchmark} r_{i,t}}_{\text{基准收益}} + \underbrace{\sum_i (w_{i,t}^{signal} - w_{i,t}^{benchmark}) r_{i,t}}_{\text{选股 Alpha}} + \underbrace{(R_t^{paper} - R_t^{signal})}_{\text{信号-纸面差异(通常为0)}} + \underbrace{(R_t^{total} - R_t^{paper})}_{\text{执行成本}} + \underbrace{\epsilon_t}_{\text{残差}}$$

## 4. 执行质量分析（Transaction Cost Analysis, TCA）

### 4.1 参考价格的选择

TCA 的核心是选择"应该成交的价格"作为比较基准。常见选择：

1. **决策时价格**（Decision Price, $P_{decision}$）：信号生成时刻的市场价格
2. **基准价格**（Benchmark Price, $P_{benchmark}$）：通常选用 VWAP、TWAP、前一收盘价、后一收盘价
3. **到达价格**（Arrival Price, $P_{arrival}$）：订单到达交易所时的市场价格

本系统采用**多基准 TCA**：
- 用决策价计算**延迟成本**（Delay Cost）
- 用到达价计算**市场冲击**（Market Impact）
- 用 VWAP 计算**执行质量**（Execution Quality）

### 4.2 延迟成本（Delay Cost / Slippage）

$$C^{delay} = \frac{P_{arrival} - P_{decision}}{P_{decision}} \times Q$$

反映了从"决定交易"到"订单到达市场"这段时间内价格的不利变动。对于低频策略（日频/周频），延迟成本是主要执行损耗来源。

### 4.3 市场冲击（Market Impact）

$$C^{impact} = \frac{P_{fill} - P_{arrival}}{P_{arrival}} \times Q$$

反映订单本身对市场价格的冲击。采用 Almgren-Chriss 框架进行建模验证：
- 临时冲击（Temporary Impact）：交易执行期间的即时价格偏离
- 永久冲击（Permanent Impact）：交易对市场价格水平的长期改变

### 4.4 执行利差（Execution Slippage）

$$C^{slippage} = \frac{P_{fill} - P_{benchmark}}{P_{benchmark}} \times Q$$

综合了延迟、冲击、流动性等因素的总执行偏差。

## 5. 机会成本（Opportunity Cost）

### 5.1 定义

对于未成交或部分成交的订单，机会成本定义为：

$$C^{opportunity} = \sum_{未成交i} Q_i^{unfilled} \times (P_{i,t+\tau}^{realized} - P_{i,t}^{decision})$$

其中 $\tau$ 为评估窗口（通常为持仓周期或固定几日）。

### 5.2 分类

1. **被动机会成本**：由于市场流动性不足导致无法成交
2. **主动机会成本**：由于风控/资金/人为干预导致订单被撤销
3. **延迟机会成本**：由于执行延迟导致价格已不利移动而取消订单

### 5.3 与执行成本的关系

$$C^{total\_friction} = C^{execution} + C^{opportunity}$$

两者必须同时计算才能完整反映"信号到收益"的转化效率。

## 6. 数据模型设计

### 6.1 核心实体

1. **Signal**（信号）：策略在 $t$ 时刻的目标状态
   - `signal_time`, `symbol`, `target_weight`, `target_shares`, `signal_price`

2. **Order**（指令）：由信号转换的交易指令
   - `order_time`, `symbol`, `side`, `order_qty`, `order_type`, `limit_price`

3. **Fill**（成交）：实际执行结果
   - `fill_time`, `symbol`, `fill_qty`, `fill_price`, `fill_status`

4. **Position**（持仓）：每日持仓快照
   - `date`, `symbol`, `shares`, `market_value`, `weight`

5. **MarketData**（市场数据）：用于归因计算
   - `date`, `symbol`, `open`, `high`, `low`, `close`, `vwap`, `volume`

### 6.2 归因计算流程

```
输入: Signals + Orders + Fills + Positions + MarketData
      |
      v
[Step 1] 构建理想组合 (Paper Portfolio)
      |
      v
[Step 2] 计算纸面收益 vs 基准收益 → 选股 Alpha + 配置效应
      |
      v
[Step 3] 计算实盘收益 vs 纸面收益 → 执行成本
      |
      v
[Step 4] TCA 分析: Delay Cost + Market Impact + Slippage
      |
      v
[Step 5] 机会成本计算 (未成交订单)
      |
      v
输出: 五维归因报告 + 执行质量报告 + 机会成本报告
```

## 7. 系统架构

```
07_trade_attribution_analyzer/
├── core/
│   ├── models.py              # 核心数据模型
│   ├── return_decomposition.py # 收益五维分解
│   ├── execution_attribution.py # TCA / 滑点归因
│   ├── opportunity_cost.py    # 机会成本计算
│   ├── benchmark.py           # 基准收益计算
│   └── engine.py              # 主引擎 orchestrator
├── tests/
│   ├── test_decomposition.py
│   ├── test_execution.py
│   ├── test_opportunity_cost.py
│   └── fixtures/              # 合成数据集
├── data/
│   └── sample_data/           # 示例数据
└── docs/
    └── THEORY.md              # 本文档
```

## 8. 设计原则

1. **数学严谨性**：所有分解公式都有清晰的数学定义，避免模糊的概念堆砌
2. **可加性验证**：各分解项之和必须严格等于总收益（允许机器精度误差）
3. **多基准支持**：不绑定单一 benchmark，支持自定义基准和多重比较
4. **可扩展性**：模块化设计，便于未来接入不同的策略类型和数据源
5. **可测试性**：提供合成数据集，所有核心计算都有对应的单元测试

## 9. 参考学术文献

- Brinson, G.P., Hood, L.R., & Beebower, G.L. (1986). "Determinants of Portfolio Performance." *Financial Analysts Journal*.
- Almgren, R., & Chriss, N. (2000). "Optimal Execution of Portfolio Transactions." *Journal of Risk*.
- Kissell, R., & Glantz, M. (2003). *Optimal Trading Strategies*. AMACOM.
- Perold, A.F. (1988). "The Implementation Shortfall: Paper versus Reality." *Journal of Portfolio Management*.
- Grinold, R.C., & Kahn, R.N. (2000). *Active Portfolio Management*. McGraw-Hill.

---
*本文档为系统设计的第一性原理基础，所有代码实现必须与此文档保持一致。*
