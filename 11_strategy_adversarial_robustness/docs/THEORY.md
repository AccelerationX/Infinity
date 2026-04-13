# 策略对抗鲁棒性与失效边界测试平台 - 理论设计文档

## 1. 项目定位

本系统旨在建立一套**主动攻击和测试量化策略的严谨框架**，系统性地发现策略的失效边界、评估其在恶劣市场环境下的生存能力。核心目标是回答：

- **策略在什么条件下会失效？**
- **参数扰动对策略表现的敏感度如何？**
- **策略的"安全运行边界"在哪里？**

系统设计遵循学术严谨性与工程可验证性并重的原则，拒绝表面工程。

## 2. 核心概念

### 2.1 对抗性市场 (Adversarial Market)

传统回测使用历史数据或随机模拟，而对抗性市场模拟**主动构造对策略最不利的价格路径**。这种思想源于对抗样本 (Adversarial Examples) 在金融领域的延伸应用。

对于给定策略 $S$ 和价格路径 $P = (p_1, p_2, ..., p_T)$，对抗性路径 $P^*$ 可形式化为：

$$P^* = \arg\min_{P \in \mathcal{P}} \mathcal{U}(S(P))$$

其中 $\mathcal{U}$ 为策略的效用函数（如累计收益），$\mathcal{P}$ 为可行价格路径集合（受波动率、趋势等约束）。

### 2.2 失效边界 (Failure Boundary)

策略的失效边界定义为参数空间或市场环境空间中，策略表现跌破预设最低可接受水平的临界超曲面：

$$\mathcal{B} = \{ \theta \in \Theta \mid \mathcal{U}(S_\theta) = \mathcal{U}_{min} \}$$

其中 $\theta$ 为策略参数或市场环境参数向量。

### 2.3 鲁棒性 (Robustness)

策略鲁棒性指其在参数扰动、市场环境变化和噪声干扰下保持表现的能力。可用**局部鲁棒性**（参数邻域内的稳定性）和**全局鲁棒性**（全参数空间内的最差表现）两个维度度量。

## 3. 对抗性市场模拟

### 3.1 基于GBM的对抗路径生成

基础模型为带 regime-switching 的几何布朗运动：

$$\frac{dS_t}{S_t} = \mu_t dt + \sigma_t dW_t + J_t dN_t$$

其中：
- $\mu_t$：时变漂移项
- $\sigma_t$：时变波动率
- $J_t$：跳跃幅度
- $dN_t$：泊松跳跃过程

**对抗性控制**：
- **趋势反转**：在策略多头暴露最大时，令 $\mu_t$ 突变为负
- **波动率跳升**：从低波动环境 ($\sigma_{low}$) 切换到高波动 ($\sigma_{high}$)
- **流动性枯竭**：通过成交量下降模拟买卖价差扩大和市场冲击增加
- **相关性崩溃**：资产间相关系数矩阵突变为高相关

### 3.2 相关性崩溃建模

正常状态下资产收益的相关矩阵为 $\mathbf{C}$。在相关性崩溃情景下：

$$\mathbf{C}_{crash} = \lambda \mathbf{1} + (1 - \lambda) \mathbf{C}$$

其中 $\mathbf{1}$ 为全1矩阵，$\lambda \in [0,1]$ 控制崩溃强度。需保证 $\mathbf{C}_{crash}$ 仍为正定矩阵，通常 $\lambda < 1/(n-1)$ 即可满足。

### 3.3 对抗强度参数化

每种对抗情景可用强度参数 $\delta \in [0,1]$ 参数化，便于构建从"温和压力"到"极端压力"的连续谱：

| 情景 | $\delta=0$ | $\delta=1$ |
|------|-----------|------------|
| Trend Reversal | 无反转 | 趋势完全反向，漂移 = $-2\mu$ |
| Volatility Jump | 基准波动率 | 波动率翻倍 |
| Liquidity Dry-up | 正常成交量 | 成交量降至10% |
| Correlation Crash | 正常相关矩阵 | 所有资产相关性 → 0.9 |

## 4. 参数敏感性分析

### 4.1 网格搜索 (Grid Search)

在参数空间的离散格点上系统评估策略表现：

$$\Theta_{grid} = \Theta_1 \times \Theta_2 \times ... \times \Theta_d$$

其中 $\Theta_i$ 为第 $i$ 个参数的离散取值集合。适用于低维参数空间（$d \leq 3$）。

### 4.2 Sobol 序列采样

对于中等维度（$3 < d \leq 10$），使用 **Sobol 低差异序列** 替代随机采样，以更少的样本获得更均匀的空间覆盖：

$$\theta^{(k)} = (S_k^{(1)}, S_k^{(2)}, ..., S_k^{(d)}), \quad k = 1, ..., N$$

其中 $S_k^{(i)}$ 为第 $i$ 维的第 $k$ 个 Sobol 点，通过逆变换映射到参数范围 $[\theta_i^{min}, \theta_i^{max}]$。

### 4.3 Morris 筛选法 (Elementary Effects)

Morris 法通过计算参数的"基本效应" (Elementary Effect) 来筛选重要参数：

对于参数 $i$，在网格 $p$ 维上选择基准点 $x$，定义基本效应：

$$EE_i(x) = \frac{\mathcal{U}(x_1, ..., x_i + \Delta, ..., x_p) - \mathcal{U}(x)}{\Delta}$$

通过多次采样取均值和方差：
- **$\mu_i$**（均值）：参数 $i$ 对输出的整体影响强度
- **$\sigma_i$**（标准差）：参数 $i$ 的非线性/交互效应强度

Morris 法的计算复杂度为 $O(p \cdot r)$，其中 $r$ 为轨迹数，适合高维初步筛选。

### 4.4 参数稳定区域

定义参数稳定区域为策略表现不低于基准表现 $90\%$ 的参数子空间：

$$\mathcal{R}_{stable} = \{ \theta \in \Theta \mid \mathcal{U}(S_\theta) \geq 0.9 \cdot \mathcal{U}(S_{baseline}) \}$$

稳定区域体积占比可量化为鲁棒性指标：

$$StabilityScore = \frac{Vol(\mathcal{R}_{stable})}{Vol(\Theta)}$$

## 5. 失效模式分类

### 5.1 Regime Breakdown（状态切换失效）

**特征**：策略依赖的历史统计关系在当前市场状态下不再成立。

**诊断指标**：
- 策略收益与市场状态隐变量的条件相关性突变
- 滚动窗口内策略夏普比率显著下降

**对抗测试**：在策略表现最好的历史状态下注入反向冲击。

### 5.2 Crowding Crash（拥挤崩溃）

**特征**：大量资金采用相似策略，导致 Alpha 反转或流动性枯竭。

**诊断指标**：
- 持仓与同类策略持仓的高度重叠
- 高换手率 + 高集中度

**对抗测试**：模拟"同类策略同时平仓"的价格冲击。

### 5.3 Liquidity Trap（流动性陷阱）

**特征**：策略在纸面回测中假设无限流动性，但实盘无法以理想价格成交。

**诊断指标**：
- 持仓市值占日均成交量的比例
- 换手率与市场冲击成本的敏感度

**对抗测试**：成交量骤降情景下的执行缺口分析。

### 5.4 Factor Decay（因子衰减）

**特征**：策略依赖的风格因子发生长期回撤或短期极端反转。

**诊断指标**：
- 组合收益与目标因子的暴露-收益关系破裂
- 因子收益率 rolling IR 持续为负

**对抗测试**：对目标因子施加持续负向漂移。

### 5.5 Data Snooping（数据窥探）

**特征**：样本内过度优化导致样本外表现崩塌。

**诊断指标**：
- 样本内/样本外夏普比率差异过大
- 参数数量 vs 有效样本量的比例

**对抗测试**：White's Reality Check 或 Hansen's Superior Predictive Ability (SPA) 检验。

## 6. 鲁棒性评分模型

### 6.1 综合评分公式

$$Robustness = w_1 \cdot AAS + w_2 \cdot PSS + w_3 \cdot STS + w_4 \cdot WFC$$

其中：
- **$AAS$ (Adversarial Survival Score)**：对抗测试中存活率（最终净值 > 0.5 的比例）
- **$PSS$ (Parameter Stability Score)**：参数稳定区域体积占比
- **$STS$ (Stress Test Score)**：压力测试下的最差夏普比率 / 基准夏普比率
- **$WFC$ (Walkforward Consistency)**：滚动窗口内表现的标准差倒数（归一化）

默认权重：$w_1 = 0.30, w_2 = 0.25, w_3 = 0.25, w_4 = 0.20$

### 6.2 评分解释

| 分数区间 | 鲁棒性评级 | 建议 |
|---------|-----------|------|
| 0.85 - 1.00 | 优秀 | 可放心实盘 |
| 0.70 - 0.85 | 良好 | 需设置止损/风控 |
| 0.55 - 0.70 | 一般 | 需大幅优化参数或逻辑 |
| 0.00 - 0.55 | 脆弱 | 不建议实盘，需重新设计 |

## 7. 系统架构

```
11_strategy_adversarial_robustness/
├── core/
│   ├── __init__.py
│   ├── models.py                  # StrategyResult, StressPath, SensitivityResult 等
│   ├── adversarial_market.py      # 对抗性价格路径生成
│   ├── parameter_sensitivity.py   # 网格搜索 / Sobol / Morris 筛选
│   ├── failure_modes.py           # 失效模式分类与诊断
│   ├── robustness_score.py        # 鲁棒性综合评分
│   └── engine.py                  # 主引擎
├── tests/                         # 单元测试
├── docs/
│   └── THEORY.md                  # 本文档
└── examples/
    └── demo_robustness.py         # 合成数据演示
```

## 8. 设计原则

1. **可复现性**：所有随机模拟固定种子，结果可完全复现
2. **可扩展性**：策略接口抽象化，支持任意满足接口的量化策略
3. **可解释性**：每个评分和诊断结果都有清晰的数学定义和经济含义
4. **渐进压力**：对抗情景从温和到极端连续可调，便于绘制"生存曲线"

## 9. 参考文献

- Szegedy, C., et al. (2013). "Intriguing Properties of Neural Networks." (对抗样本思想来源)
- Saltelli, A., et al. (2008). *Global Sensitivity Analysis: The Primer*. Wiley.
- Morris, M.D. (1991). "Factorial Sampling Plans for Preliminary Computational Experiments." *Technometrics*.
- White, H. (2000). "A Reality Check for Data Snooping." *Econometrica*.
- Harvey, C.R., & Liu, Y. (2015). "Backtesting." *Journal of Portfolio Management*.
- Bailey, D.H., & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio." *Journal of Portfolio Management*.

---
*本文档为项目11的第一性原理基础，所有代码实现必须与此文档保持一致。*
