# 投资组合风险预警与诊断引擎 - 理论设计文档

## 1. 项目定位

本系统旨在建立一套**机构级的投资组合风险诊断框架**，对量化策略组合进行多维度的风险分解、实时监控、压力测试与智能预警。系统设计遵循金融工程理论与工程实践并重的原则，拒绝表面工程。

## 2. 核心符号体系

| 符号 | 含义 |
|------|------|
| $w$ | 组合权重向量 $(n \times 1)$ |
| $r$ | 资产收益率向量 $(n \times 1)$ |
| $R_p = w'r$ | 组合收益率（标量） |
| $\Sigma$ | 资产收益率协方差矩阵 $(n \times n)$ |
| $\sigma_p = \sqrt{w'\Sigma w}$ | 组合收益波动率 |
| $B$ | 因子暴露矩阵 $(n \times k)$ |
| $f$ | 因子收益率向量 $(k \times 1)$ |
| $\Sigma_f$ | 因子收益率协方差矩阵 $(k \times k)$ |
| $D$ | 特异性收益方差对角矩阵 $(n \times n)$ |
| $\beta_p$ | 组合对市场指数的 Beta |

## 3. 风险分解模型

### 3.1 组合方差的因子模型分解

假设资产收益服从多因子模型：

$$r = B f + \epsilon$$

其中 $\epsilon$ 为特异性收益，满足 $E[\epsilon]=0$, $Cov(\epsilon)=D$, 且 $Cov(\epsilon, f)=0$。

则协方差矩阵可分解为：

$$\Sigma = B \Sigma_f B' + D$$

组合方差 $\sigma_p^2 = w'\Sigma w$ 可进一步分解为：

$$\sigma_p^2 = \underbrace{w'B \Sigma_f B'w}_{\text{因子风险}} + \underbrace{w'Dw}_{\text{特异性风险}}$$

### 3.2 因子风险的多层分解

将因子风险继续细分为：

$$w'B \Sigma_f B'w = \sum_{k=1}^{K} \sum_{j=1}^{K} (w'b_k)(w'b_j) Cov(f_k, f_j)$$

其中 $b_k$ 为第 $k$ 个因子的暴露向量。根据因子的经济含义，可将 $K$ 个因子划分为：
- **市场风险 (Market Risk)**：市场收益因子
- **行业风险 (Sector Risk)**：行业 dummy 因子
- **风格风险 (Style Risk)**：市值、价值、动量等风格因子
- **宏观风险 (Macro Risk)**：利率、通胀、汇率等宏观因子（如适用）

通过将因子分组，可计算每类风险对组合总风险的边际贡献 (MRC) 和绝对贡献 (ARC)：

**边际风险贡献 (Marginal Risk Contribution)**：
$$MRC_i = \frac{\partial \sigma_p}{\partial w_i} = \frac{(\Sigma w)_i}{\sigma_p}$$

**绝对风险贡献 (Absolute Risk Contribution)**：
$$ARC_i = w_i \cdot MRC_i = \frac{w_i (\Sigma w)_i}{\sigma_p}$$

满足可加性：$\sum_i ARC_i = \sigma_p$

### 3.3 集中度风险

集中度风险用 Herfindahl-Hirschman Index (HHI) 度量：

$$HHI = \sum_i w_i^2$$

- $HHI$ 接近 $1/N$：高度分散
- $HHI$ 接近 $1$：高度集中

有效资产数量（Effective Number of Assets）：
$$N_{eff} = \frac{1}{HHI}$$

### 3.4 风险分解的可加性验证

对于基于波动率的风险分解，必须满足：

$$\sigma_p = \sum_{g \in \text{Groups}} ARC_g + ARC_{idio}$$

其中 $ARC_g$ 为第 $g$ 类因子的绝对风险贡献，$ARC_{idio}$ 为特异性风险的绝对贡献。

## 4. 风险度量指标

### 4.1 VaR (Value at Risk)

**参数法（方差-协方差法）**：
假设组合收益服从正态分布 $R_p \sim N(\mu_p, \sigma_p^2)$，则：

$$VaR_{\alpha} = \mu_p + z_{\alpha} \cdot \sigma_p$$

其中 $z_{\alpha}$ 为标准正态分布的 $\alpha$ 分位数（如 $z_{0.05} = -1.645$）。

**历史模拟法**：

$$VaR_{\alpha} = \text{Quantile}(\{R_{p,t}\}_{t=1}^{T}, \alpha)$$

不假设分布，但对历史数据长度敏感。

### 4.2 CVaR / ES (Expected Shortfall)

CVaR 度量在损失超过 VaR 条件下的平均损失：

**参数法**：
$$CVaR_{\alpha} = \mu_p - \frac{\phi(z_{\alpha})}{\alpha} \cdot \sigma_p$$

其中 $\phi(\cdot)$ 为标准正态密度函数。

**历史模拟法**：
$$CVaR_{\alpha} = \frac{1}{|\{t: R_{p,t} \leq VaR_{\alpha}\}|} \sum_{t: R_{p,t} \leq VaR_{\alpha}} R_{p,t}$$

### 4.3 最大回撤 (Max Drawdown)

设 $V_t$ 为 $t$ 时刻的组合净值，则：

$$MDD = \max_{0 \leq s \leq t \leq T} \left( \frac{V_s - V_t}{V_s} \right)$$

等价于：

$$MDD = \max_{t} \left( \frac{\max_{s \leq t} V_s - V_t}{\max_{s \leq t} V_s} \right)$$

### 4.4 Beta 与跟踪误差

**组合 Beta**：
$$\beta_p = \frac{Cov(R_p, R_m)}{Var(R_m)} = \frac{w' \Sigma_{rm}}{\sigma_m^2}$$

其中 $R_m$ 为市场指数收益，$\Sigma_{rm}$ 为各资产与市场收益的协方差向量。

**跟踪误差 (Tracking Error)**：
$$TE = \sqrt{(w - w_b)' \Sigma (w - w_b)}$$

其中 $w_b$ 为基准权重。

### 4.5 风险调整后收益

**夏普比率 (Sharpe Ratio)**：
$$SR = \frac{\mu_p - r_f}{\sigma_p}$$

**索提诺比率 (Sortino Ratio)**：
$$SOR = \frac{\mu_p - r_f}{\sigma_d}$$

其中 downside deviation：
$$\sigma_d = \sqrt{\frac{1}{T} \sum_{t=1}^{T} \min(R_{p,t} - r_f, 0)^2}$$

## 5. 压力测试与情景分析

### 5.1 线性近似法

假设组合收益对各风险因子呈线性关系：

$$R_p = w' r \approx w' (B f + \epsilon) = \sum_k (w' b_k) f_k + w' \epsilon$$

在情景 $(\Delta f, \Delta \epsilon)$ 下，组合损益 (P&L) 为：

$$\Delta R_p = \sum_k (w' b_k) \Delta f_k + w' \Delta \epsilon$$

### 5.2 预定义情景库

| 情景名称 | 描述 | 冲击参数 |
|---------|------|---------|
| Market Crash | 市场整体下跌20% | $\Delta f_{market} = -0.20$ |
| Liquidity Crisis | 换手率降50%，冲击成本翻倍 | 流动性因子 $-0.50$，波动率因子 $+0.30$ |
| Sector Shock | 单一行业下跌30% | 目标行业因子 $-0.30$ |
| Style Reversal | 价值/动量极端反转 | 价值因子 $+0.15$，动量因子 $-0.15$ |
| Interest Rate Rise | 利率上行100bps | 利率因子 $+0.01$（日度近似） |
| Correlation Breakdown | 分散化失效 | 所有资产间相关系数 $\rightarrow 1$ |

### 5.3 蒙特卡洛压力测试

对于非线性组合或复杂衍生品，可采用蒙特卡洛模拟：

1. 从历史数据估计因子模型参数 $(B, \Sigma_f, D)$
2. 对情景因子施加均值漂移：$f_{shock} \sim N(\Delta f, \Sigma_f)$
3. 模拟 $N$ 条资产收益路径
4. 计算组合损益分布，提取关键分位数

## 6. 预警分级体系

### 6.1 阈值设定

支持两种阈值模式：

**静态阈值**：基于业务规则或监管要求预设

**动态阈值**：基于滚动历史分布的分位数

$$Threshold_{level} = \text{Quantile}(\{RiskMetric_t\}_{t=\tau-L}^{\tau}, q_{level})$$

其中 $q_{level}$ 根据预警级别设定：
- 提示 (Info)：$q = 0.70$
- 警告 (Warning)：$q = 0.85$
- 紧急 (Critical)：$q = 0.95$

### 6.2 预警触发逻辑

对于每个风险指标 $x$，在当前时刻 $t$：

```
if x_t >= Threshold_critical:   level = CRITICAL
elif x_t >= Threshold_warning:  level = WARNING
elif x_t >= Threshold_info:     level = INFO
else:                            level = NORMAL
```

### 6.3 多指标综合预警

可采用"最高级别优先"原则：若多个指标同时触发不同级别，取最高级别。

或采用加权评分法：
$$AlertScore = \sum_i w_i \cdot \mathbb{1}(x_i \geq Threshold_i) \cdot Severity_i$$

## 7. 系统架构

```
08_portfolio_risk_diagnosis/
├── core/
│   ├── models.py              # 数据模型：RiskSnapshot, RiskDecomposition, StressResult, Alert
│   ├── risk_decomposition.py  # 因子模型风险分解（MRC / ARC）
│   ├── risk_metrics.py        # VaR, CVaR, MDD, Beta, TE, Sharpe, Sortino
│   ├── stress_testing.py      # 情景压力测试 + 蒙特卡洛
│   ├── risk_alerts.py         # 预警分级与触发逻辑
│   └── engine.py              # 风险诊断主引擎
├── tests/                     # 单元测试（可加性验证、单调性验证等）
├── docs/THEORY.md             # 本文档
└── examples/                  # 合成数据演示
```

## 8. 设计原则

1. **可加性验证**：风险分解各分量之和必须严格等于总风险
2. **多方法交叉验证**：VaR/CVaR 同时提供参数法和历史模拟法，结果应大致一致
3. **情景单调性**：压力测试的冲击越大，损失越大（线性近似下严格成立）
4. **可扩展性**：支持用户自定义因子、自定义情景、自定义风险指标
5. **实时性**：引擎接口设计支持每日/每小时的风险快照计算

## 9. 参考文献

- Litterman, R. (1996). "Hot Spots and Hedges." *Journal of Portfolio Management*.
- Jorion, P. (2006). *Value at Risk: The New Benchmark for Managing Financial Risk*. McGraw-Hill.
- Grinold, R.C., & Kahn, R.N. (2000). *Active Portfolio Management*. McGraw-Hill.
- Rockafellar, R.T., & Uryasev, S. (2000). "Optimization of Conditional Value-at-Risk." *Journal of Risk*.
- Meucci, A. (2005). *Risk and Asset Allocation*. Springer.

---
*本文档为项目8的第一性原理基础，所有代码实现必须与此文档保持一致。*
