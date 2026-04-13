# 投资组合风险预警与诊断引擎 (Portfolio Risk Diagnosis Engine)

> 一个专业级的投资组合风险分析框架，提供风险分解、标准风险指标、压力测试与智能预警。

## 项目定位

本系统不是简历装饰，而是一个具备完整理论基础、可验证、可扩展的**机构级风险诊断引擎**。核心目标：

1. **数学严谨性**：风险分解满足可加性恒等式，各风险分量之和严格等于总风险
2. **工程可落地**：模块化设计，支持未来无缝接入实盘交易系统
3. **全维度覆盖**：从因子暴露、行业配置、个股集中到尾部风险、压力情景的全链路诊断

## 核心能力

### 1. 风险分解 (Risk Decomposition)

基于多因子模型将组合波动率严格分解为：
- **市场风险 (Market Risk)**：系统性市场因子暴露
- **行业风险 (Sector Risk)**：行业配置偏离
- **风格风险 (Style Risk)**：市值、价值、动量等风格暴露
- **个股集中风险 (Concentration Risk)**：通过 Effective N / HHI 度量
- **特异性风险 (Idiosyncratic Risk)**：无法被因子解释的残余风险

实现基于 **Litterman (1996) Hot Spots and Hedges** 框架的边际风险贡献 (MRC) 和绝对风险贡献 (ARC) 计算。

### 2. 标准风险指标 (Risk Metrics)

- **VaR (Value at Risk)**：参数法（正态假设）+ 历史模拟法
- **CVaR / ES (Expected Shortfall)**：尾部风险度量
- **最大回撤 (Max Drawdown)**：路径依赖型峰值-谷值分析
- **Beta / 跟踪误差 (Tracking Error)**：相对基准的系统性风险与主动风险
- **夏普比率 / 索提诺比率 (Sortino)**：风险调整后收益

### 3. 压力测试与情景分析 (Stress Testing)

- **线性近似法**：基于因子暴露快速估算情景下的组合损益
- **蒙特卡洛模拟**：对因子分布施加均值漂移，生成受压波动率估计
- **预定义情景库**：市场崩盘、流动性危机、风格反转、利率冲击等

### 4. 智能预警 (Risk Alerts)

三级预警体系：
- **提示 (INFO)**：70% 历史分位数
- **警告 (WARNING)**：85% 历史分位数
- **紧急 (CRITICAL)**：95% 历史分位数

支持静态阈值和动态滚动分位数两种模式。

## 项目结构

```
08_portfolio_risk_diagnosis/
├── core/
│   ├── __init__.py
│   ├── models.py                  # RiskSnapshot, FactorModel, RiskDecomposition 等
│   ├── risk_decomposition.py      # MRC / ARC / 分组风险分解
│   ├── risk_metrics.py            # VaR, CVaR, MDD, Beta, TE, Sharpe, Sortino
│   ├── stress_testing.py          # 情景压力测试 + 蒙特卡洛
│   ├── risk_alerts.py             # 预警分级与触发逻辑
│   └── engine.py                  # 风险诊断主引擎
├── tests/
│   ├── test_risk_decomposition.py
│   ├── test_risk_metrics.py
│   ├── test_stress_testing.py
│   ├── test_risk_alerts.py
│   └── test_engine.py
├── docs/
│   └── THEORY.md                  # 理论设计文档（必读）
├── examples/
│   └── demo_risk_diagnosis.py     # 合成数据演示
└── README.md
```

## 理论基础

详见 `docs/THEORY.md`，涵盖：
- 多因子模型下的组合方差分解
- Marginal / Absolute Risk Contribution 的数学推导
- VaR/CVaR 的参数法与历史模拟法
- 压力测试的线性近似与蒙特卡洛方法
- 预警分级的动态阈值设计

## 测试与验证

项目包含 **23 个单元测试**，覆盖所有核心计算路径。最重要的不变量：

```python
# 对于任何风险分解结果，以下恒等式必须成立
assert result.validate()  # ARC sum == total_volatility
```

运行测试：
```bash
python -m pytest tests/ -v
```

## 使用示例

```python
from core.engine import EngineConfig, RiskDiagnosisEngine
from core.models import FactorModel, RiskSnapshot
from core.risk_alerts import AlertRule

# Setup
engine = RiskDiagnosisEngine(EngineConfig(risk_free_rate=0.04))
engine.load_returns_history(returns_df)
engine.load_benchmark_returns(benchmark_series)
engine.load_factor_model(factor_model)
engine.load_factor_groups({"MARKET": ["MKT"], "SECTOR": ["TECH"]})
engine.load_alert_rules([
    AlertRule("volatility", info_threshold=0.18, warning_threshold=0.22, direction="above"),
])

# Run diagnosis
report = engine.run(RiskSnapshot(timestamp=pd.Timestamp.now(), weights=weights))

# Validate
assert report.decomposition.validate()

# Inspect
print(report.metrics.var_parametric_95)
print(report.stress_results)
print(report.alerts)
```

## 与 Trading 项目的集成远景

当前 Trading 项目已有 `risk_warning_system`（8轮迭代）、`position_risk_management`、`strategy_drawdown_concurrency_map` 等丰富研究。项目8的定位是**将这些研究成果工程化、标准化**：

1. **数据映射层**：将各策略线的持仓和收益数据映射为 `RiskSnapshot` + `FactorModel`
2. **统一风险日报**：用 `RiskDiagnosisEngine` 为每条策略线生成标准化风险报告
3. **组合级诊断**：把多条策略线合并为一个总组合，诊断策略间的相关性风险与拥挤度

## 开发状态

- [x] 理论设计文档
- [x] 核心数据模型
- [x] 风险分解实现 + 可加性验证
- [x] 风险指标计算（VaR/CVaR/MDD/Beta/TE/Sharpe/Sortino）
- [x] 压力测试模块（线性 + 蒙特卡洛）
- [x] 预警分级模块
- [x] 主引擎集成
- [x] 单元测试（23 tests passed）
- [x] Demo 脚本

## 参考文献

- Litterman, R. (1996). "Hot Spots and Hedges." *Journal of Portfolio Management*.
- Jorion, P. (2006). *Value at Risk: The New Benchmark for Managing Financial Risk*. McGraw-Hill.
- Grinold, R.C., & Kahn, R.N. (2000). *Active Portfolio Management*. McGraw-Hill.
- Rockafellar, R.T., & Uryasev, S. (2000). "Optimization of Conditional Value-at-Risk." *Journal of Risk*.
- Meucci, A. (2005). *Risk and Asset Allocation*. Springer.
