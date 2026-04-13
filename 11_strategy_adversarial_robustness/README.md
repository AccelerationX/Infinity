# 策略对抗鲁棒性与失效边界测试平台 (Strategy Adversarial Robustness)

> 一个专业级的策略压力测试框架，主动构造对抗性市场环境、系统扫描参数敏感性、诊断失效模式并给出综合鲁棒性评分。

## 项目定位

本系统不是简历装饰，而是一个具备完整理论基础、可验证、可扩展的**机构级策略健壮性测试引擎**。核心目标：

1. **主动发现边界**：不是被动等待策略失效，而是主动构造最不利条件来探测失效边界
2. **数学严谨性**：对抗路径生成、敏感性分析、鲁棒性评分都有清晰的数学定义
3. **可扩展性**：策略接口完全抽象化，支持任意满足协议的量化策略

## 核心能力

### 1. 对抗性市场模拟 (Adversarial Market Simulation)

基于带 regime-switching 的几何布朗运动，生成对策略不利的价格路径：
- **趋势反转 (Trend Reversal)**：在策略多头暴露最大时，漂移突变为负
- **波动率跳升 (Volatility Jump)**：波动率在基准基础上倍增
- **相关性崩溃 (Correlation Crash)**：资产间相关性突增至接近 1
- **流动性枯竭 (Liquidity Dry-up)**：成交量骤降，冲击成本扩大

### 2. 参数敏感性分析 (Parameter Sensitivity)

- **网格搜索 (Grid Search)**：低维参数空间的穷尽扫描
- **Sobol 序列采样**：中等维度下的低差异均匀覆盖
- **Morris 筛选法**：高维参数空间的基本效应 (Elementary Effects) 筛选，快速识别关键参数
- **稳定区域体积**：量化参数空间中策略表现不低于基准 90% 的体积占比

### 3. 失效模式分类 (Failure Mode Classification)

识别并量化五种典型的策略失效模式：
- **Regime Breakdown**：市场状态切换导致信号失效
- **Crowding Crash**：策略拥挤导致 Alpha 反转
- **Liquidity Trap**：高换手假设在流动性不足时崩塌
- **Factor Decay**：依赖的因子发生长期回撤或短期极端反转
- **Data Snooping**：样本内过拟合导致的样本外表现崩塌

### 4. 鲁棒性评分 (Robustness Score)

综合评分模型：
```
Robustness = 0.30 * Adversarial_Survival
           + 0.25 * Parameter_Stability
           + 0.25 * Stress_Test_Score
           + 0.20 * Walkforward_Consistency
```

评级标准：
- **Excellent (0.85-1.00)**：可放心实盘
- **Good (0.70-0.85)**：需设置风控
- **Fair (0.55-0.70)**：需大幅优化
- **Fragile (<0.55)**：不建议实盘

## 项目结构

```
11_strategy_adversarial_robustness/
├── core/
│   ├── __init__.py
│   ├── models.py                  # Strategy, StrategyResult, AdversarialPath 等
│   ├── adversarial_market.py      # GBM 路径生成 + 对抗性注入
│   ├── parameter_sensitivity.py   # Grid / Sobol / Morris / Stability
│   ├── failure_modes.py           # 五种失效模式诊断
│   ├── robustness_score.py        # 综合鲁棒性评分
│   └── engine.py                  # 主引擎
├── tests/
│   ├── test_adversarial_market.py
│   ├── test_parameter_sensitivity.py
│   ├── test_failure_modes.py
│   ├── test_robustness_score.py
│   └── test_engine.py
├── docs/
│   └── THEORY.md                  # 理论设计文档（必读）
├── examples/
│   └── demo_robustness.py         # 合成数据演示
└── README.md
```

## 理论基础

详见 `docs/THEORY.md`，涵盖：
- 对抗性市场路径的数学形式化
- GBM + Regime-Switching + Jump 模型
- Morris Elementary Effects 的定义与解释
- 失效模式的经济学含义与诊断指标
- 鲁棒性综合评分的加权设计

## 测试与验证

项目包含 **26 个单元测试**，覆盖所有核心计算路径。

运行测试：
```bash
python -m pytest tests/ -v
```

## 使用示例

```python
from core.engine import RobustnessEngine, EngineConfig
from core.models import Strategy, StrategyResult

class MyStrategy(Strategy):
    def run(self, prices: pd.DataFrame, **params) -> StrategyResult:
        # Your strategy logic here
        ...

engine = RobustnessEngine(EngineConfig(seed=42))
report = engine.run_full_analysis(
    strategy=MyStrategy(),
    strategy_name="my_strategy",
    base_params={"lookback": 10, "top_n": 5},
    baseline_prices=prices_df,
    param_sensitivity_config={"lookback": [5, 10, 20]},
    param_bounds={"lookback": (2.0, 30.0), "top_n": (1.0, 10.0)},
)

print(report.robustness_score.rating)
print(report.failure_diagnoses)
```

## 与 Trading 项目的集成远景

对 Trading 项目的 46 条策略线批量运行失效边界扫描：
1. 为每条策略线生成对抗性测试报告
2. 按鲁棒性评分排序，筛选出"稳健策略"和"脆弱策略"
3. 对脆弱策略进行参数敏感性分析，找出优化方向
4. 输出 `strategy_robustness_ranking.csv` 供策略配置决策使用

## 开发状态

- [x] 理论设计文档
- [x] 核心数据模型
- [x] 对抗性市场模拟（Trend / Vol / Correlation）
- [x] 参数敏感性分析（Grid / Sobol / Morris）
- [x] 失效模式诊断（5 modes）
- [x] 鲁棒性综合评分
- [x] 主引擎集成
- [x] 单元测试（26 tests passed）
- [x] Demo 脚本

## 参考文献

- Szegedy, C., et al. (2013). "Intriguing Properties of Neural Networks."
- Saltelli, A., et al. (2008). *Global Sensitivity Analysis: The Primer*. Wiley.
- Morris, M.D. (1991). "Factorial Sampling Plans for Preliminary Computational Experiments." *Technometrics*.
- White, H. (2000). "A Reality Check for Data Snooping." *Econometrica*.
- Harvey, C.R., & Liu, Y. (2015). "Backtesting." *Journal of Portfolio Management*.
