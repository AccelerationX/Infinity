# 量化研究工作台与策略实验平台 (Quant Research Workbench)

> 一个轻量级、零侵入的量化研究实验管理平台，支持策略实验追踪、结果对比和参数优化。

## 项目定位

本系统是量化研究的**元平台 (Meta-Platform)**，不直接产生交易策略，而是解决研究过程中的核心基础设施问题：

1. **实验可复现**：完整记录每次运行的代码版本、参数配置、输出结果
2. **结果可对比**：方便地并排比较不同实验的指标差异
3. **参数可优化**：内置网格搜索、随机搜索，预留贝叶斯优化接口

系统设计遵循**最小依赖**原则：仅依赖 pandas 和 numpy，不需要数据库或外部服务。

## 核心能力

### 1. 实验追踪 (Experiment Tracking)

上下文管理器风格，零侵入集成：

```python
with tracker.start_run(experiment_id="exp_001") as run:
    run.log_params({"top_n": 30, "rebalance": 5})
    run.log_metrics({"cagr": 1.23, "mdd": -0.15})
    run.log_artifact("backtest_report.csv")
```

支持嵌套字典参数的自动扁平化（如 `{"model": {"lr": 0.01}}` → `model.lr=0.01`），便于后续分析。

### 2. 结果对比 (Experiment Comparison)

- **对比表生成**：将多个运行的参数和指标合并为一张宽表
- **Bootstrap 显著性检验**：判断两组实验的指标差异是否具有统计显著性
- **参数敏感性热图**：1D/2D 参数-指标响应分析

### 3. 参数搜索 (Parameter Search)

- **网格搜索 (Grid Search)**：低维离散空间的穷尽扫描
- **随机搜索 (Random Search)**：中等维度的均匀采样，带离散参数支持
- **贝叶斯优化接口**：预留 `BayesianSearchInterface`，可无缝接入 Optuna / Ax

### 4. 持久化存储

默认采用本地 **JSONL + CSV** 混合存储：
- `experiments.jsonl`：实验元数据
- `runs.jsonl`：运行元数据
- `params.jsonl`：参数快照
- `metrics.jsonl`：指标快照
- `artifacts.jsonl`：产物索引

无需数据库，文件可版本控制、可手工审计。

## 项目结构

```
12_quant_research_workbench/
├── core/
│   ├── __init__.py
│   ├── models.py                  # Experiment, Run, ParamSnapshot, MetricSnapshot, Artifact
│   ├── experiment_tracker.py      # 文件式实验追踪核心
│   ├── experiment_compare.py      # 结果对比 + Bootstrap 检验
│   ├── parameter_search.py        # Grid / Random / Bayesian接口
│   └── engine.py                  # 统一工作台引擎
├── tests/
│   ├── test_experiment_tracker.py
│   ├── test_experiment_compare.py
│   ├── test_parameter_search.py
│   └── test_engine.py
├── docs/
│   └── THEORY.md                  # 理论设计文档
├── examples/
│   └── demo_workbench.py          # 合成数据演示
└── README.md
```

## 理论基础

详见 `docs/THEORY.md`，涵盖：
- 实验元数据模型设计
- 参数扁平化与 JSON 序列化规则
- Bootstrap 显著性检验方法
- 随机搜索在高维空间中的效率优势（Bergstra & Bengio, 2012）

## 测试与验证

项目包含 **16 个单元测试**，覆盖追踪、对比、搜索三大模块。

运行测试：
```bash
python -m pytest tests/ -v
```

## 使用示例

```python
from core.engine import WorkbenchEngine

engine = WorkbenchEngine(root_dir=".workbench")
exp_id = engine.create_experiment("my_strategy", "Test strategy variants")

# Grid search
def objective(params):
    lookback = params["lookback"]
    # ... run backtest ...
    return sharpe_ratio

results = engine.run_grid_search(
    experiment_id=exp_id,
    objective=objective,
    param_grid={"lookback": [5, 10, 20, 30]},
)

# Compare
comp = engine.compare_experiment(exp_id)
print(comp)

# Significance test
sig = engine.compare_significance(
    experiment_id=exp_id,
    metric_name="objective",
    param_name="lookback",
    value_a=10,
    value_b=30,
)
print(sig)
```

## 与 Trading 项目的集成远景

Trading 项目目前有 46 个子项目，各自管理自己的 `data/`, `scripts/`, `outputs/`, `logs/`。项目12可以统一这些分散的实验产出：

1. **在每个子项目的入口脚本中集成 `WorkbenchEngine`**
2. **自动生成实验对比报告**：汇总不同策略线的回测指标
3. **参数优化标准化**：所有子项目使用统一的参数搜索接口
4. **构建中心化的 `experiment_catalog.jsonl`**，支持跨项目的策略检索与对比

## 开发状态

- [x] 理论设计文档
- [x] 核心数据模型
- [x] 文件式实验追踪器（JSONL 持久化）
- [x] 参数扁平化与嵌套字典支持
- [x] 结果对比表生成
- [x] Bootstrap 显著性检验
- [x] 网格搜索 + 随机搜索
- [x] 贝叶斯优化预留接口
- [x] 统一工作台引擎
- [x] 单元测试（16 tests passed）
- [x] Demo 脚本

## 参考文献

- Bergstra, J., & Bengio, Y. (2012). "Random Search for Hyper-Parameter Optimization." *JMLR*.
- Mitchell, T.M. (1997). *Machine Learning*. McGraw-Hill.
- Claerbout, J.F., & Karrenbach, M. (1992). "Electronic Documents Give Reproducible Research a New Meaning."
