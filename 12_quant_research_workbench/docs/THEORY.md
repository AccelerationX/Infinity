# 量化研究工作台与策略实验平台 - 理论设计文档

## 1. 项目定位

本系统旨在为量化研究提供统一的**实验管理基础设施**，解决策略开发、参数优化、结果对比、研究文档沉淀的全流程可追溯性问题。它不是交易策略本身，而是支撑策略研究的**元平台 (Meta-Platform)**。

系统设计遵循以下核心原则：
1. **可复现性 (Reproducibility)**：任何实验都可以被完整复现
2. **可对比性 (Comparability)**：不同实验之间的差异可以被量化分析
3. **可扩展性 (Extensibility)**：支持多种策略类型、数据源、优化算法

## 2. 实验元数据模型

### 2.1 核心实体关系

```
Experiment
    └── Run 1
            ├── ParamSnapshot
            ├── MetricSnapshot
            └── Artifact
    └── Run 2
            ├── ParamSnapshot
            ├── MetricSnapshot
            └── Artifact
```

### 2.2 Experiment（实验）

一个实验对应一个研究主题或策略假设。属性包括：
- `experiment_id`: 全局唯一标识符（UUID）
- `name`: 实验名称
- `description`: 实验描述/假设
- `created_at`: 创建时间
- `tags`: 标签列表（便于检索和分类）

### 2.3 Run（单次运行）

一次运行对应一次具体的策略回测或训练。属性包括：
- `run_id`: 运行唯一标识符
- `experiment_id`: 所属实验
- `status`: 状态（PENDING / RUNNING / COMPLETED / FAILED）
- `start_time` / `end_time`: 起止时间
- `duration_ms`: 运行耗时

### 2.4 ParamSnapshot（参数快照）

记录运行时的所有超参数配置：
- `run_id`
- `param_name`
- `param_value`（支持 int, float, str, bool, list, dict）
- `param_type`: 参数类型标记

### 2.5 MetricSnapshot（指标快照）

记录运行产出的关键指标：
- `run_id`
- `metric_name`
- `metric_value`（float）
- `step`: 可选的时间步/epoch（支持序列指标）
- `timestamp`

### 2.6 Artifact（产物）

记录运行产生的文件：
- `run_id`
- `artifact_path`: 文件路径
- `artifact_type`: 类型（csv, png, model, report）
- `checksum`: 文件校验和（MD5/SHA256，确保完整性）
- `size_bytes`

## 3. 实验追踪接口设计

### 3.1 上下文管理器模式

```python
with tracker.start_run(experiment_id="exp_001") as run:
    run.log_params({"top_n": 30, "rebalance": 5})
    run.log_metrics({"cagr": 1.23, "mdd": -0.15})
    run.log_artifact("backtest_report.csv")
```

### 3.2 参数快照的序列化规则

为了支持复杂的参数类型（如嵌套 dict、list），采用 JSON 序列化后扁平化存储：

```python
{"model": {"lr": 0.01, "epochs": 100}}
# 扁平化后
{"model.lr": 0.01, "model.epochs": 100}
```

这样可以方便地进行参数-指标关联分析。

## 4. 结果对比与差异分析

### 4.1 实验对比表

将多个实验的运行结果并表，生成对比 DataFrame：

| run_id | param.top_n | param.rebalance | metric.cagr | metric.mdd |
|--------|-------------|-----------------|-------------|------------|
| run_1  | 30          | 5               | 1.23        | -0.15      |
| run_2  | 20          | 5               | 1.10        | -0.12      |

### 4.2 参数敏感性热图

对于网格搜索或随机搜索产生的多组实验，绘制参数-指标响应曲面或热图：
- X轴：参数 A 的取值
- Y轴：参数 B 的取值
- 颜色：目标指标（如 Sharpe Ratio）

### 4.3 统计显著性检验

比较两组实验的某指标差异是否具有统计显著性：
- **配对 t 检验**：当两组实验在其他条件相同、仅单一参数变化时
- **Bootstrap 置信区间**：从实验结果的分布中重采样，估计差异的置信区间

Bootstrap 差异估计：
```
Δ* = metric_A* - metric_B*
CI = [quantile(Δ*, 0.025), quantile(Δ*, 0.975)]
```
若 CI 不包含 0，则差异在 95% 水平上显著。

## 5. 参数搜索模块

### 5.1 网格搜索 (Grid Search)

对离散化的参数空间进行穷尽搜索：
```python
param_grid = {"top_n": [10, 20, 30], "rebalance": [3, 5, 10]}
```
总运行数 = $3 \times 3 = 9$。

### 5.2 随机搜索 (Random Search)

在参数空间中随机均匀采样：
```python
bounds = {"top_n": (5, 50), "rebalance": (2, 20)}
```
优点：在高维空间中，随机搜索往往比网格搜索更高效（Bergstra & Bengio, 2012）。

### 5.3 贝叶斯优化接口（预留）

预留与 Optuna / Ax / scikit-optimize 的接口：
- 定义目标函数 `objective(params) -> metric`
- 由外部贝叶斯优化库推荐下一组参数
- 工作台负责记录每一轮推荐的参数和对应的指标

不直接实现贝叶斯优化算法本身（这属于专门的优化库范畴），而是提供统一的**参数-指标记录接口**，使任何优化库都可以与本工作台无缝集成。

## 6. 数据持久化

### 6.1 存储后端

为了降低依赖复杂度，默认采用**本地 JSONL + CSV** 的混合存储：
- `experiments.jsonl`: 实验元数据
- `runs.jsonl`: 运行元数据
- `params.jsonl`: 参数快照
- `metrics.jsonl`: 指标快照
- `artifacts.csv`: 产物索引

对于生产环境，可扩展为 SQLite / PostgreSQL 后端。

### 6.2 查询接口

支持基于实验名称、标签、参数范围、指标阈值的条件查询：
```python
tracker.query(
    experiment_name="momentum_strategy",
    metric_filters={"cagr": (">=", 1.0)},
    param_filters={"top_n": ("in", [20, 30])},
)
```

## 7. 系统架构

```
12_quant_research_workbench/
├── core/
│   ├── __init__.py
│   ├── models.py              # Experiment, Run, ParamSnapshot, MetricSnapshot, Artifact
│   ├── experiment_tracker.py  # 实验追踪核心
│   ├── experiment_compare.py  # 结果对比与差异分析
│   ├── parameter_search.py    # 网格/随机搜索
│   └── engine.py              # 主引擎
├── tests/
├── docs/
│   └── THEORY.md
└── examples/
    └── demo_workbench.py
```

## 8. 设计原则

1. **最小依赖**：仅依赖 pandas、numpy，不强制要求数据库或外部服务
2. **向后兼容**：JSONL 格式易于版本控制和手工审计
3. **模块化**：tracker / compare / search 三个模块可独立使用
4. **零侵入**：策略代码只需在运行前后调用 `start_run` 和 `log_*` 接口

## 9. 参考文献

- Bergstra, J., & Bengio, Y. (2012). "Random Search for Hyper-Parameter Optimization." *JMLR*.
- Mitchell, T.M. (1997). *Machine Learning*. McGraw-Hill. (实验设计基础)
- Claerbout, J.F., & Karrenbach, M. (1992). "Electronic Documents Give Reproducible Research a New Meaning."

---
*本文档为项目12的第一性原理基础，所有代码实现必须与此文档保持一致。*
