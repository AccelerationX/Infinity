# 15 Agent Lifelong Learning —— 持续学习与技能蒸馏系统

> 让 Agent 在与用户的长期交互中自动提炼技能、积累经验、避免重复犯错。

## 1. 项目定位

本项目研究如何让 Agent 具备**终身学习能力（Lifelong Learning）**。核心不是预训练更大的模型，而是在交互过程中：
- **发现重复模式**：从对话历史中识别可复用的操作模式
- **抽象为技能模板**：把具体案例泛化为带参数的技能定义
- **调用与迭代**：后续直接调用技能，并根据用户反馈持续优化

这与传统的 Prompt Engineering 和 RAG 不同——它关注的是**从经验中生成新的可执行知识**。

## 2. 核心问题

1. 技能发现：如何从海量对话中自动识别"值得记录的模式"？
2. 技能泛化：如何把"查茅台 K 线"抽象为"查 {stock} 的 K 线"？
3. 技能冲突与遗忘：当技能库膨胀时，如何避免调用错误的技能？

## 3. 技术架构

```
对话经验 / MCP 审计日志
    ↓
ExperienceBuffer (SQLite)
    ↓
PatternMiner (Sentence-BERT + K-Means)
    ↓
SkillDistiller (LLM / Fallback)
    ↓
SkillLibrary (SQLite + Embedding 索引)
    ↓
LearningAgent
    ├── Skill Retrieval (语义匹配，阈值 0.45)
    ├── Conflict Resolution (成功率加权排序)
    ├── Failure Replay (注入历史注意事项)
    ├── Param Filler (从请求中提取参数值)
    └── Fallback (无匹配时回退到通用推理)
```

## 4. 目录结构

```
15_agent_lifelong_learning/
├── src/
│   ├── schemas/models.py          # Experience / Skill / Failure 数据模型
│   ├── storage/sqlite_store.py    # SQLite 持久化层
│   ├── pattern_miner.py           # K-Means 模式挖掘
│   ├── skill_distiller.py         # LLM / 规则技能蒸馏
│   ├── skill_library.py           # 技能库存储与语义检索
│   ├── failure_replay.py          # 失败记录与注意事项注入
│   ├── conflict_resolver.py       # 多技能冲突消解
│   ├── param_filler.py            # 技能参数自动填充
│   └── learning_agent.py          # 主 Agent 入口
├── tests/
├── demos/
│   ├── demo_lifelong_learning.py  # 完整终身学习闭环演示
│   └── demo_mcp_bridge.py         # 与 14 MCP Hub 的数据联动演示
├── benchmark/
│   └── run_benchmark.py           # 5 维度学习效果评估
├── docs/
│   └── integration_with_mcp_hub.md
└── README.md
```

## 5. 已验证的端到端 Demo

### Demo 1：终身学习闭环
```bash
python demos/demo_lifelong_learning.py
```
效果：注入股票/天气/计算三类经验 → K-Means 聚类挖掘模式 → 蒸馏成技能 → LearningAgent 直接调用技能响应新请求 → 模拟失败注入注意事项 → 多次反馈后成功率收敛。

### Demo 2：与 MCP Hub 的数据联动
```bash
python demos/demo_mcp_bridge.py
```
效果：模拟从 **14_mcp_tool_hub** 导入 7 条审计日志，自动学习出 `query_weather` 和 `calculate_expression` 技能，然后对新请求直接生成可执行 action。

### Demo 3：批量 Benchmark
```bash
python benchmark/run_benchmark.py
```
效果：运行 5 个维度的标准化测试，当前结果：
- **技能发现准确率: 100%**
- **技能路由准确率: 100%**
- **参数填充准确率: 100%**
- **失败回放有效性: PASS**
- **成功率追踪正确性: PASS**

**Total Score: 5.0 / 5.0**

## 6. 当前能力与边界

| 能力 | 状态 | 说明 |
|------|------|------|
| SQLite 经验池 | ✅ | 支持 CRUD、时间衰减、事务 |
| SQLite 技能库 | ✅ | 版本管理、成功率追踪、embedding 检索 |
| 模式挖掘 | ✅ | Sentence-BERT + K-Means，自动发现重复模式 |
| 技能蒸馏 (Fallback) | ✅ | 基于规则的后备蒸馏，覆盖股票/天气/计算等常见类型 |
| 技能蒸馏 (LLM) | 🟡 | 接口已预留，接入 OpenAI-compatible LLM 即可 |
| 语义检索 | ✅ | Trigger patterns 的平均 embedding 做匹配 |
| 冲突消解 | ✅ | 综合语义相似度 × 历史成功率 × 使用频次 |
| 失败回放 | ✅ | 注入历史注意事项，避免重复犯错 |
| 参数填充 (Regex) | ✅ | 支持股票代码、城市名、表达式、文件路径等 |
| 参数填充 (LLM) | 🟡 | 接口已预留 |

## 7. 与 14 MCP Tool Hub 的联动设计

14 的 `audit.jsonl` 是 15 经验池的最优质输入。设计文档见：

```bash
cat docs/integration_with_mcp_hub.md
```

核心流程：
1. **14 执行工具调用** → 写入审计日志
2. **15 的 ExperienceImporter** 读取日志，生成 `ExperienceRecord`
3. **PatternMiner** 周期性聚类，发现重复意图-动作模式
4. **SkillDistiller** 抽象为 `SkillTemplate`
5. **LearningAgent** 下次收到类似请求时直接调用技能，**绕过复杂的语义匹配和 DAG 规划**

## 8. 依赖

```bash
pip install sentence-transformers scikit-learn pyyaml pydantic openai numpy pytest
```

## 9. 快速开始

```bash
cd 15_agent_lifelong_learning
python demos/demo_lifelong_learning.py
```
