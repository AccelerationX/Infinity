# 15 Agent 终身学习系统 —— 面试完全指南

---

## 一、项目一句话定位

这是一个**支持技能自动发现、蒸馏、参数填充和失败回放注入的 Agent 终身学习系统**，能够让 Agent 从交互经验中持续积累可复用技能，并在遇到新任务时优先调用已学技能。

---

## 二、核心技术栈

- **Python 3.10** + **Pydantic**
- **SQLite**（经验、技能、失败案例持久化）
- **scikit-learn (K-Means)** + **sentence-transformers (SBERT)**（模式挖掘聚类）
- **YAML / JSON**（技能结构化表示）
- **Pytest**（10/10 测试通过）
- **MCP Tool Hub (14号项目)**（外部工具调用集成）

---

## 三、核心原理

当前大多数 LLM Agent 都是"一次性"的——每次对话都从头推理，不会记住过去的成功经验。本项目试图解决的核心问题是：**如何让 Agent 像人类一样，从重复的任务中提炼出通用技能，并在未来快速复用？**

### 核心循环（Lifelong Learning Loop）

1. **经验收集（Experience Collection）**：
   每次 Agent 完成任务（无论是通过 LLM 推理还是调用工具），都会把 `(request, context, action, outcome)` 存入经验池。

2. **模式挖掘（Pattern Mining）**：
   当经验池积累到一定数量时，`PatternMiner` 会对所有经验的用户请求做 **SBERT 语义嵌入 + K-Means 聚类**，自动发现重复出现的任务模式。例如"查天气"相关的请求会被聚到一类。

3. **技能蒸馏（Skill Distillation）**：
   `SkillDistiller` 对每个聚类中的经验进行抽象，生成一个**结构化技能模板**。技能包含：
   - `skill_id`：唯一标识
   - `description`：技能的语义描述
   - `template`：可参数化的执行模板（如 `"query_weather(city={city})"`）
   - `required_params`：需要的参数列表

4. **智能路由（LearningAgent）**：
   当新请求到来时，`LearningAgent` 首先在技能库中做语义检索。如果匹配到已有技能，就用 `ParamFiller` 填充参数并直接执行；如果没有匹配，则 fallback 到 LLM 推理。

5. **失败回放（Failure Replay）**：
   `FailureReplay` 模块记录每次执行失败的原因。当 Agent 准备执行某个技能时，会自动注入与该技能相关的历史失败 caution，提醒 Agent 避免重蹈覆辙。

6. **冲突解决（Conflict Resolution）**：
   如果新挖掘出的技能和已有技能高度相似，`SkillLibrary` 会比较两者的历史成功率，保留更优版本，并做版本管理。

---

## 四、实现细节

### 架构设计
```
src/
  ├── learning_agent.py      # 智能路由：技能匹配 → 参数填充 → 执行
  ├── pattern_miner.py       # SBERT + K-Means 模式挖掘
  ├── skill_distiller.py     # 从聚类经验中蒸馏技能
  ├── param_filler.py        # 参数自动填充（LLM + Regex）
  ├── failure_replay.py      # 失败案例库与 caution 注入
  ├── skill_library.py       # 技能持久化、检索、版本管理
  ├── sqlite_store.py        # SQLite CRUD
  └── experience.py          # 经验数据结构
```

### 关键模块
- `PatternMiner`：
  - 用 `all-MiniLM-L6-v2` 对用户请求做 embedding；
  - 用 K-Means 聚类（K 通过肘部法则或 Silhouette Score 自动选择）；
  - 输出每个 cluster 的中心描述和代表性请求列表。
- `SkillDistiller`：
  - 对每个 cluster 中的经验，提取公共动作模式；
  - 用 LLM 或规则启发式生成参数化模板；
  - Fallback：如果 LLM 不可用，则用简单的字符串替换发现可变部分。
- `LearningAgent.act(request)`：
  1. `skill_library.search(request)` 做语义 top-k 检索；
  2. 如果最佳匹配分数超过阈值，进入技能执行路径；
  3. `param_filler.fill(skill, request)` 解析参数；
  4. `failure_replay.get_cautions(skill_id)` 获取失败提醒；
  5. 返回 `{action, cautions}` 供上层执行。

### 难点与解决
- **聚类数 K 的选择**：K 太小会把不同任务混在一起，K 太大会过度细分。我们通过**Silhouette Score + 肘部法则**做自动选择，并允许用户手动覆盖。
- **技能模板泛化能力不足**：初期模板太具体（如 `"query_weather(city='北京')"`），无法应用到其他城市。我们通过 LLM 抽象和规则替换，把常量提取为参数，生成 `"query_weather(city={city})"`。
- **失败案例的 caution 注入过多**：如果某个技能失败很多次，每次执行都注入大量 caution 会拖慢速度。我们做了**去重和摘要**：只保留最常见的 3 条失败原因，并定期让 LLM 对 caution 做压缩总结。

---

## 五、对应岗位

- AI 算法工程师（Agent / LLM 方向）
- 大模型应用研究员
- AI 系统架构师
- 智能体/机器人学习研究员

---

## 六、简历描述建议

> **独立设计并实现了 Agent 终身学习系统**，支持从交互经验中自动挖掘任务模式（SBERT+K-Means）、蒸馏参数化技能模板、并通过语义检索实现新请求到已有技能的智能路由。系统内置失败回放注入和技能冲突解决机制，确保 Agent 持续进化而不退化。端到端 benchmark 在 12 条经验上成功挖掘出 3 个技能，新请求路由准确率 100%，参数填充准确率 100%，综合评分 5.0/5.0；全部 10 个单元测试通过。该系统与 14 号 MCP Tool Hub 无缝集成，可自动将外部工具调用日志转化为可复用技能。

---

## 七、高频面试问题与回答

### Q1：终身学习和简单的 Prompt 缓存（Cache）有什么区别？
**A**：Prompt 缓存只是记住了"某个输入对应的输出"，它是**纯记忆**，没有抽象和泛化能力。而终身学习的核心是**技能蒸馏**：Agent 会从大量相似经验中抽象出一个通用的参数化模板。比如它不会只记住"查北京天气 → 晴天"，而是会学会"查 {city} 天气 → 调用 query_weather(city={city})"。这个技能可以应用到任何城市，而缓存做不到。

### Q2：如果用户请求和已有技能相似但不完全一样，你们怎么处理？
**A**：`LearningAgent` 会用语义相似度模型（SBERT）在技能库中做 top-k 搜索。如果最佳匹配的相似度分数超过阈值（比如 0.8），就会尝试用该技能；然后 `ParamFiller` 会从用户请求中提取具体参数填入模板。如果相似度低于阈值，则 fallback 到 LLM 推理，并把这次交互作为新经验存入池子，等待未来的模式挖掘。

### Q3：K-Means 聚类会不会把语义相似但任务类型不同的请求混在一起？
**A**：有可能。SBERT 嵌入主要捕获语义相似性，但语义相似不代表任务相同（比如"今天天气怎么样"和"今天股市怎么样"语义结构相似，但任务完全不同）。为了缓解这个问题，我们做了两件事：
1. **加入动作特征**：聚类时不仅看用户请求文本，还结合 Agent 最终执行的动作做联合 embedding；
2. **人工校验 + 合并/拆分**：`SkillDistiller` 生成技能后，如果发现一个 cluster 内动作差异太大，会自动提示用户或尝试子聚类。

### Q4：失败回放是怎么工作的？会不会导致 Agent 过度保守？
**A**：失败回放的工作原理是：当 Agent 准备执行某个技能时，去 `FailureReplay` 库中查询该技能的历史失败记录，把失败原因作为 caution 注入到执行上下文中。例如："注意：之前执行此技能时曾因 city 参数为空而失败。"

为了避免过度保守，我们做了两个限制：
1. **只保留高频失败**：只有出现 2 次以上的相同失败原因才会被注入；
2. **caution 数量上限**：单次执行最多注入 3 条 caution，避免上下文爆炸。

### Q5：这个系统和 14 号 MCP Tool Hub 是怎么联动的？
**A**：14 号的 Tool Hub 负责"外部工具的发现和执行"，15 号的终身学习系统负责"经验的积累和技能的复用"。两者的集成逻辑是：
1. Tool Hub 执行完用户的自然语言请求后，会把 `(intent, plan, result)` 的完整轨迹写入审计日志；
2. 15 号的 `demo_mcp_bridge.py` 会读取这些日志，把它们转化为 Experience 对象存入 SQLite；
3. PatternMiner 对这些 Experience 做聚类，自动生成像 `query_weather`、`calculate_expression` 这样的技能；
4. 下次用户再发类似的意图，LearningAgent 就可以直接调用技能，无需再经过 LLM 规划和 Tool Hub 的复杂匹配流程。

这个联动展示了"从工具调用到技能沉淀"的完整闭环。
