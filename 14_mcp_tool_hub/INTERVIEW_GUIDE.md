# 14 MCP Tool Hub（生产级工具路由中枢）—— 面试完全指南

---

## 一、项目一句话定位

这是一个**基于官方 MCP SDK 构建的生产级多工具路由中枢**，支持动态注册 MCP 服务器、语义意图匹配（BGE）、DAG 任务规划、参数自动填充和安全审计，实现了 100% 工具召回与参数准确率。

---

## 二、核心技术栈

- **Python 3.10** + **Asyncio**
- **MCP SDK (official, `mcp>=1.27.0`)**
- **sentence-transformers (BGE-small-zh-v1.5)**（语义匹配）
- **NetworkX**（DAG 规划与拓扑执行）
- **Pydantic**（参数校验与结构化输出）
- **Pytest**（12/13 测试通过，1 跳过）

---

## 三、核心原理

大模型 Agent 的能力边界很大程度上取决于它能调用多少外部工具。MCP（Model Context Protocol）是 Anthropic 推出的开放协议，旨在统一 LLM 与外部工具/数据源/环境的通信方式。本项目的目标是在 MCP 之上构建一个**生产可用的工具路由中枢**，解决以下问题：

### 1. 工具发现与语义匹配
用户用自然语言描述意图，系统需要从几十甚至上百个工具中找到最合适的一个。我们使用 **BGE 语义嵌入 + Cosine Similarity** 做意图-工具匹配：
- 离线阶段：对每个工具的 `name` 和 `description` 生成 embedding 并存入向量库；
- 在线阶段：对用户意图做 embedding，检索 top-k 最相似的工具。

### 2. DAG 任务规划
当用户请求涉及多个步骤时（如"先查天气，再把这个结果写到文件里"），系统会生成一个 **DAG（有向无环图）执行计划**。每个节点是一个工具调用，边表示数据依赖关系。`DAGPlanner` 负责构建图，`ExecutionEngine` 按拓扑序异步执行。

### 3. 参数自动填充（ArgumentResolver）
工具通常有多个必填参数，但用户不会显式提供所有参数。我们设计了两级参数解析：
- **LLM 解析**：让 LLM 从用户意图中提取参数值；
- **正则 Fallback**：LLM 失败时，用预定义的正则模板提取常见实体（路径、表达式、股票代码、城市名等）。

### 4. 安全审计（SecurityManager）
所有执行计划都会经过安全审查：
- **敏感操作拦截**：如 `rm -rf`、`del C:\Windows` 等高危命令直接拒绝；
- **用户确认**：对于写文件、网络请求、Shell 执行等操作，要求用户显式确认；
- **审计日志**：完整记录 intent → plan → execution 的全链路，便于事后溯源。

---

## 四、实现细节

### 架构设计
```
src/
  ├── hub.py                 # ToolHub：注册、匹配、规划、执行
  ├── matcher.py             # SemanticMatcher (BGE)
  ├── planner.py             # DAGPlanner
  ├── argument_resolver.py   # 参数解析（LLM + Regex Fallback）
  ├── security.py            # SecurityManager
  ├── audit_logger.py        # JSONL 审计日志
  └── clients/               # MCP 客户端封装
       ├── filesystem_client.py
       ├── calculator_client.py
       ├── browser_client.py
       └── shell_client.py
```

### 关键模块
- `ToolHub`：核心路由器。`register_client()` 注册 MCP 客户端；`match_and_plan()` 做语义匹配和 DAG 规划；`execute_plan()` 安全执行并记录日志。
- `SemanticMatcher`：基于 `sentence-transformers` 的 `BAAI/bge-small-zh-v1.5`，支持中英双语意图匹配。在 benchmark 上实现了 **100% 工具召回率**。
- `ArgumentResolver`：LLM 解析失败时，自动降级到正则模板。例如股票代码用 `[0-9]{6}`，数学表达式用简单算术匹配器。
- `AuditLogger`：以 JSON Lines 格式持久化，每条记录包含 `timestamp`、`session_id`、`intent`、`plan`、`execution_result`。

### 难点与解决
- **MCP SDK 官方 API 不稳定**：`mcp>=1.27.0` 的某些接口和文档不完全一致。我们通过仔细阅读源码和运行官方 examples，自行封装了适配层。
- **跨工具参数传递复杂**：DAG 中后序工具需要用到前序工具的输出。我们用**变量占位符语法**（如 `$step_1.result`）在规划阶段表示依赖，执行阶段由 `ExecutionEngine` 动态替换为实际值。
- **BGE 模型首次加载慢**：`sentence-transformers` 在冷启动时需要下载模型。我们在 Docker 镜像构建阶段预下载并缓存模型，避免运行时延迟。

---

## 五、对应岗位

- AI 应用工程师（Agent / Tool Calling 方向）
- LLM 应用架构师
- AI Infra 工程师
- 后端开发工程师（Python）

---

## 六、简历描述建议

> **设计并实现了一个基于官方 MCP SDK 的生产级多工具路由中枢（Tool Hub）**，支持动态注册 MCP 服务器、BGE 语义意图匹配、DAG 任务规划、LLM+正则双模参数解析和多层安全审计。在端到端 benchmark 中实现 100% 工具召回率、100% 参数正确率和 100% 执行成功率；安全层成功拦截所有高危操作（如删除系统目录），12/13 个单元测试通过。该系统作为底层基础设施，为后续 15 号项目的终身学习 Agent 提供了外部工具调用能力。

---

## 七、高频面试问题与回答

### Q1：MCP 和传统的 Function Calling 有什么区别？
**A**：传统 Function Calling 是模型厂商自己定义的调用格式（比如 OpenAI 的 `tools` 字段），不同厂商之间互不兼容。而 **MCP 是一个开放的协议层**，它定义了 LLM 客户端与外部服务器之间的标准通信方式。只要一个工具实现了 MCP 协议，任何支持 MCP 的模型或平台都可以调用它，不需要为每个模型单独适配。

### Q2：DAG 规划是怎么做的？能举个例子吗？
**A**：当用户说"查一下北京的天气，然后把结果写进 report.txt"，DAGPlanner 的输出大致是：
```
Step 1: query_weather(city="北京")
Step 2: write_file(path="report.txt", content="$step_1.result")
```
`$step_1.result` 是一个占位符。`ExecutionEngine` 会先执行 Step 1，拿到天气结果后，把占位符替换为真实值，再执行 Step 2。这种方式可以自然地支持任意复杂的多步骤任务。

### Q3：如果 LLM 参数解析失败了，正则 Fallback 能保证准确率吗？
**A**：不能保证 100%，但对于常见的结构化实体（如文件路径、数学表达式、城市名、股票代码），正则的准确率非常高。在我们的 benchmark 中，LLM+正则双模方案的参数正确率达到了 100%。如果未来遇到更复杂的参数类型（如嵌套 JSON），可以进一步扩展正则模板库，或者引入更小的专用 NER 模型。

### Q4：安全层是怎么设计的？怎么防止 Agent 执行危险命令？
**A**：安全层分为三道防线：
1. **静态拦截**：基于关键词和模式匹配（如 `rm -rf /`、`del C:\Windows`、`format C:`）直接拒绝；
2. **动态确认**：对于写文件、网络请求、Shell 执行等操作，强制要求用户显式确认（`user_confirmed=True`）；
3. **审计日志**：所有操作完整记录到 JSONL，包括输入意图、生成的计划、执行结果，便于事后审查和追溯。

### Q5：这个 Tool Hub 和 15 号终身学习 Agent 是怎么集成的？
**A**：15 号项目的 LearningAgent 需要调用外部工具来完成任务。我们没有让 LearningAgent 直接和每个工具打交道，而是通过 14 号的 Tool Hub 作为统一代理。LearningAgent 只需要生成自然语言意图（如"计算 23+45"），Tool Hub 负责匹配工具、解析参数、执行调用并返回结果。这样 LearningAgent 可以专注于学习和推理，而不需要关心底层工具的接入细节。我们甚至写了一个 `demo_mcp_bridge.py`，演示如何把 Tool Hub 的执行日志导入到 15 号项目的终身学习系统中，自动生成可复用的技能。
