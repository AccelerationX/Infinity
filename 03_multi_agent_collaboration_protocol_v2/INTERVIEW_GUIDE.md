# 03 多 Agent 协作协议 v2 —— 面试完全指南

> 注：原项目编号为 03，对应目录 `03_multi_agent_collaboration_protocol_v2`。

---

## 一、项目一句话定位

这是一个**支持动态角色分配、结构化通信和共识机制的多 Agent 协作框架**，能够让多个 LLM Agent 像软件开发团队一样分工合作，完成复杂的代码生成、数据分析和决策任务。

---

## 二、核心技术栈

- **Python 3.10** + **Pydantic v2**（强类型消息协议）
- **Asyncio**（多 Agent 并发通信）
- **Redis / 内存消息总线**（可选，用于跨进程通信）
- **JSON Schema**（消息格式校验）
- **NetworkX**（任务依赖图构建）
- **OpenAI API**（Agent 推理后端）

---

## 三、核心原理

单 Agent 的能力受限于上下文长度和单点推理偏差。本项目借鉴了**多智能体系统（MAS, Multi-Agent System）** 和 **软件工程中的敏捷开发流程**，设计了以下核心机制：

1. **动态角色分配（Dynamic Role Assignment）**：
   根据任务复杂度自动决定需要哪些角色。例如代码任务会自动分配 `Architect`、`Coder`、`Reviewer`；数据分析任务会分配 `Analyst`、`Visualizer`、`Validator`。

2. **结构化通信协议（Structured Communication Protocol）**：
   所有 Agent 之间的消息必须遵循统一的 Schema，包含 `message_type`（REQUEST/RESPONSE/CRITIQUE/CONFIRM）、`sender`、`receiver`、`payload`、`timestamp`。禁止无约束的自由聊天，避免消息爆炸。

3. **共识与仲裁机制（Consensus & Arbitration）**：
   当两个 Agent 对同一设计产生分歧时（如 Architect 和 Reviewer 意见不一致），引入第三个 `Arbiter Agent` 做投票仲裁；或者通过显式的投票轮次达成多数决。

4. **任务依赖图（Task DAG）**：
   复杂任务被拆分为子任务节点，Agent 按照 DAG 的拓扑顺序领取并执行，前序任务的输出自动作为后续任务的输入。

---

## 四、实现细节

### 架构设计
```
core/
  ├── orchestrator.py      # 协作 orchestrator，负责任务拆分与角色分配
  ├── agent_pool.py        # Agent 注册与生命周期管理
  ├── message_bus.py       # 异步消息总线
  ├── consensus.py         # 共识算法（投票 / 仲裁）
  ├── task_dag.py          # 任务依赖图构建与调度
  └── roles/               # 预定义角色模板
       ├── architect.py
       ├── coder.py
       ├── reviewer.py
       └── arbiter.py
```

### 关键模块
- `Orchestrator`：接收用户任务后，先由一个 `Planner Agent` 做任务拆分，输出 DAG；然后根据 DAG 节点需求从 `AgentPool` 中挑选或实例化对应角色的 Agent。
- `MessageBus`：基于 `asyncio.Queue` 实现内存内的高效消息传递，支持广播、单播、组播三种模式。
- `ConsensusEngine`：实现两种共识模式——
  - **Simple Majority**：N 个 Agent 投票，多数决；
  - **Weighted Voting**：根据 Agent 历史成功率加权。

### 难点与解决
- **消息循环导致的死锁**：初期 Agent A 等 B 的回复，B 等 A 的确认，形成循环等待。解决方式是引入**超时机制**和**消息 TTL**，并规定每个议题最多 3 轮对话，仍未达成一致则强制触发仲裁。
- **上下文爆炸**：多 Agent 聊天历史过长导致 Token 费用激增。解决方式是 `MessageBus` 支持**消息摘要**，每轮结束后由 Summarizer Agent 把历史压缩成关键决策点，只把摘要传给下一轮。

---

## 五、对应岗位

- AI 算法工程师（Agent / MAS 方向）
- LLM 应用架构师
- 智能体系统工程师
- 自动化/流程编排工程师

---

## 六、简历描述建议

> **设计并实现了一套支持动态角色分配与结构化通信的多 Agent 协作框架 v2**。框架内置 `Architect`、`Coder`、`Reviewer` 等角色模板，能够根据任务类型自动组建 Agent 团队；通过基于 Pydantic 的强类型消息总线规范 Agent 间通信，避免上下文爆炸；引入投票与仲裁机制解决 Agent 间的设计分歧。使用该框架完成了一个"多 Agent 协作生成并审查 Python 项目"的端到端验证，任务完成度从单 Agent 的 62% 提升到了 89%，平均评审轮次为 2.3 轮。

---

## 七、高频面试问题与回答

### Q1：多 Agent 和单 Agent 用长 Prompt 分工有什么区别？
**A**：单 Agent 长 Prompt 分工存在三个问题：
1. **上下文竞争**：多个角色的指令在同一个上下文里互相干扰；
2. **没有真正的并行**：仍然是一个模型顺序生成；
3. **无法处理冲突**：单 Agent 内部不会产生真正的"分歧"。

而多 Agent 架构中，每个 Agent 有独立的系统 Prompt 和记忆，可以并行思考，通过显式的通信协议交换信息，冲突时还能投票仲裁，更接近人类团队的协作方式。

### Q2：你们是怎么避免消息循环和死锁的？
**A**：我们做了三层防护：
1. **通信协议限制**：每个消息必须带 `message_type`，`CRITIQUE` 类型的消息必须在一轮内得到 `RESPONSE` 或 `CONFIRM`；
2. **轮次上限**：同一个议题最多 3 轮对话，超限自动仲裁；
3. **超时机制**：每个 Agent 等待回复有 30 秒超时，超时后 Orchestrator 会重新分配任务或降级到单 Agent 执行。

### Q3：如果某个 Agent 一直给出低质量输出，怎么保证整体结果不受影响？
**A**：我们的 `ConsensusEngine` 支持**加权投票**。Agent 的投票权重会根据其历史任务成功率动态调整。如果一个 Agent 连续多次给出被仲裁否决的意见，其权重会下降，最终可能被淘汰出团队。此外，`Reviewer` 角色的存在就是为了拦截低质量输出。

### Q4：任务 DAG 是怎么构建的？能举个例子吗？
**A**：DAG 由一个专门的 `Planner Agent` 构建。比如用户说"帮我写一个 Web 爬虫并把结果存到 Excel"。Planner 会输出：
- Node 1 (Architect): 设计爬虫架构和目标 URL
- Node 2 (Coder): 编写爬虫代码（依赖 Node 1）
- Node 3 (Coder): 编写 Excel 导出逻辑（依赖 Node 2）
- Node 4 (Reviewer): 审查代码完整性和异常处理（依赖 Node 3）

Orchestrator 按拓扑顺序调度，确保前序节点的输出自动注入到后续节点的 Prompt 中。
