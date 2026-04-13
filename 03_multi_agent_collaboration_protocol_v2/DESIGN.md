# MACP v2 - 设计文档

> Multi-Agent Collaboration Protocol v2
> 通用多Agent协作框架

---

## 1. 项目背景与动机

### 1.1 为什么需要MACP？

**传统单Agent的局限：**
- 一个Agent难以同时精通多个领域（编程+设计+测试+运维）
- 复杂任务需要多步骤、多角色协作
- 单Agent上下文窗口有限，难以处理大型项目

**现有方案的不足：**
- AutoGen、MetaGPT等框架过于重量级
- 多数框架专注于单一领域（如仅软件开发）
- 缺乏通用的、可扩展的协作协议

### 1.2 MACP的设计目标

1. **通用性**：不限于软件开发，支持金融、科研、写作等多个领域
2. **可扩展**：通过"领域模板"快速适配新场景
3. **可观测**：提供完整的监控和干预机制
4. **轻量级**：核心代码简洁，易于理解和定制

---

## 2. 核心架构设计

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                      User Interface                         │
│                   (Web Dashboard / CLI)                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                  CollaborationFramework                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Template  │  │  Scheduler  │  │   Agent Registry    │ │
│  │   System    │  │   (DAG)     │  │                     │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
│         │                │                     │            │
│  ┌──────▼──────┐  ┌──────▼──────┐  ┌──────────▼──────────┐ │
│  │ Role Def    │  │ Task Queue  │  │   LLM Agents        │ │
│  │ Workflow    │  │ Executor    │  │   (Kimi/DeepSeek)   │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                    Workspace / Storage                      │
│              (Job persistence / Project files)              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 核心组件详解

#### 2.2.1 领域模板系统 (Domain Template)

**核心思想**：将"领域知识"从"协作引擎"中分离

```python
class DomainTemplate(ABC):
    @abstractmethod
    def get_roles(self) -> List[AgentRole]:
        """定义该领域需要的角色"""
        pass
    
    @abstractmethod
    def decompose_task(self, input_text: str) -> List[Dict]:
        """将用户输入分解为DAG任务"""
        pass
```

**软件研发模板示例：**
- 7个角色：PM、架构师、TechLead、前端、后端、QA、DevOps
- DAG流程：需求分析 → 架构设计 → 任务规划 → (前端|后端|QA|DevOps并行) → 集成验证

**为什么这样设计？**
- 新增领域只需实现Template接口
- 核心引擎完全通用，不耦合任何业务逻辑
- 方便团队协作分工（业务专家定义Template，工程师优化引擎）

#### 2.2.2 DAG调度器 (Scheduler)

**核心思想**：基于依赖关系的动态调度

**执行流程：**
```
1. 解析任务依赖 → 构建DAG
2. 找出就绪任务（所有依赖已完成）
3. 分配可用Agent执行
4. 重复2-3直到全部完成
```

**关键设计决策：**
- **线程池执行**：使用ThreadPoolExecutor控制并发度
- **状态机管理**：每个任务有明确状态(PENDING→RUNNING→COMPLETED/FAILED)
- **容错机制**：任务失败不中断整个流程，记录到Job状态

**代码核心逻辑：**
```python
def _execute_job(self, job: Job):
    while len(completed) < len(job.tasks):
        # 1. 找出就绪任务
        ready = [t for t in job.tasks if all_deps_completed(t)]
        
        # 2. 限制并行度
        for task in ready[:max_workers]:
            self._execute_task(job, task)
        
        # 3. 保存状态（可恢复）
        self.workspace.save_job(job)
```

#### 2.2.3 Agent系统

**三层架构：**

```
┌─────────────────────────────────────┐
│         AgentRegistry               │  ← 管理所有Agent生命周期
│    (注册/发现/状态跟踪)              │
└──────────────────┬──────────────────┘
                   │
    ┌──────────────┼──────────────┐
    │              │              │
┌───▼───┐    ┌────▼────┐   ┌─────▼────┐
│  PM   │    │Architect│   │ Backend  │  ← 不同角色的Agent实例
│ Agent │    │  Agent  │   │  Agent   │
└───┬───┘    └────┬────┘   └─────┬────┘
    │             │              │
    └─────────────┼──────────────┘
                  │
          ┌───────▼────────┐
          │   LLMAgent     │  ← 统一的LLM调用层
          │ (Kimi/DeepSeek)│
          └────────────────┘
```

**Agent状态机：**
```
         ┌──────────┐
         │   idle   │ ← 空闲，可接受任务
         └────┬─────┘
              │ execute()
              ▼
         ┌──────────┐
         │   busy   │ ← 正在执行任务
         └────┬─────┘
              │ complete/error
              ▼
    ┌─────────┴─────────┐
    ▼                   ▼
┌────────┐        ┌──────────┐
│completed│        │  error   │
└────────┘        └────┬─────┘
                       │ retry
                       └──────→ idle
```

#### 2.2.4 LLM适配层

**设计原则**：屏蔽不同LLM API的差异

```python
class LLMAdapter:
    def __init__(self, model_name: str):
        # 统一初始化不同模型
        if model_name == "kimi":
            self.client = OpenAI(base_url="...", api_key="...")
            self.temperature = 1.0  # Kimi要求固定
        elif model_name == "deepseek":
            self.client = OpenAI(base_url="...", api_key="...")
            self.temperature = 0.7
    
    def chat(self, system_prompt, user_prompt, tools=None):
        # 统一接口，返回标准格式
        return {
            "content": "...",
            "tool_calls": [...],
            "usage": {...}
        }
```

**为什么用OpenAI SDK？**
- Kimi和DeepSeek都兼容OpenAI API格式
- 避免重复造轮子
- 社区生态成熟

---

## 3. 关键技术决策

### 3.1 为什么用DAG而非线性流程？

**对比：**

| 方案 | 优点 | 缺点 |
|------|------|------|
| 线性流程 | 简单 | 无法并行，效率低 |
| **DAG** | **支持并行执行，效率高** | **实现稍复杂** |
| 状态机 | 灵活 | 对于简单任务过于复杂 |

**实际场景：**
- 前端开发、后端开发、测试用例编写可以并行
- DAG能自动识别并利用这些并行机会

### 3.2 为什么分离Template和Engine？

**单一职责原则：**
- Engine只关心"如何调度"
- Template只关心"做什么"

**扩展性：**
- 新增领域只需写Template（几百行代码）
- 不需要修改Engine核心

**团队协作：**
- 业务专家可以独立设计Template
- 工程师专注优化Engine性能

### 3.3 为什么用线程而非异步(asyncio)？

**考虑因素：**

| 因素 | 线程 | asyncio |
|------|------|---------|
| 实现复杂度 | 简单 | 复杂 |
| LLM调用I/O | 阻塞等待 | 非阻塞 |
| 调试难度 | 容易 | 困难 |
| 性能 | 足够 | 更高 |

**决策：** 先用线程实现MVP，后续有需要再迁移到asyncio

### 3.4 监控数据流设计

```
Agent执行 → 更新Agent状态 → Registry存储状态
      ↓
Scheduler保存Job状态 → Workspace持久化
      ↓
Web Dashboard轮询API → 前端展示
```

**为什么是拉模式(polling)而非推模式(websocket)？**
- 实现简单，可靠性高
- 监控场景对实时性要求不高（秒级即可）
- 2秒轮询足够，避免过度复杂

---

## 4. 扩展性设计

### 4.1 如何添加新领域？

**步骤：**
1. 继承 `DomainTemplate`
2. 实现 `get_roles()` 定义角色
3. 实现 `decompose_task()` 分解任务
4. （可选）自定义 `aggregate_outputs()`

**示例：金融分析领域**

```python
class FinancialAnalysisTemplate(DomainTemplate):
    def get_roles(self):
        return [
            AgentRole("DataCollector", "数据收集", [...]),
            AgentRole("Analyst", "分析师", [...]),
            AgentRole("RiskManager", "风控", [...]),
        ]
    
    def decompose_task(self, input_text):
        # 返回股票分析流程
        return [
            {"id": "fetch", "name": "数据获取", "role": "DataCollector"},
            {"id": "analyze", "name": "技术分析", "role": "Analyst", "deps": ["fetch"]},
            {"id": "risk", "name": "风险评估", "role": "RiskManager", "deps": ["analyze"]},
        ]
```

### 4.2 如何添加新模型？

1. 在 `config.py` 添加模型配置
2. LLMAdapter会自动识别

```python
MACPConfig.MODELS["claude"] = {
    "api_key": "...",
    "base_url": "https://api.anthropic.com/v1",
    "model": "claude-3-opus",
    "temperature": 0.7,
}
```

### 4.3 如何添加新工具？

1. 在 `tools/registry.py` 定义Tool
2. Agent会自动加载

```python
def my_tool_handler(args, context):
    return {"result": "..."}

Tool(
    name="my_tool",
    description="...",
    parameters={...},
    handler=my_tool_handler
)
```

---

## 5. 项目亮点

### 5.1 架构层面

1. **清晰的层次分离**：Template-Engine-Agent三层架构
2. **可插拔设计**：领域、模型、工具均可热插拔
3. **状态驱动**：所有状态变更可追溯、可持久化

### 5.2 工程层面

1. **轻量级核心**：除去模板，核心引擎约1000行代码
2. **类型安全**：全程使用Type Hint，便于IDE提示
3. **完整监控**：从Agent状态到任务进度的全链路观测

### 5.3 产品层面

1. **即开即用**：内置软件研发模板，可直接使用
2. **可视化**：Web界面实时展示协作过程
3. **人机协作**：告警机制支持人工干预

---

## 6. 已知局限与未来优化

### 6.1 当前局限

1. **冲突解决基础**：仅实现了事件通知，复杂的Consensus策略待完善
2. **工具调用简单**：未实现真正的代码执行环境
3. **单点调度**：Scheduler是单线程的，大量任务可能成为瓶颈

### 6.2 未来优化方向

1. **异步化**：迁移到asyncio提升并发性能
2. **持久化队列**：使用Redis/RabbitMQ支持分布式部署
3. **智能路由**：基于Agent负载和历史表现动态分配任务
4. **人机协作增强**：支持人工审核关键步骤

---

## 7. 相关技术对比

| 项目 | 定位 | 特点 | MACP差异 |
|------|------|------|---------|
| AutoGen (Microsoft) | 多Agent对话 | 灵活的消息路由 | MACP更强调结构化流程(DAG) |
| MetaGPT | 软件开发 | 标准化SOP | MACP不限于软件领域 |
| CrewAI | Agent团队 | 角色扮演 | MACP有更完整的监控体系 |
| LangGraph | 工作流编排 | 与LangChain集成 | MACP更轻量、模板化 |

**MACP的定位**：介于重量级框架(AutoGen/MetaGPT)和轻量级库(LangGraph)之间，提供完整的领域模板系统+监控能力，同时保持核心代码简洁。

---

## 8. 快速参考

### 8.1 核心类图

```
DomainTemplate (ABC)
    ├─ get_roles() → List[AgentRole]
    ├─ decompose_task() → List[Task]
    └─ aggregate_outputs()

CollaborationFramework
    ├─ execute() → Job
    ├─ get_job_status() → Dict
    ├─ get_agent_status() → List[Dict]
    └─ get_overall_progress() → Dict

Agent (ABC)
    ├─ LLMAgent
    │   ├─ execute()
    │   └─ get_state()
    └─ AgentRegistry
        ├─ register()
        └─ find_by_role()

Scheduler
    ├─ submit_job()
    ├─ cancel_job()
    └─ _execute_job()
```

### 8.2 关键数据流

```
User Input → Framework.execute() → Template.decompose_task()
                                      ↓
Scheduler.submit_job() ← Task[] ← Job
       ↓
Executor.parallel_execute() → Agent.execute() → LLM.chat()
       ↓
Workspace.save() + EventBus.publish()
       ↓
Web Dashboard polling → UI Update
```

---

*Document Version: 1.0*
*Last Updated: 2026-04-13*
