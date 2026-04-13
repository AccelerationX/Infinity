# MACP v2 - 面试速查卡

> 面试前快速复习用

---

## 30秒项目介绍

"MACP是一个**通用多Agent协作框架**，核心设计是**领域模板 + DAG调度引擎**的分离。软件研发场景下有7个角色（PM、架构师、前后端、QA、DevOps），任务按DAG依赖并行执行。我同时做了Web监控界面，可以实时看到Agent状态和整体进度。"

---

## 核心数据

| 指标 | 数值 |
|------|------|
| 代码行数 | ~3000行（不含模板） |
| Agent角色 | 7个（软件研发模板） |
| 支持模型 | Kimi K2.5, DeepSeek V3.2 |
| 调度并发 | 3个（可配置） |
| 监控刷新 | 2秒轮询 |

---

## 三大设计亮点

1. **Template-Engine分离**
   - Engine通用，Template领域相关
   - 新增领域只需写Template

2. **DAG并行调度**
   - 自动识别可并行任务
   - 状态机驱动执行

3. **全链路监控**
   - Agent状态、任务进度、告警
   - Web界面可视化

---

## 常见对比

| 问题 | 回答 |
|------|------|
| vs AutoGen | AutoGen太灵活缺结构，MACP有DAG约束 |
| vs MetaGPT | MetaGPT仅软件，MACP通用可扩展 |
| 线程vs异步 | 线程够用，asyncio留待未来优化 |
| 轮询vsWebSocket | 轮询简单可靠，2秒足够 |

---

## 可能深挖的技术点

### DAG调度算法
```python
while len(completed) < len(tasks):
    ready = [t for t in tasks if all(d in completed for d in t.dependencies)]
    for task in ready[:max_workers]:
        executor.submit(task)
```

### Agent状态机
```
idle → execute() → busy → complete → idle
                          ↓
                        error → (retry) → idle
```

### 错误处理策略
- 任务级：失败不影响其他任务
- 重试：LLM层3次重试
- 告警：EventBus通知Web界面
- 人工：暂停/恢复功能

---

## 项目局限（诚实回答）

1. **冲突解决基础**：只有事件通知，无自动仲裁
2. **工具调用简单**：无真实代码执行环境
3. **调度单点**：Scheduler单线程，大量任务可能瓶颈

**改进方向**：asyncio、Redis队列、智能路由

---

## 一句话回答清单

| 问题 | 一句话回答 |
|------|-----------|
| 为什么做多Agent？ | 单Agent能力边界+上下文限制+并行提效 |
| 核心架构？ | Template定义角色流程，Engine负责DAG调度，Agent执行具体任务 |
| 最大难点？ | DAG死锁检测/Agent状态同步竞态条件 |
| 如何保证质量？ | 预定义流程+Web监控告警+人工介入机制 |
| 后续优化？ | asyncio迁移、更多领域模板、智能任务分解 |

---

## 画图准备

**架构图（必须会画）：**
```
User → Framework → Template (角色定义)
              ↓
         Scheduler → Agent Pool
              ↓
         Workspace (持久化)
              ↓
         Web Dashboard
```

**DAG示例（软件研发）：**
```
需求分析 → 架构设计 → 任务规划
                         ↓
        ┌───────────────┼───────────────┐
        ↓               ↓               ↓
     前端开发        后端开发         测试用例
        └───────────────┴───────────────┘
                         ↓
                    集成验证
```

---

## 代码片段（可现场展示）

**Template接口（简洁优雅）：**
```python
class DomainTemplate(ABC):
    @abstractmethod
    def get_roles(self) -> List[AgentRole]: ...
    
    @abstractmethod
    def decompose_task(self, input: str) -> List[Task]: ...
```

**Agent状态获取（Web监控用）：**
```python
def get_agent_status(self) -> List[Dict]:
    return [{
        "agent_id": agent.get_id()[:8],
        "name": agent.get_name(),
        "role": agent.get_role_name(),
        "status": state.get("status", "idle"),
        "current_task": state.get("current_task"),
    } for agent in self.registry.get_all()]
```

---

## 心态提醒

1. **诚实**：不会就说"这方面经验有限，但我思路是..."
2. **引导**：把问题引向你熟悉的部分
3. **互动**：多问"您想了解哪方面细节？"
4. **展示**：主动画图、展示代码

---

**祝你面试成功！** 🎯
