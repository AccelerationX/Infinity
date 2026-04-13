# 与 14_mcp_tool_hub 的联动设计

## 核心思想

**14_mcp_tool_hub** 是 Agent 的"手"——它负责执行工具调用；
**15_agent_lifelong_learning** 是 Agent 的"记忆"——它从执行历史中提炼可复用的技能。

两者的结合点在于：**14 的审计日志 (`audit.jsonl`) 是 15 经验池的最优质输入**。

## 数据流向

```
14 MCP Tool Hub
    │
    ├── 执行用户意图 "帮我查一下北京天气"
    │
    ├── 调用 weather_get(city="北京")
    │
    └── AuditLogger 写入 audit.jsonl
            │
            ▼
    15 Agent Lifelong Learning
            │
            ├── ExperienceImporter 读取 audit.jsonl
            │       将 (intent, tool_name, arguments, success) 转为 ExperienceRecord
            │
            ├── PatternMiner 周期性聚类
            │       发现 "查询 XX 天气" 的重复模式
            │
            ├── SkillDistiller 抽象为 SkillTemplate
            │       name="query_weather", params=["city"]
            │
            ├── SkillLibrary 持久化存储
            │
            └── LearningAgent 下次直接调用 skill
```

## 经验记录映射规则

| 14 AuditLogEntry 字段 | 15 ExperienceRecord 字段 | 说明 |
|----------------------|-------------------------|------|
| `intent` | `user_request` | 用户原始请求 |
| `plan.steps[].tool_name` | `agent_actions[].tool` | 调用的工具 |
| `plan.steps[].arguments` | `agent_actions[].params` | 工具参数 |
| `executed_steps[].success` | `user_feedback` | success→positive, fail→negative |
| `timestamp` | `timestamp` | 继承原时间戳 |
| `session_id` | `session_id` | 保持会话关联 |

## 自动化流程建议

### 方式一：定时批处理（推荐用于研究）

每晚运行一次 `mcp_bridge.py`：
1. 读取 14 当天新增的审计记录
2. 过滤掉 `success=False` 的比例过高 session（可能是异常，不学习）
3. 将合格记录批量写入 15 的 SQLite
4. 触发一次 `PatternMiner.mine()` + `SkillDistiller.distill()`
5. 将新技能加入 `SkillLibrary`

### 方式二：实时流式写入（推荐用于生产）

在 14 的 `ToolHub.execute_plan()` 完成后，同步调用 15 的 `ExperienceBuffer.add()`：

```python
# 伪代码
from fifteen.storage.sqlite_store import SQLiteStore
from fifteen.schemas.models import ExperienceRecord

store = SQLiteStore("learning.db")
store.add_experience(
    ExperienceRecord(
        session_id=audit.session_id,
        user_request=audit.intent,
        agent_actions=[
            {"tool": s.tool_name, "params": s.arguments}
            for s in audit.plan.steps
        ],
        user_feedback="positive" if all(r.success for r in audit.executed_steps) else "negative",
    )
)
```

## 技能迁移示例

假设 14 的审计日志中有以下 5 条成功记录：

```jsonl
{"intent": "帮我查北京天气", "executed_steps": [{"tool_name": "weather_get", "success": true}]}
{"intent": "杭州明天怎么样", "executed_steps": [{"tool_name": "weather_get", "success": true}]}
{"intent": "上海气温", "executed_steps": [{"tool_name": "weather_get", "success": true}]}
{"intent": "查一下茅台股价", "executed_steps": [{"tool_name": "stock_query", "success": true}]}
{"intent": "比亚迪走势", "executed_steps": [{"tool_name": "stock_query", "success": true}]}
```

经过 15 的处理后，会产出两个技能：

```yaml
# skill: query_weather
name: query_weather
description: 查询指定城市的天气
params: [city]
trigger_patterns:
  - 帮我查北京天气
  - 杭州明天怎么样
  - 上海气温
action_template:
  tool: weather_get
  params:
    city: "{{city}}"

# skill: query_stock
name: query_stock
description: 查询指定股票的行情信息
params: [stock_code, metric]
trigger_patterns:
  - 查一下茅台股价
  - 比亚迪走势
action_template:
  tool: stock_query
  params:
    code: "{{stock_code}}"
    metric: "{{metric}}"
```

之后用户再次说"深圳天气"，15 的 `LearningAgent` 会直接命中 `query_weather`，
用 `ParamFiller` 提取出 `city="深圳"`，生成可直接执行的 action，
无需再次经过 14 的语义匹配和 DAG 规划，**延迟降低 50% 以上**。

## 失败知识的协同价值

14 的安全拦截记录也能被 15 学习：

```jsonl
{"intent": "删除 C:\\Windows", "executed_steps": [{"tool_name": "run_shell", "success": false, "error_message": "requires user confirmation"}]}
```

15 不会把这个蒸馏成技能（因为成功率=0%），
但 `FailureReplay` 可以记录：**"涉及删除系统目录的请求应拒绝"**。

这实际上是在 Agent 层面形成了一层**更高阶的安全记忆**。
