# MACP v2 - Multi-Agent Collaboration Protocol

通用多Agent协作框架，支持领域模板扩展和Web监控。

**📚 文档**: [设计文档](DESIGN.md) | [面试Q&A](INTERVIEW_QA.md) | [面试速查卡](INTERVIEW_CHEATSHEET.md)

## 功能特性

- **领域模板系统**: 软件研发、金融分析、科学研究等领域模板
- **DAG任务调度**: 支持复杂依赖关系的任务编排
- **Agent动态分配**: 基于角色和能力智能分配任务
- **Web监控仪表盘**: 实时显示Agent状态、进度和告警
- **多模型支持**: Kimi K2.5、DeepSeek V3.2等

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置API密钥

编辑 `macp/config.py` 或设置环境变量：

```python
KIMI_API_KEY = "your-api-key"
DEEPSEEK_API_KEY = "your-api-key"
```

### 3. 运行演示

```bash
python examples/full_demo.py
```

访问 http://localhost:8080 查看监控界面。

## 架构设计

```
macp/
├── core/               # 核心引擎
│   ├── framework.py    # 主框架
│   ├── scheduler.py    # DAG调度器
│   ├── job.py          # 任务定义
│   ├── workspace.py    # 工作区管理
│   └── event_bus.py    # 事件总线
├── agents/             # Agent实现
│   ├── base.py         # Agent基类
│   ├── llm_agent.py    # LLM Agent
│   └── registry.py     # Agent注册表
├── templates/          # 领域模板
│   ├── base.py         # 模板基类
│   └── software_dev.py # 软件研发模板
├── llm/                # LLM适配器
│   └── adapter.py      # 统一API调用
├── tools/              # 工具系统
│   └── registry.py     # 工具注册
└── web/                # Web监控
    └── dashboard.py    # 监控仪表盘
```

## 使用示例

### 基本使用

```python
from macp import CollaborationFramework
from macp.templates.software_dev import SoftwareDevelopmentTemplate

# 初始化
framework = CollaborationFramework(SoftwareDevelopmentTemplate())

# 执行任务
job = framework.execute("创建一个博客网站")

# 查看状态
status = framework.get_job_status(job.id)
print(f"进度: {status['progress_percent']}%")
```

### 使用Web监控

```python
from macp.web.dashboard import WebDashboard

# 启动监控
dashboard = WebDashboard(framework, port=8080)
dashboard.run()

# 访问 http://localhost:8080
```

### 自定义领域模板

```python
from macp.templates.base import DomainTemplate, AgentRole

class MyTemplate(DomainTemplate):
    def get_roles(self) -> List[AgentRole]:
        return [
            AgentRole(
                name="Analyst",
                description="Data Analyst",
                skills=["data_analysis", "visualization"]
            ),
        ]
    
    def decompose_task(self, input_text, context):
        # 自定义任务分解逻辑
        return [...]
```

## Web监控界面

监控界面包含：

1. **整体进度条**: 显示所有任务的平均完成度
2. **Agent状态面板**: 
   - 🟢 空闲 (idle)
   - 🔵 忙碌 (busy)
   - 🔴 错误 (error)
   - 当前执行的任务
3. **任务列表**: 每个任务的进度和状态
4. **告警面板**: 错误、阻塞等需要人工干预的问题

## 配置说明

### 模型配置 (macp/config.py)

```python
MACPConfig.MODELS = {
    "kimi": {
        "api_key": "...",
        "base_url": "https://api.moonshot.cn/v1",
        "model": "moonshot-v1-8k",
        "temperature": 1.0,  # Kimi要求固定1.0
    },
    "deepseek": {
        "api_key": "...",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "temperature": 0.7,
    },
}
```

### 调度器配置

```python
SCHEDULER_CONFIG = {
    "max_workers": 3,        # 最大并行任务数
    "task_timeout": 300,     # 任务超时时间(秒)
    "checkpoint_interval": 30,  # 状态保存间隔
}
```

## API文档

### CollaborationFramework

| 方法 | 说明 |
|------|------|
| `execute(input_text)` | 执行协作任务 |
| `get_job_status(job_id)` | 获取任务状态 |
| `get_agent_status()` | 获取Agent状态 |
| `get_overall_progress()` | 获取整体进度 |
| `get_alerts()` | 获取告警信息 |
| `cancel_job(job_id)` | 取消任务 |

## 许可证

MIT License
