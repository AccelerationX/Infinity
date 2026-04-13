# 14 MCP Tool Hub —— 可插拔、可发现、可编排的 Agent 工具中枢

> 基于 MCP（Model Context Protocol）标准，让 Agent 动态发现工具、自动编排工具链，并在安全沙箱中执行高风险操作。

## 1. 项目定位

本项目研究如何构建一个**基于开放标准的 Agent 工具中枢（Tool Hub）**，解决当前 Agent 系统中工具"硬编码、难扩展、不安全"的问题。

核心研究点：
- **动态工具发现**：Agent 不需要在代码里写死工具列表，通过 MCP Server 自动注册和发现。
- **工具链自动编排**：识别工具间的输入输出依赖，自动生成最优执行 DAG。
- **安全隔离与权限降级**：高风险工具（文件删除、命令执行）在隔离环境中运行，并需要用户显式确认。

## 2. 核心问题

1. 工具语义匹配：用户说"帮我查天气"，Hub 如何从 20 个工具中精准选出 `weather_get`？
2. 工具链规划：复杂任务需要多个工具串行/并行，如何生成最小执行图？
3. 安全与扩展的平衡：新工具加入时，如何自动评估风险等级并施加对应沙箱策略？

## 3. 技术架构

```
自然语言意图
    ↓
SemanticMatcher (BGE + Keyword Gate)
    ↓
ArgumentResolver (Regex / LLM)
    ↓
DAGPlanner (拓扑排序 + 风险评估)
    ↓
SecurityManager (safe / caution / dangerous)
    ↓
MCPClient → stdio → MCP Server → 执行
    ↓
AuditLogger (JSONL 审计日志)
```

## 4. 目录结构

```
14_mcp_tool_hub/
├── src/
│   ├── schemas/models.py          # 核心数据模型
│   ├── client.py                  # 官方 MCP stdio Client
│   ├── hub.py                     # Tool Hub：注册、发现、路由、执行
│   ├── security.py                # 风险评估与权限确认
│   ├── audit_logger.py            # 审计日志
│   ├── semantic_matcher.py        # 语义匹配（embedding + keyword gate）
│   ├── argument_resolver.py       # 参数推断（regex / llm）
│   ├── planner.py                 # DAG 工具链规划器
│   └── servers/
│       ├── filesystem_server.py   # 文件读写（FastMCP 标准）
│       ├── calculator_server.py   # 安全数学计算（AST 沙箱）
│       ├── browser_server.py      # 网页抓取与搜索
│       └── shell_server.py        # 受限 Shell（黑名单过滤）
├── tests/
├── demos/
│   ├── demo_calculate_and_filesystem.py
│   └── demo_security_rejection.py
├── benchmark/
│   └── run_benchmark.py
└── README.md
```

## 5. 已验证的端到端 Demo

### Demo 1：自然语言 → 工具链执行
```bash
python demos/demo_calculate_and_filesystem.py
```
效果：输入 "帮我列出当前目录并计算 23+45"，Hub 自动匹配 `list_directory` 和 `calculate`，推断参数，串行执行，输出结果并写入审计日志。

### Demo 2：安全策略拦截
```bash
python demos/demo_security_rejection.py
```
效果：输入危险 shell 命令，Hub 识别风险等级为 dangerous，在未获得用户确认时拒绝执行。

### Demo 3：批量 Benchmark
```bash
python benchmark/run_benchmark.py
```
效果：运行 4 个标准测试用例，输出 Recall / Args / Exec 三项指标。当前结果：
- **Recall Rate: 100%**
- **Args Rate: 100%**
- **Exec Rate: 100%**

## 6. 当前能力与边界

| 能力 | 状态 | 说明 |
|------|------|------|
| 标准 MCP Server (FastMCP) | ✅ | filesystem / calculator / browser / shell |
| stdio Client | ✅ | 基于官方 MCP SDK 1.27 |
| 语义匹配 (BGE + Keyword Gate) | ✅ | embedding 相似度 + 关键词硬过滤，减少误召回 |
| DAG 规划 | ✅ | 基于优先级的串行执行，含风险评估 |
| 参数推断 (Regex Fallback) | ✅ | 支持文件路径、数学表达式、URL、命令等自动提取 |
| 参数推断 (LLM) | 🟡 | 接口已预留，接入任意 OpenAI-compatible API 即可 |
| 安全策略 | ✅ | 三级风险校验 + 用户确认 + 黑白名单 |
| 审计日志 | ✅ | JSONL 持久化，支持成功率统计和错误回放 |

## 7. 依赖

```bash
pip install mcp sentence-transformers numpy networkx pydantic requests pytest
```

## 8. 快速开始

```bash
cd 14_mcp_tool_hub
python demos/demo_calculate_and_filesystem.py
```
