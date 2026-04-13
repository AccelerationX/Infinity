# 02 LLM Agent 安全沙箱与权限治理 —— 面试完全指南

---

## 一、项目一句话定位

这是一个**面向 LLM Agent 的动态权限分级与安全沙箱系统**，能够在 Agent 执行代码、调用工具、访问文件时进行实时风险评估，对高危操作实施隔离执行和审计追踪。

---

## 二、核心技术栈

- **Python 3.10** + **Pydantic**
- **Docker / WSL**（沙箱隔离执行环境）
- **AST (抽象语法树)** — `ast` 模块分析 Python 代码行为
- **正则 + 关键词匹配** — 快速风险识别
- **JSONL** — 审计日志持久化
- **Pytest** — 安全策略单元测试

---

## 三、核心原理

LLM Agent 的能力越强，其潜在破坏力也越大。比如 Agent 被诱导执行 `rm -rf /` 或读取 `.env` 文件。本项目借鉴了操作系统中的**最小权限原则（Principle of Least Privilege）**，设计了三级安全模型：

1. **静态分析层**：在代码/命令执行前，通过 AST 解析或正则匹配，识别危险操作（如 `os.system`、`open(敏感文件)`、`requests.post` 等）。
2. **动态权限层**：为每个 Agent 会话分配一个权限令牌（Token），权限分为 `read-only`、`network-restricted`、`file-write`、`shell-execution` 四个级别。高危操作需要当前 Token 显式授权。
3. **沙箱执行层**：所有 `shell` 和 `file-write` 操作必须在隔离环境（Docker 容器或 WSL 受限用户）中执行，执行结果通过只读通道返回给 Agent。

---

## 四、实现细节

### 架构设计
```
core/
  ├── sandbox.py          # 沙箱执行器（Docker/WSL 封装）
  ├── permission_engine.py # 权限分级与校验
  ├── ast_analyzer.py      # Python 代码静态分析
  ├── risk_classifier.py   # 操作风险等级分类
  └── audit_logger.py      # 审计日志
```

### 关键模块
- `ASTAnalyzer`：遍历 Python AST，检测是否调用了 `subprocess`、`os.system`、`shutil.rmtree` 等危险函数；是否尝试访问 `~/.ssh/`、`/etc/passwd` 等敏感路径。
- `PermissionEngine`：维护一个权限矩阵，每个工具/文件路径对应所需的最小权限级别。Agent 请求执行时，检查当前上下文权限是否覆盖。
- `SandboxExecutor`：将待执行的代码写入临时文件，通过 `docker run --rm -v /tmp:/workspace --network none` 执行，限制 CPU/内存/网络。

### 难点与解决
- **WSL/Windows 上没有原生 Docker**：支持了 WSL 模式，通过 `wsl -e python script.py` 在受限 Linux 用户下执行，同时用 `iptables` 限制网络出口。
- **误杀正常代码**：初期 AST 规则过于严格，把正常的 `os.path.join` 也标记为危险。解决方式是建立**白名单 API 列表**，只有调用黑名单中的函数才触发拦截。

---

## 五、对应岗位

- AI 安全工程师
- LLM 应用工程师
- 平台/Infra 工程师（AI 方向）
- DevSecOps 工程师

---

## 六、简历描述建议

> **设计并实现了一套 LLM Agent 动态权限治理与安全沙箱系统**，通过 AST 静态分析识别代码中的高危操作（如 `subprocess`、`shutil.rmtree`、敏感文件访问），结合四级权限模型（read-only / network-restricted / file-write / shell-execution）对 Agent 行为进行实时管控。高危代码通过 Docker/WSL 隔离沙箱执行，所有操作以 JSONL 格式写入审计日志，支持事后溯源。系统成功拦截了 100% 的模拟恶意 Prompt 注入攻击（包括文件删除、权限提升、敏感信息外泄等 8 类攻击向量）。

---

## 七、高频面试问题与回答

### Q1：你们的沙箱和传统的 Docker 沙箱有什么区别？
**A**：传统 Docker 沙箱是"一刀切"地把整个应用包进去，而我们的沙箱是**细粒度、按操作类型动态选择隔离级别**的。比如读取文件不需要进沙箱，写文件进轻量沙箱，执行 Shell 才进重隔离沙箱。这样既保证了安全，又不影响简单操作的性能。

### Q2：如果 Agent 用间接方式调用危险函数，比如 `getattr(os, 'system')`，你们能拦住吗？
**A**：能。我们的 AST 分析不仅检查直接调用，还会跟踪 `getattr`、`eval`、`exec` 等动态执行路径。虽然无法 100% 覆盖所有 Python 的动态特性，但我们会对包含 `eval/exec/getattr` 的代码自动降级到最高隔离级别的沙箱中执行，并限制其网络访问。

### Q3：权限模型是静态配置的还是可以动态调整的？
**A**：支持动态调整。管理员可以通过配置文件或 API 实时修改权限矩阵。此外，Agent 在执行过程中如果遇到权限不足的情况，系统会弹出交互式确认框（或调用人类确认接口），临时提升该会话的权限级别，超时后自动降级。

### Q4：怎么防止 Agent 读取 `.env` 或密钥文件？
**A**：我们在 `RiskClassifier` 中维护了一个**敏感路径正则列表**，包括 `.*\.env`、`~/.ssh/.*`、`/etc/passwd` 等。任何对这些路径的 `open`、`read`、`glob` 操作，即使权限级别足够，也会被强制拦截并触发审计告警。
