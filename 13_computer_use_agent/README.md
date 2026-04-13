# 13 Computer Use Agent —— 视觉-操作闭环的 GUI Agent

> 让 AI 像人一样看屏幕、点鼠标，并在每一步操作后用视觉模型自我验证。

## 1. 项目定位

本项目研究如何让 LLM 通过**视觉感知 + 结构化规划 + 操作执行 + 视觉断言**的闭环，自主完成跨应用的 GUI 任务。

不同于简单的"截图→GPT→pyautogui"单跳代理，本项目的核心研究点是：
- **视觉断言（Visual Assertion）**：每一步操作后，用本地视觉小模型判断"当前屏幕是否符合预期"，降低误操作率。
- **操作原子化**：把鼠标操作拆成 `定位 → 悬停验证 → 点击 → 结果验证`，减少误触。
- **跨应用工作流记忆**：把成功的任务序列保存为"宏"，后续直接调用。

## 2. 核心问题

1. UI 元素的语义定位：分辨率变化、主题变化时如何稳定找到目标？
2. 多步任务的容错与回退：走错页面后，Agent 如何自我诊断并重新规划？
3. 执行结果的可验证性：Agent 说"已完成"，但真的完成了吗？

## 3. 技术路线（分阶段）

### Phase 1：最小可行闭环（第 1-2 周）
- 用 `pyautogui` 截图，调用 **Qwen2-VL-2B-Instruct** 或 **OmniParser** 解析屏幕元素
- 实现一个 ReAct 风格循环：`观察(截图) → 思考 → 行动(点击/输入) → 验证`
- 完成 3 个基础任务：打开浏览器搜索、打开记事本写文字、打开计算器做运算

### Phase 2：视觉断言机制（第 3-4 周）
- 设计"执行前截图"与"执行后截图"的对比逻辑
- 引入轻量视觉模型做"状态是否变化"的判断
- 设计回退策略：如果断言失败，回退到上一步并重新生成行动

### Phase 3：跨应用工作流与技能记忆（第 5-6 周）
- 把成功的任务序列保存为 YAML 格式的"GUI 宏"
- 支持参数化：例如"在 {app} 中搜索 {keyword}"
- 复杂任务："把桌面截图移动到 D 盘，然后打开微信发送给文件传输助手"

## 4. 目录结构

```
13_computer_use_agent/
├── docs/
│   ├── 01-week1-plan.md          # 第一周执行计划
│   ├── 02-visual-assertion-design.md
│   └── 03-evaluation-benchmark.md
├── src/
│   ├── __init__.py
│   ├── agent.py                   # 主循环：观察-思考-行动-验证
│   ├── perception.py              # 截图 + UI 解析
│   ├── planner.py                 # 任务分解与行动规划
│   ├── executor.py                # pyautogui / keyboard 操作执行
│   ├── verifier.py                # 视觉断言与状态对比
│   ├── memory.py                  # 成功工作流的记忆存储
│   └── utils.py
├── tests/
│   ├── test_perception.py
│   ├── test_executor.py
│   └── test_verifier.py
├── experiments/
│   └── benchmark_tasks/           # 定义 10-20 个测试任务
├── requirements.txt
└── README.md
```

## 5. 已验证的端到端 Demo

### Demo 1：记事本文字输入（UIA 后端，无需 GPU/API Key）
```bash
python demos/task_notepad_type.py
```
效果：Agent 会自动打开记事本，输入 `"Hello from Computer Use Agent"`，
用 pixel_diff 验证屏幕变化，并用 UIA 独立验证文本确实已写入。

### Demo 2：UIA 控件树探测
```bash
python demos/probe_uia_notepad.py
```
效果：打开记事本，打印当前前台窗口的所有可交互控件信息。

## 6. 当前能力与边界

| 能力 | 状态 | 说明 |
|------|------|------|
| 截图 + UIA 控件解析 | ✅ 可用 | Windows UI Automation 精确获取控件类型、文本、坐标 |
| Mock 视觉后端 | ✅ 可用 | 用于无 Windows 环境的单元测试 |
| Qwen2-VL 后端 | 🟡 待接入 | 模型下载脚本已就绪，需要 ~4GB 显存/内存 |
| OpenAI Vision 后端 | 🟡 待接入 | 需要 `OPENAI_API_KEY` |
| Fallback Planner | ✅ 可用 | 支持点击、输入、打开应用，能从引号提取目标文本 |
| LLM Planner | 🟡 待接入 | Prompt 和 JSON 解析已就绪，只需提供 `llm_client` |
| 原子化执行 | ✅ 可用 | 点击拆分为 hover → pause → click → pause |
| 视觉断言 (pixel_diff) | ✅ 可用 | 自适应阈值：type 0.3%，click 2% |
| 视觉断言 (LLM) | 🟡 预留 | 接口已设计，待接入视觉模型 |
| 技能记忆 (Macro) | ✅ 可用 | 成功序列自动保存为 JSON，可复用 |
| 任务评估器 | ✅ 可用 | UIA 读取控件内容，独立验证任务是否成功 |

## 7. Benchmark 任务集

见 `benchmark/tasks.yaml`，定义了从 easy 到 hard 的 5 个标准任务，
用于后续量化评估 Agent 的泛化能力。

## 8. 依赖

```bash
pip install pyautogui Pillow opencv-python transformers qwen-vl-utils torch accelerate uiautomation
```

## 9. 快速开始

```bash
cd 13_computer_use_agent
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python demos/task_notepad_type.py
```
