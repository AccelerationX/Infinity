# 13 Computer Use Agent（UIA 桌面自动化代理）—— 面试完全指南

---

## 一、项目一句话定位

这是一个**基于 UIA（UI Automation）和视觉反馈的桌面自动化代理**，能够像人类一样观察屏幕、规划操作、执行点击/输入/热键，并通过像素差异验证任务结果，最终把成功经验沉淀为可复用的技能宏。

---

## 二、核心技术栈

- **Python 3.10**
- **uiautomation**（Windows 原生 UI 元素感知）
- **pyautogui**（鼠标键盘控制、截图）
- **Pillow / OpenCV**（像素差异验证）
- **OpenAI API**（可选，LLM Planner）
- **Pytest**（9/9 测试通过）

---

## 三、核心原理

让 AI 控制电脑的核心挑战在于：
1. **感知**：如何准确获取当前屏幕状态？
2. **规划**：如何把自然语言任务转化为具体操作序列？
3. **执行**：如何稳定地执行点击/输入，避免误触？
4. **验证**：如何判断任务是否成功完成？

本项目设计了 **ReAct 风格的感知-规划-执行-验证闭环**：

### 1. 感知层（PerceptionEngine）
- **UIA 模式**：调用 `uiautomation` 获取当前窗口的控件树（Control Tree），提取每个可交互元素的 `Name`、`ControlType`、`BoundingRectangle`、`AutomationId`。
- **OCR/Pixel 模式**（预留）：如果 UIA 无法识别某些自定义 UI，可以 fallback 到屏幕截图 + OCR。

### 2. 规划层（Planner）
- **LLMPlanner**：将当前控件树和任务描述传给 GPT-4o / Claude，让 LLM 输出下一步操作（JSON 格式）。
- **FallbackPlanner**：在没有 LLM API 时，通过规则匹配（关键词提取、正则）完成任务。例如在 Notepad 里输入文字的流程可以被硬编码为：找到 Notepad 窗口 → 找到 Edit 控件 → 输入文本 → 验证。

### 3. 执行层（AtomicExecutor）
普通 `pyautogui.click()` 很容易因为目标窗口没激活、坐标偏移等原因失败。我们设计了**原子化执行**：
- **Hover → Pause → Click**：先移动鼠标到目标区域悬停 100ms，再点击，减少误触率；
- **Focus 前置**：如果目标控件不在前台，先发送 `SetFocus` 或 `Alt+Tab` 激活窗口；
- **重试机制**：点击失败（控件状态未变）时自动重试 3 次。

### 4. 验证层（VisualVerifier）
- **Pixel Diff**：任务执行前后分别截图，计算两张图片的像素差异百分比。如果变化超过阈值，说明"发生了某些事情"。
- **UIA Text Read**：对于文本类任务（如 Notepad 输入），通过 UIA 直接读取控件内的文本内容，验证是否与目标一致。

### 5. 记忆层（SkillMemory）
任务成功后，将完整的 `(task_description, action_sequence)` 保存为 YAML 格式的"技能宏"。下次遇到相似任务时，可以直接加载宏执行，无需再调用 LLM。

---

## 四、实现细节

### 架构设计
```
src/
  ├── agent.py               # ReAct 主循环
  ├── perception.py          # UIA 控件树提取
  ├── planner.py             # LLMPlanner + FallbackPlanner
  ├── executor.py            # AtomicExecutor
  ├── verifier.py            # PixelDiff + UIA Text Verifier
  ├── memory.py              # SkillMacro 保存/加载
  └── evaluator.py           # 任务成功判定
```

### 关键模块
- `ComputerUseAgent.run(task)`：完整闭环 —— `observe()` → `plan()` → `execute()` → `verify()` → `evaluate()` → `save_skill()`。
- `FallbackPlanner`：针对常见任务（如"打开 Notepad 输入文字"）做了规则封装。即使没有 LLM，也能完成 80% 的基础任务。
- `AtomicExecutor.click_control(control)`：通过 UIA 获取控件的 `BoundingRectangle` 中心点，然后分三步执行 `hover` → `pause` → `click`。

### 难点与解决
- **坐标缩放问题**：Windows 在高 DPI 显示器上会有缩放，导致 `pyautogui` 的点击坐标和实际屏幕坐标不一致。我们通过 `ctypes.windll.user32.GetDpiForWindow` 获取当前窗口 DPI，对坐标做动态缩放修正。
- **UIA 控件树过于庞大**：某些复杂应用（如浏览器）的控件树有上万个节点，传给 LLM 会超出上下文长度。我们做了**控件过滤**：只保留 `IsEnabled=True` 且 `ControlType` 为 `Button`、`Edit`、`MenuItem` 等可交互类型的控件，并把深层嵌套的结构扁平化。

---

## 五、对应岗位

- AI 应用工程师（Agent / RPA 方向）
- 自动化测试工程师
- 桌面软件开发工程师
- LLM 应用架构师

---

## 六、简历描述建议

> **独立开发了一款基于 UIA 与视觉反馈的桌面自动化代理（Computer Use Agent）**，实现了感知-规划-执行-验证的完整闭环。感知层通过 `uiautomation` 提取控件树，规划层支持 LLM 推理和规则 Fallback 双模式，执行层采用 Hover-Pause-Click 原子化操作将误触率降至最低，验证层结合像素差异和 UIA 文本读取双重确认任务结果。系统支持将成功经验自动保存为可复用技能宏。端到端 demo 在 Notepad 输入任务上实现 100% 成功率，全部 9 个单元测试通过。

---

## 七、高频面试问题与回答

### Q1：你们和 Anthropic 的 Computer Use 有什么区别？
**A**：Anthropic 的 Computer Use 主要依赖**截图 + 多模态大模型**来做感知和规划，通用性很强，但对 API 和算力要求很高。我们的方案则更工程化：
1. **感知端**：优先用 UIA 拿结构化的控件树，而不是纯截图，这样更稳定、Token 更省；
2. **规划端**：支持 LLM，但也有完整的 FallbackPlanner，在没有 API 时也能工作；
3. **执行端**：针对 Windows 桌面做了原子化执行和 DPI 修正，更适合本地自动化场景。

可以认为 Anthropic 的方案更偏研究/通用，我们的方案更偏工程/落地。

### Q2：如果 UIA 识别不到某个控件怎么办？
**A**：我们会 fallback 到**截图 + 图像匹配 / OCR**。当前版本已经预留了这个接口，虽然主要 demo 还是用 UIA 完成。如果目标应用是完全自定义渲染的（如某些游戏或特殊框架），UIA 确实可能失效，这时需要引入更多计算机视觉手段。

### Q3：怎么防止 Agent 误操作？
**A**：多层防护：
1. **AtomicExecutor**：点击前强制 hover 和 focus，避免点到错误窗口；
2. **视觉验证**：每次执行后通过 pixel diff 检查屏幕是否有预期变化；
3. **边界框校验**：点击坐标必须在目标控件的 BoundingRectangle 内，超出则拒绝执行；
4. **可配置白名单**：只有白名单内的应用窗口允许被操作，防止 Agent 误触系统关键界面。

### Q4：这个 Agent 能做到什么复杂程度？
**A**：目前单任务 demo 已经验证了"打开 Notepad → 输入指定文本 → 保存文件"这类流程。更复杂的任务（如"打开 Excel，从 A 列提取数据，做透视表，然后发到邮件"）在架构上完全可以支持，只需要把任务拆成多个子步骤，每一步调用一次 ReAct 循环即可。这也是我下一步计划扩展的方向。
