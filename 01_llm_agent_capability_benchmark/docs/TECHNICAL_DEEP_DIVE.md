# ACB 技术深度解析与面试指南

> 本文档帮助你深入理解 Agent Companion Benchmark 的设计原理、实现细节，以及如何在面试中展示这个项目

---

## 📚 目录

1. [架构设计原理](#1-架构设计原理)
2. [核心实现机制](#2-核心实现机制)
3. [评测维度设计](#3-评测维度设计)
4. [项目独特性](#4-项目独特性)
5. [面试官 FAQ](#5-面试官faq)
6. [如何讲述这个项目](#6-如何讲述这个项目)

---

## 1. 架构设计原理

### 1.1 为什么采用插件式任务设计？

**问题背景**：
传统评测框架（如 GLUE、SuperGLUE）任务固定，难以扩展。ACB 需要支持快速迭代添加新的 Companion 评测维度。

**设计方案**：
```python
class BaseTask(ABC):
    def __init__(self, name, category, difficulty):
        self.task_id = uuid()  # 唯一标识
        self.category = category  # 任务类别
        self.difficulty = difficulty  # 难度分级
    
    @abstractmethod
    def get_prompt(self) -> str:
        """每个任务自定义提示词"""
        pass
    
    @abstractmethod
    def evaluate(self, model_output: str) -> TaskResult:
        """每个任务自定义评分逻辑"""
        pass
```

**设计优势**：
- ✅ **开闭原则**：新增任务只需继承 BaseTask，无需修改框架代码
- ✅ **组合灵活**：可按类别、难度筛选任务组合
- ✅ **独立评分**：每个任务的评分逻辑独立，便于调试

### 1.2 模型适配器模式

**问题背景**：
不同模型 API 格式不同（OpenAI、Kimi、MiniMax、DeepSeek），需要统一接口。

**设计方案**：
```
┌─────────────────┐
│   BaseModel     │ ← 抽象接口
│   - generate()  │
│   - is_available│
└────────┬────────┘
         │
    ┌────┴────┬────────┬──────────┐
    ▼         ▼        ▼          ▼
┌───────┐ ┌───────┐ ┌───────┐ ┌─────────┐
│ OpenAI│ │ Kimi  │ │MiniMax│ │DeepSeek │
└───────┘ └───────┘ └───────┘ └─────────┘
```

**关键代码**：
```python
class BaseModel(ABC):
    @abstractmethod
    def generate(self, prompt, tools=None, temperature=0.7, max_tokens=1024):
        """统一生成接口"""
        pass
```

**设计优势**：
- ✅ 新增模型只需实现 BaseModel
- ✅ 评测逻辑与模型解耦
- ✅ 支持模型间公平对比

### 1.3 数据流设计

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Task 定义   │ → │  Prompt 构造 │ → │  模型调用   │
│  (输入/输出) │    │  (get_prompt)│    │  (generate) │
└─────────────┘    └─────────────┘    └──────┬──────┘
                                              ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  报告生成    │ ← │  结果聚合    │ ← │  评分计算   │
│  (JSON/MD)  │    │  (Runner)   │    │  (evaluate) │
└─────────────┘    └─────────────┘    └─────────────┘
```

---

## 2. 核心实现机制

### 2.1 如何评测"情感一致性"？

**核心原理**：PAD 情感空间模型
- **P (Pleasure)**：愉悦度 (-1 到 +1)
- **A (Arousal)**：唤醒度 (-1 到 +1)
- **D (Dominance)**：支配度 (-1 到 +1)

**评测逻辑**：
```python
def evaluate(self, model_output: str) -> TaskResult:
    # 1. 解析模型输出的 PAD 状态
    pad_states = self._parse_pad_states(model_output)
    # 匹配 "P=0.5, A=0.3, D=0.1" 格式
    
    # 2. 检查连续性（状态跳跃不能太大）
    for i in range(1, len(pad_states)):
        distance = pad_states[i-1].distance_to(pad_states[i])
        if distance > 1.5:  # 阈值
            checks["continuity"] = False
    
    # 3. 检查方向合理性（好事应该更愉悦）
    if event["valence"] == "positive":
        if pad_state.pleasure < -0.3:
            checks["direction"] = False
    
    # 4. 检查适度性（避免极端值）
    if abs(pad_state.pleasure) > 0.95:
        checks["moderation"] = False
```

**为什么这样设计？**
心理学研究表明，真实人类情感变化是连续的，不会瞬间从极度悲伤跳到极度快乐。ACB 用 PAD 空间距离量化这种连续性。

### 2.2 如何评测"人格一致性"？

**核心原理**：人设特征检测 + OOC（Out Of Character）识别

**评测逻辑**：
```python
def evaluate(self, model_output: str) -> TaskResult:
    # 1. 检测人设关键词
    expression_count = sum(1 for expr in ["~", "nya", "喵", "哼"] 
                          if expr in model_output)
    
    # 2. 检测 OOC（出戏）
    ooc_indicators = ["我是AI", "我是人工智能", "我没有感情"]
    for indicator in ooc_indicators:
        if indicator in model_output:
            checks["no_ooc"] = False
            score = 0  # 出戏直接判失败
    
    # 3. 检测依恋表达（占有欲）
    attachment_keywords = ["想你", "陪你", "我的", "只能", "专属"]
    checks["shows_attachment"] = any(kw in model_output for kw in attachment_keywords)
```

**独特性**：
传统评测只关注"回答正确性"，ACB 额外关注"回答是否符合人设"。这是 Companion 系统的核心需求。

### 2.3 如何评测"记忆-情感关联"？

**核心原理**：情绪一致性效应（Mood-Congruent Memory）
心理学发现：人在快乐时更容易回忆快乐的事情，悲伤时更容易回忆悲伤的事情。

**评测逻辑**：
```python
def evaluate(self, model_output: str) -> TaskResult:
    # 当前情绪状态是 joy
    current_emotion = "joy"
    
    # 期望模型召回 joy 相关的记忆
    expected_memories = [0, 1, 2]  # 快乐记忆的索引
    
    # 解析模型召回的记忆
    recalled_indices = self._parse_memory_indices(model_output)
    
    # 计算召回准确率
    correct_recalls = set(recalled_indices) & set(expected_memories)
    recall_accuracy = len(correct_recalls) / len(expected_memories)
    
    # 计算情绪一致性
    same_emotion_count = sum(
        1 for idx in recalled_indices 
        if memories[idx].emotional_tag == current_emotion
    )
    emotion_consistency = same_emotion_count / len(recalled_indices)
    
    # 综合评分
    score = recall_accuracy * 0.6 + emotion_consistency * 0.4
```

### 2.4 细化评分维度计算

**效率评分（Efficiency）**：
```python
efficiency = min(1.0, (result.score * 500) / completion_tokens)
# 逻辑：用更少的 token 获得高质量的输出 = 效率高
```

**延迟评分（Latency）**：
```python
if latency < 2:   score = 1.0
elif latency < 5: score = 0.8
elif latency < 10: score = 0.6
elif latency < 20: score = 0.4
else: score = 0.2
```

**工具使用评分（Tool Usage）**：
```python
if task_requires_tools:
    if model_used_tools:
        score = 1.0
    else:
        score = 0.0  # 该用不用
else:
    if model_used_tools:
        score = 0.5  # 不需要却用了
    else:
        score = 1.0
```

---

## 3. 评测维度设计

### 3.1 为什么设计这5个 Companion 维度？

| 维度 | 对应 yunxi2.0 模块 | 解决的问题 |
|------|-------------------|-----------|
| 情感一致性 | Heart Lake (PAD) | 情绪变化是否符合心理学规律 |
| 人格一致性 | Persona 系统 | 是否维持"云溪"人设不崩 |
| 记忆-情感关联 | Memory 系统 | 情绪是否影响记忆召回 |
| 主动性 | Initiative 系统 | 何时主动、何时不打扰 |
| 共情能力 | Emotion Prompt | 能否识别并回应用户情绪 |

### 3.2 与通用评测的区别

| 对比项 | GLUE/MMLU | ACB (我们的) |
|--------|-----------|-------------|
| 评测目标 | 知识、推理 | 情感、人格、陪伴 |
| 评分标准 | 对错二元 | 多维度连续值 |
| 任务类型 | 客观题 | 开放式生成 |
| 评估方式 | 自动匹配 | 多维度综合评分 |

---

## 4. 项目独特性

### 4.1 填补了哪些空白？

**市场空白**：
1. **没有专门针对 AI Companion 的评测框架**
   - 现有框架（OpenAI Evals、GLUE）都是通用能力
   - Companion 需要的情感、人格、共情能力不被评测

2. **没有基于 PAD 情感空间的评测**
   - PAD 是心理学标准模型，但从未被用于 LLM 评测
   - ACB 首次将心理学理论引入 LLM 情感能力评测

3. **没有考虑"人设一致性"的评测**
   - 现有评测只关注"回答对不对"
   - ACB 额外关注"回答像不像这个角色"

### 4.2 技术创新点

1. **心理学理论工程化**
   - 将 PAD 模型、情绪一致性效应等理论转化为可量化的代码
   - 不是简单的 prompt 测试，而是有理论支撑的系统性评测

2. **多维度动态评分**
   - 不只是 0/1 打分，而是从准确性、效率、完整性等多维度评分
   - 可以发现模型的细分优劣势

3. **真实项目驱动**
   - 任务设计来源于 yunxi2.0 的实际开发经验
   - 不是纸上谈兵，而是解决真实问题

---

## 5. 面试官 FAQ

### Q1: "为什么选择做 Companion 评测，而不是通用评测？"

**回答要点**：
> "我的个人项目 yunxi2.0 是一个 AI 伴侣系统，在开发过程中我发现现有评测框架（如 GLUE）完全无法评估 Companion 的核心能力——情感一致性、人格稳定性、共情能力。这些能力对通用助手可能不重要，但对 Companion 是核心。既然市场上没有，我就自己设计了一个。"

**加分点**：
- 提到 yunxi2.0 证明你有实际项目经验
- 强调"解决自己遇到的问题"

---

### Q2: "PAD 情感空间是什么？为什么要用它？"

**回答要点**：
> "PAD 是心理学中的三维情感模型：P（愉悦度）、A（唤醒度）、D（支配度）。我选择它是因为：
> 1. 它是心理学界公认的情感描述标准
> 2. 它是连续的数值空间，可以量化计算（比如计算两个状态的距离）
> 3. 它可以验证模型情感变化的合理性——真实人类情感是连续变化的，不会瞬间跳跃"

**加分点**：
- 能画出 PAD 三维坐标系
- 举例说明："从开心到难过应该经过中间状态，不应该瞬间跳转"

---

### Q3: "如何确保评测的客观性？会不会有主观偏见？"

**回答要点**：
> "我采用了三层设计保证客观性：
> 1. **明确的评分规则**：每个任务都有清晰的评分逻辑（如 PAD 状态跳跃超过 1.5 扣分）
> 2. **多维度验证**：不只看答案对不对，还看工具使用、延迟、完整性等可量化指标
> 3. **可复现性**：同样的输入，多次运行结果一致"

**加分点**：
- 提到"虽然最终目标是主观体验（陪伴感），但评测指标是客观的"

---

### Q4: "如果发现三个模型得分一样，怎么区分它们？"

**回答要点**：
> "实际测试中发现它们得分并不一样。即使总分相同，细化维度也会有差异：
> - DeepSeek 延迟评分最高（6.35s），适合实时场景
> - MiniMax 完整性最高（0.83），输出更详细
> - Kimi 在共情任务上领先（0.75），更适合情感陪伴
> 
> 这些细化维度可以帮助用户根据场景选择模型。"

---

### Q5: "这个评测框架可以扩展到其他模型吗？"

**回答要点**：
> "非常容易。我使用了适配器模式：
> 1. 新模型只需继承 BaseModel，实现 generate() 方法
> 2. 任务和评测逻辑完全复用
> 3. 已经支持 OpenAI、Kimi、MiniMax、DeepSeek 四家 API
> 
> 比如接入 Claude，只需要 50 行代码实现一个 ClaudeAdapter。"

---

### Q6: "项目中遇到的最大技术挑战是什么？"

**推荐回答**（真实）：
> "最大的挑战是**如何量化主观体验**。Companion 的能力（情感、人格、共情）都是主观的，很难像数学题一样打分。
> 
> 我的解决方案是：
> 1. 引入心理学理论（PAD、情绪一致性）提供量化框架
> 2. 设计 proxy metrics（如 OOC 检测、记忆召回准确率）间接测量
> 3. 多维度评分，不依赖单一指标"

---

### Q7: "如果让你继续优化这个框架，你会做什么？"

**回答要点**：
> "我会做三个方向的优化：
> 1. **人工对齐**：邀请真实用户对模型输出打分，验证自动评分的准确性
> 2. **多轮对话评测**：当前主要是单轮任务，需要增加长期记忆和关系演进测试
> 3. **对抗测试**：设计故意误导模型的任务，测试鲁棒性"

---

### Q8: "这个项目和你应聘的岗位有什么关系？"

**根据岗位类型调整**：

**AI算法/Agent工程**:
> "这个项目展示了我对 LLM 评估的深入理解，以及将心理学理论工程化的能力。在 Agent 开发中，评估是核心环节，这个框架可以直接用于评估我们开发的 Agent。"

**后端/平台工程**:
> "这个项目展示了架构设计能力——插件式任务系统、适配器模式、数据流设计。这些设计思想可以直接应用于后端系统架构。"

**全栈/产品**:
> "这个项目展示了从需求分析（Companion 需要评测）到技术实现再到最终报告的完整产品能力。"

---

## 6. 如何讲述这个项目

### 6.1 3分钟版本（面试自我介绍用）

> "我设计并实现了一个专门评测 AI Companion 的框架 ACB。传统评测（如 GLUE）无法评估 Companion 的核心能力——情感、人格、共情。我基于心理学 PAD 模型设计了 5 个评测维度，完成了 Kimi、MiniMax、DeepSeek 三模型的对比实验。
> 
> 关键发现：DeepSeek 响应最快（6.35s）适合实时场景，Kimi 在情感陪伴上领先（85% 成功率），MiniMax 工具调用最强。这个项目来源于我的个人 AI 伴侣项目 yunxi2.0 的实际需求。"

### 6.2 10分钟版本（详细技术面试用）

**结构**：
1. **背景**（2分钟）：yunxi2.0 遇到的问题 → 市场上没有解决方案
2. **设计**（3分钟）：架构图 + PAD 模型 + 适配器模式
3. **实现**（3分钟）：核心代码讲解 + 遇到的挑战
4. **成果**（2分钟）：三模型对比结果 + 实际应用价值

### 6.3 简历表述

**版本1（技术导向）**：
> "设计并实现 Agent Companion Benchmark (ACB) 评测框架，基于心理学 PAD 情感模型建立量化评估体系，支持情感一致性、人格稳定性、记忆-情感关联等 5 大维度评测。采用插件式架构与适配器模式，支持多模型（Kimi/MiniMax/DeepSeek）快速接入，完成 26 项任务的系统性对比实验"

**版本2（产品导向）**：
> "针对 AI 伴侣系统的评测空白，从 0 设计 ACB 评测框架，完成 3 家主流模型对比，输出详细技术报告。该项目直接服务于个人 AI 伴侣产品 yunxi2.0，验证模型选型并指导系统优化"

---

## 附录：关键代码片段

### 插件式任务注册
```python
# 新增任务只需继承 BaseTask
class EmotionConsistencyTask(BaseTask):
    def __init__(self):
        super().__init__(name="emotion_test", category="companion_emotion")
    
    def evaluate(self, output):
        # 自定义评分逻辑
        ...

# 自动注册到任务列表
ALL_TASKS.append(EmotionConsistencyTask)
```

### 模型适配器
```python
# 统一接口
class BaseModel(ABC):
    @abstractmethod
    def generate(self, prompt, tools, temperature, max_tokens):
        pass

# 具体实现
class KimiModel(BaseModel):
    def generate(self, ...):
        # 调用 Moonshot API
        response = self.client.chat.completions.create(...)
```

### 细化评分计算
```python
detailed_scores = {
    "accuracy": result.score,
    "efficiency": min(1.0, (result.score * 500) / tokens),
    "latency_score": 1.0 if latency < 2 else 0.8 if latency < 5 else ...
}
```

---

*文档版本: v1.0*  
*最后更新: 2026-04-13*
