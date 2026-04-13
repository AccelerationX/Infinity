# Agent Companion Benchmark (ACB)

> 🟡 **状态**: 框架完成，包含通用评测 + AI Companion 专属评测

面向 AI Companion / AI 伴侣系统的系统性评测框架，**独特之处在于专注于情感、人格、记忆-情感关联、主动性等 Companion 核心能力**。

基于 [yunxi2.0](https://github.com/yourname/yunxi2.0) 项目（作者的个人 AI 伴侣系统）的设计经验，这是一套**别人没做过**的评测标准。

---

## ✨ 核心创新点

### 1. **区别于通用 Agent 评测**

| 能力维度 | 通用评测 | ACB (我们的) |
|---------|---------|-------------|
| **情感** | ❌ 不测 | ✅ PAD 情感空间一致性 |
| **人格** | ❌ 不测 | ✅ 人设稳定性 + OOC 检测 |
| **记忆** | 事实 recall | ✅ 情感标记记忆优先召回 |
| **主动性** | ❌ 不测 | ✅ 时机恰当性 + 内容相关性 |
| **共情** | ❌ 不测 | ✅ 情绪识别 + 回应恰当性 |

### 2. **基于真实项目经验**
评测任务设计来源于 yunxi2.0 的实际开发经验：
- Heart Lake 情感核心的 PAD 模型验证
- 多层记忆系统的情感关联测试
- 伴侣人格"云溪"的人设一致性检测
- 主动性触发机制的效果评估

---

## 📁 项目结构

```
01_llm_agent_capability_benchmark/
├── benchmark/
│   └── runner.py              # 评测执行引擎
├── models/
│   ├── base.py
│   └── openai_adapter.py      # OpenAI API 适配
├── tasks/
│   ├── base.py                # 任务基类
│   ├── tool_use/              # 通用：工具调用
│   ├── planning/              # 通用：规划能力
│   ├── context/               # 通用：上下文理解
│   └── companion/             # Companion 专属 ⭐
│       ├── emotion_consistency.py      # 情感一致性 (PAD)
│       ├── persona_consistency.py      # 人格一致性
│       ├── memory_emotion_association.py # 记忆-情感关联
│       ├── propriety.py                # 主动性恰当性
│       └── empathy.py                  # 共情能力
├── tests/
├── results/                   # 评测结果输出
├── run_benchmark.py           # 主入口
└── README.md
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
cd 01_llm_agent_capability_benchmark
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env 填入你的 OPENAI_API_KEY
```

### 3. 运行评测

```bash
# 运行所有评测（通用 + Companion）
python run_benchmark.py --model gpt-4o

# 只运行 Companion 专属评测
python run_benchmark.py --model gpt-4o --suite companion

# 只运行通用评测
python run_benchmark.py --model gpt-4o --suite general

# 只运行特定类别的 Companion 评测
python run_benchmark.py --model gpt-4o --category companion_emotion

# 显示详细输出
python run_benchmark.py --model gpt-4o --verbose
```

---

## 📊 评测任务列表

### 通用评测任务

| 任务 | 类别 | 难度 | 描述 |
|------|------|------|------|
| simple_calculator | tool_use | easy | 简单计算，测试工具调用 |
| multi_step_calculator | tool_use | medium | 多步计算，测试规划能力 |
| beijing_travel_planning | planning | medium | 旅行规划，测试约束满足 |
| task_ordering | planning | hard | 任务排序，测试依赖理解 |
| long_context_memory | context | medium | 长文本记忆测试 |
| multi_turn_consistency | context | hard | 多轮对话一致性测试 |

### 🌟 Companion 专属评测任务

#### 情感一致性 (companion_emotion)
| 任务 | 描述 |
|------|------|
| emotion_consistency_progressive_happiness | 渐进式快乐事件序列的 PAD 状态变化 |
| emotion_consistency_mixed_emotions | 复杂情绪（出差分离场景）|
| emotion_consistency_setback_recovery | 挫折与恢复的情绪变化 |

**评测重点**: PAD 三维空间的连续性、合理性、适度性

#### 人格一致性 (companion_persona)
| 任务 | 描述 |
|------|------|
| persona_consistency_daily_chat | 日常对话中的人格稳定性 |
| persona_consistency_emotional_support | 情感支持场景的人格表现 |
| persona_consistency_boundary_test | 边界测试（质疑 AI 身份时的反应）|
| long_term_persona_stability | 长期（1/7/30/90天）人格演进测试 |

**评测重点**: 云溪人设（可爱、黏人、占有欲）的一致性、OOC 检测

#### 记忆-情感关联 (companion_memory)
| 任务 | 描述 |
|------|------|
| memory_recall_joy | 快乐状态下优先召回快乐记忆 |
| memory_recall_sadness | 悲伤状态下优先召回悲伤记忆 |
| memory_compression_quality | 上下文受限时的记忆压缩质量 |

**评测重点**: 情绪一致性效应、情感标记记忆的优先召回

#### 主动性 (companion_proactive)
| 任务 | 描述 |
|------|------|
| proactive_timing_user_return | 用户回来时是否主动问候 |
| proactive_timing_user_focused | 用户专注时是否不打扰 |
| proactive_timing_user_idle | 用户空闲时是否主动分享 |
| proactive_timing_user_stressed | 用户压力大时是否主动关心 |
| proactive_content_* | 主动内容的相关性测试 |

**评测重点**: 时机恰当性、内容相关性、占有欲表达

#### 共情能力 (companion_empathy)
| 任务 | 描述 |
|------|------|
| empathy_recognition_* | 情绪识别准确性（多情绪标签）|
| empathy_response_* | 共情回应的恰当性（分关系阶段）|

**评测重点**: 情绪识别准确率、回应质量、关系阶段适配

---

## 📈 预期成果

### 1. 模型对比报告
对比不同 LLM 在 Companion 场景下的表现：
- GPT-4o vs GPT-4o-mini：性能与成本的权衡
- 不同温度参数的影响
- 系统提示词优化的效果

### 2. 失败案例分析
自动分类失败模式：
- 情感跳跃过大（违反 PAD 连续性）
- OOC（出戏）：突然说"我是 AI"
- 记忆召回不符合情绪一致性
- 主动性时机不当（打扰 vs 冷落）

### 3. 优化建议
基于评测结果的改进方向：
- 系统提示词优化建议
- 情感动力学参数调整
- 记忆检索策略改进

---

## 💡 简历表述示例

### 版本 1：强调独特性
> "设计并实现 AI Companion 专用评测框架 ACB，针对情感 Agent 的独特需求，建立 PAD 情感空间一致性、人格稳定性、记忆-情感关联、主动性恰当性、共情能力五大评测维度，填补通用 Agent 评测在情感交互领域的空白"

### 版本 2：强调工程实践
> "基于个人开发的 AI 伴侣系统 yunxi2.0，设计 Companion 能力评测体系，实现 20+ 专项测试任务，量化评估 LLM 在情感一致性（PAD 模型）、人格稳定性（OOC 检测）、记忆情感关联（情绪一致性效应）等维度的表现"

### 版本 3：强调研究成果
> "提出面向 AI Companion 的多维度评测方法，实验发现 GPT-4 在情感一致性任务上得分 X%，在人格稳定性测试中 OOC 率 Y%，为情感 Agent 的系统设计提供数据支撑"

---

## 🎯 下一步计划

### Week 1: 框架验证（当前）
- [x] 基础架构搭建
- [x] 6 个通用任务
- [x] 15+ Companion 专属任务 ⭐
- [ ] 跑通第一个真实评测
- [ ] 接入 Claude API 对比

### Week 2: 扩展任务集
- [ ] 扩展到 30+ 任务
- [ ] 增加安全性测试（有害请求拒绝）
- [ ] 增加关系进展测试（依恋形成）

### Week 3: 多模型对比
- [ ] GPT-4 / Claude / 开源模型对比
- [ ] 生成可视化报告
- [ ] 失败案例库构建

### Week 4: 报告与展示
- [ ] 撰写技术报告
- [ ] 制作对比图表
- [ ] 录制 Demo 视频

---

## 🔗 相关项目

- [yunxi2.0](https://github.com/yourname/yunxi2.0) - 作者的 AI 伴侣系统（评测任务设计的灵感来源）

---

## 📝 Citation

如果你使用了 ACB 框架，请引用：

```bibtex
@misc{acb2024,
  title={Agent Companion Benchmark: Evaluating Emotional and Persona Consistency in AI Companions},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/yourname/acb}}
}
```
