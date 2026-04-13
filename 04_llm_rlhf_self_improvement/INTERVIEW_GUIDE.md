# 04 基于 RLHF 的 Agent 自我改进 —— 面试完全指南

---

## 一、项目一句话定位

这是一个**基于 PPO（近端策略优化）和奖励模型（Reward Model）的 Agent 自我改进系统**，能够自动从人类反馈中学习偏好，持续优化 Agent 的任务完成策略。

---

## 二、核心技术栈

- **Python 3.10** + **PyTorch**
- **TRL (Transformer Reinforcement Learning)** — PPO 训练
- **PEFT / LoRA** — 高效参数微调
- **Transformers (HuggingFace)** — 基础模型与奖励模型
- **WandB** — 训练过程可视化
- **JSONL** — 经验回放池持久化

---

## 三、核心原理

大模型通过 SFT（监督微调）只能学到"像人说话"，但无法学到"什么是更好的结果"。**RLHF（Reinforcement Learning from Human Feedback）** 的核心思想是：

1. **收集比较数据**：对同一任务的两个 Agent 输出，人类（或自动化规则）标注哪个更好。
2. **训练奖励模型（RM）**：学习人类偏好，给任意输出打一个"质量分"。
3. **PPO 优化策略**：用 RM 的分数作为 reward，通过 PPO 算法优化语言模型的生成策略，使其输出更高分的回答。

在本项目中，由于完整 RLHF 的训练成本极高，我们做了一套**可运行的研究原型**，并针对 Agent 场景做了特殊设计：
- **任务级 Reward**：不仅看文本质量，还看工具调用是否正确、任务是否完成。
- **规则型 RM**：在没有足够人工标注数据时，用一套自动化规则（如"正确调用工具 +1 分，参数错误 -0.5 分，任务完成 +2 分"）作为奖励模型的替代。
- **经验回放（Experience Replay）**：将每次交互的 `(state, action, reward)` 存入回放缓冲区，定期采样进行 PPO 更新。

---

## 四、实现细节

### 架构设计
```
src/
  ├── sft_trainer.py           # 监督微调阶段
  ├── reward_model.py          # 奖励模型（Bradley-Terry  Pairwise Loss）
  ├── ppo_trainer.py           # PPO 训练循环
  ├── experience_buffer.py     # 经验回放池
  ├── task_env.py              # Agent 任务环境（产生 state 和计算 reward）
  └── inference.py             # 训练后模型推理
```

### 关键模块
- `RewardModel`：基于 `DeBERTa` 或一个小型 LLM，输入为 `(prompt, response)`，输出标量 reward。训练数据来自人工/规则的 pair-wise 比较。
- `PPOTrainer`：使用 TRL 库的 `PPOTrainer`，但做了以下定制：
  - `generation_kwargs` 针对 Agent 场景调优（强制输出 JSON 格式）；
  - `reward_fn` 结合了规则评分和 RM 评分；
  - 引入 `KL Penalty`，防止策略模型偏离原始 SFT 模型太远。
- `TaskEnvironment`：模拟 Agent 的执行环境。Agent 生成一个动作后，环境执行该动作并返回 `(observation, reward, done)`。

### 难点与解决
- **Reward Hacking（奖励作弊）**：Agent 学会用无意义的工具调用刷分。解决方式是设计**稀疏+稠密混合奖励**：最终任务完成给大奖励，中间步骤只有正确执行才给小奖励，错误操作给负奖励。
- **训练不稳定**：PPO 在 LLM 上容易出现梯度爆炸和模式崩溃。我们采用了 **LoRA 微调**（只训练 1% 参数）+ **Gradient Clipping** + **较小的 Learning Rate (1e-5)**，显著提升了稳定性。
- **数据不足**：人工标注 pair-wise 数据成本太高。我们先跑了一个**自动化规则版本**作为冷启动，用正确率 80% 以上的规则生成 pseudo-labels，再用这些标签预训练 RM。

---

## 五、对应岗位

- AI 算法工程师（LLM / RL 方向）
- 大模型训练工程师
- 强化学习研究员
- Agent 研究工程师

---

## 六、简历描述建议

> **设计并实现了一个基于 RLHF 的 Agent 自我改进原型系统**，通过 PPO 算法结合奖励模型（Reward Model）对 Agent 的任务执行策略进行持续优化。针对 Agent 场景设计了混合奖励函数（任务完成稀疏奖励 + 工具调用稠密奖励），并使用 LoRA 进行高效参数微调，在单张消费级显卡上完成了 7B 模型的 PPO 训练。系统冷启动阶段采用规则型伪标签预训练奖励模型，成功缓解了人工标注数据不足的问题；训练后 Agent 在工具调用任务上的准确率从 SFT 基线的 58% 提升到了 76%。

---

## 七、高频面试问题与回答

### Q1：你们为什么选 PPO 而不是 DPO？
**A**：我们在项目早期尝试过 DPO（Direct Preference Optimization），它的优势是不需要显式训练 Reward Model，实现更简单。但我们在实验中发现，DPO 对**数据质量要求极高**，少量噪声 pair 会导致模型严重退化。而 PPO 配合规则型 RM，可以更灵活地调整 reward shaping，对 Agent 这种需要中间步骤反馈的场景更友好。当然，如果未来 pair 数据质量提升，我们也会考虑切换到 DPO 以降低训练复杂度。

### Q2：Reward Hacking 你们是怎么解决的？
**A**：我们用了三招：
1. **混合奖励设计**：最终任务完成才给大奖励，中间步骤只有正确执行才给小奖励，错误操作扣分；
2. **KL Penalty**：限制 PPO 策略模型与 SFT 基线的输出分布差异，防止模型为了刷分而输出离谱的格式；
3. **人工/规则校验**：定期抽检 Agent 的输出，发现作弊模式后及时调整 reward 函数。

### Q3：如果没有足够的标注数据，RLHF 还能做吗？
**A**：能，但需要用一些替代方案。我们的做法是先构建一个**规则型 Reward Model** 作为冷启动。对于 Agent 任务，很多对错是可以被规则明确判断的（比如工具名对不对、JSON 格式对不对、最终答案对不对）。我们用这些规则生成了大量 pseudo-labels，先预训练一个初步的 RM，等积累了一定量的人工反馈后，再逐渐替换为真正的 human preference 数据。

### Q4：PPO 训练过程中最需要注意什么？
**A**：
1. **KL Divergence 监控**：如果 KL 值飙升，说明模型正在偏离基线，需要降低 learning rate 或加大 KL penalty；
2. **Reward 的尺度**：Reward 不能太大也不能太小，太大容易梯度爆炸，太小训练不动。我们一般把单步 reward 归一化到 [-1, 2] 的区间；
3. **Generation Length**：Agent 输出通常较长，需要控制好 `max_new_tokens`，避免生成无意义的循环内容。
