# RLHF Agent 自我改进系统 - 设计文档

> 从第一性原理理解 RLHF 和 Agent 自我改进

---

## 1. 项目背景与问题定义

### 1.1 为什么要做 Agent 自我改进？

**传统 LLM 的局限：**
- 预训练知识是静态的，无法从交互中学习
- SFT（监督微调）需要大量标注数据
- 人类反馈收集成本高、速度慢

**Agent 场景的特殊性：**
- Agent 可以与环境交互，获得即时的执行反馈
- 成功/失败信号可以自动获取（如代码是否运行成功）
- 相同类型的任务会反复出现，有学习价值

### 1.2 核心问题

如何让 Agent 能够：
1. 从自己的成功/失败案例中学习
2. 不需要人工标注每个样本
3. 避免灾难性遗忘
4. 在有限计算资源下高效训练

---

## 2. 理论基础

### 2.1 RLHF 核心思想

**从人类反馈中学习奖励函数：**

传统 RL: 奖励函数 r(s,a) 是人工设计的
RLHF: 奖励函数 r_φ(s,a) 是从人类偏好中学习得到的

**为什么不用人类直接标注奖励？**
- 人类擅长比较（A比B好），不擅长绝对评分（给A打8.5分）
- 评分标准会随时间漂移

### 2.2 为什么要用 PPO？

**策略梯度的问题：**
- 梯度方差大，训练不稳定
- 步长敏感，容易崩溃

**PPO 的解决方案：**
- 裁剪目标函数限制策略更新幅度
- 重要性采样支持离线数据复用
- 在实践中更稳定、更易调参

### 2.3 LoRA 原理

**全量微调的问题：**
- LLM 参数量大（7B-70B），显存需求高
- 训练速度慢
- 容易过拟合

**LoRA 的核心思想：**

对于权重矩阵 W ∈ R^{d×k}，不直接更新 W，而是引入低秩分解：

```
W' = W + ΔW = W + BA

其中:
- B ∈ R^{d×r}
- A ∈ R^{r×k}
- r << min(d, k) (通常 r=8,16)
```

**为什么有效？**
- 预训练权重已经捕获了丰富的特征
- 微调只需要在特定方向上做小调整
- 低秩矩阵足以表示这些调整

---

## 3. 系统设计

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Self-Improvement Loop                    │
├─────────────────────────────────────────────────────────────┤
│  Phase 1: Experience Collection                             │
│    ├─ Policy.generate() → actions                           │
│    ├─ Environment.execute() → results                       │
│    └─ RewardModel.compute() → rewards                       │
│                                                              │
│  Phase 2: Experience Storage                                │
│    ├─ ReplayBuffer.add()                                    │
│    ├─ Priority update (based on TD error)                   │
│    └─ Success case distillation                             │
│                                                              │
│  Phase 3: Policy Update                                     │
│    ├─ ReplayBuffer.sample()                                 │
│    ├─ GAE advantage estimation                              │
│    ├─ PPO loss computation                                  │
│    └─ Gradient update with LoRA                             │
│                                                              │
│  Phase 4: Evaluation                                        │
│    ├─ Success rate tracking                                 │
│    ├─ Convergence detection                                 │
│    └─ Checkpoint saving                                     │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 关键设计决策

#### 决策 1: 冷启动策略（规则奖励 vs 学习奖励）

**问题**：初始时没有训练好的奖励模型

**方案**：
- Phase 1: 使用规则奖励冷启动
- Phase 2: 收集足够数据后，训练神经网络奖励模型
- Phase 3: 切换到学习奖励模型继续训练

**为什么这样设计？**
- 规则奖励虽然粗糙，但能提供初步信号
- 避免冷启动时的"鸡生蛋蛋生鸡"问题

#### 决策 2: 经验回放缓冲区设计

**关键功能：**
1. **优先回放**：TD 误差大的样本优先采样
2. **成功/失败平衡**：避免成功样本过多导致乐观偏差
3. **成功案例蒸馏**：单独存储高质量成功案例

**采样策略：**
```python
def sample_balanced(batch_size):
    n_success = batch_size // 2
    n_failure = batch_size - n_success
    
    success_batch = sample_by_priority(success_buffer, n_success)
    failure_batch = sample_by_priority(failure_buffer, n_failure)
    
    return shuffle(success_batch + failure_batch)
```

#### 决策 3: KL 散度约束

**问题**：策略更新后可能偏离原始模型太远，导致模式崩溃或生成质量下降

**解决方案**：
```
L_total = L_PPO + β * KL(π_θ || π_ref)
```

其中 π_ref 是冻结的参考模型（初始策略）

**作用**：
- 保持语言模型的通用能力
- 防止策略过度优化奖励（reward hacking）
- 提高训练稳定性

#### 决策 4: 优势归一化

**为什么重要？**
- 不同任务的奖励尺度可能不同
- 归一化后梯度更稳定

**实现：**
```python
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

---

## 4. 算法细节

### 4.1 GAE (Generalized Advantage Estimation)

**直观理解：**
- 只看一步的 reward（蒙特卡洛）→ 方差大
- 只看 value function 的差分（TD）→ 偏差大
- GAE：平衡两者

**数学推导：**

```
GAE(γ, λ): Â_t = Σ_{l=0}^∞ (γλ)^l δ_{t+l}

其中 δ_t = r_t + γV(s_{t+1}) - V(s_t)

当 λ=0: Â_t = δ_t (TD error，低方差高偏差)
当 λ=1: Â_t = Σ γ^l r_{t+l} - V(s_t) (蒙特卡洛，高方差低偏差)
```

**实现代码：**
```python
def compute_gae(rewards, values, masks, gamma=0.99, lam=0.95):
    advantages = torch.zeros_like(rewards)
    last_gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * masks[t] - values[t]
        advantages[t] = last_gae = delta + gamma * lam * masks[t] * last_gae
    
    return advantages
```

### 4.2 PPO 裁剪目标

**为什么裁剪？**

标准策略梯度：
```
∇J = E[∇log π_θ(a|s) * A(s,a)]
```

问题：梯度步长过大时，策略可能崩溃

PPO 解决方案：
```
L^{CLIP}(θ) = E[min(
    r_t(θ) * A_t,
    clip(r_t(θ), 1-ε, 1+ε) * A_t
)]

其中 r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
```

**两种情况：**
1. A_t > 0（动作好）：限制 r_t 不超过 1+ε，防止过度优化
2. A_t < 0（动作差）：限制 r_t 不小于 1-ε，防止过度抑制

### 4.3 LoRA 训练细节

**初始化：**
- A 矩阵用高斯初始化
- B 矩阵初始化为零 → 初始 ΔW = 0 → 从预训练模型开始

**可训练参数选择：**
- 只训练 attention 层的 q_proj, v_proj
- 不训练 feed-forward 层（节省显存）

**代码实现：**
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()  # 显示可训练参数比例
```

---

## 5. 工程实现要点

### 5.1 内存优化

**问题**：LLM 训练显存需求大

**解决方案：**
1. **LoRA**: 只训练少量参数
2. **梯度累积**: 小 batch size，多步累积
3. **混合精度**: FP16/BF16 训练
4. **梯度检查点**: 时间换空间

**显存计算（以 7B 模型为例）：**

| 配置 | 显存占用 |
|------|---------|
| 全量微调 FP32 | ~112GB |
| 全量微调 FP16 | ~56GB |
| LoRA FP16 | ~16GB |
| LoRA + 8-bit | ~8GB |

### 5.2 训练稳定性

**常见问题及解决：**

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| 奖励爆炸 | 策略过度优化 | KL 惩罚、奖励裁剪 |
| 模式崩溃 | 策略坍缩到单一模式 | 熵奖励、多样性奖励 |
| 训练震荡 | 学习率过大 | 学习率衰减、梯度裁剪 |
| 遗忘 | 新任务覆盖旧知识 | EWC、经验回放 |

### 5.3 评估指标

**核心指标：**
1. **成功率**: task_success / total_episodes
2. **平均奖励**: mean(rewards)
3. **策略熵**: 衡量策略的随机性（探索程度）
4. **KL 散度**: 与参考模型的偏离程度

**训练曲线：**
- 奖励应该单调上升
- KL 散度应该保持在合理范围（如 0.1-0.5）
- 成功率应该逐渐提高

---

## 6. 局限与改进方向

### 6.1 当前局限

1. **稀疏奖励**: 某些任务只有最终有奖励，中间过程无信号
2. **样本效率**: 需要大量交互才能收敛
3. **泛化能力**: 学到的策略可能过拟合到训练任务分布

### 6.2 改进方向

#### 方向 1: 密集奖励塑造

**问题**：稀疏奖励导致学习慢

**方案**：
- 中间步骤奖励（如代码编译通过给部分奖励）
- 基于规则的部分奖励
- 使用专家演示进行初始化（行为克隆）

#### 方向 2: 元学习 (Meta-Learning)

**目标**：快速适应新任务

**方案**：
- MAML: 学习好的初始化参数
- Reptile: 简单的元学习算法
- 上下文学习：让模型学习"如何学习"

#### 方向 3: 离线强化学习

**问题**：在线交互成本高

**方案**：
- 使用历史数据离线训练
- CQL (Conservative Q-Learning) 避免分布外动作
- AWAC (Accelerated Offline RL)

#### 方向 4: 多任务与迁移

**目标**：学到的能力可以迁移到新任务

**方案**：
- 任务嵌入：学习任务表示
- 模块化策略：不同子任务用不同模块
- 课程学习：从简单任务到复杂任务

---

## 7. 与相关工作的对比

| 工作 | 核心方法 | 与我们的区别 |
|------|---------|-------------|
| InstructGPT | RLHF 训练 GPT-3 | 我们专注于 Agent 场景，强调环境交互反馈 |
| AutoGPT | 链式调用工具 | 我们没有改进机制，AutoGPT 不会从错误学习 |
| Voyager | 技能库 + 代码生成 | 我们使用 RL 优化策略，Voyager 是启发式 |
| MetaGPT | SOP 驱动 | 我们使用 RL 学习策略，不是固定 SOP |

**我们的定位**：轻量级的 Agent 自我改进框架，强调实用性和可扩展性。

---

## 8. 面试常问问题

### Q: 为什么用 PPO 而不是其他 RL 算法？

**A:**
1. PPO 在实践中更稳定，对超参数不敏感
2. 支持离线数据复用（重要性采样）
3. 实现简单，易于调试
4. 在 LLM fine-tuning 场景有成功案例（InstructGPT）

### Q: LoRA 为什么只训练部分层？

**A:**
1. Attention 层的 q, v 投影对任务适应最重要
2. 减少可训练参数量，降低过拟合风险
3. 节省显存和计算
4. 实践中效果与全量微调接近

### Q: 如何处理奖励稀疏问题？

**A:**
1. 使用 GAE 进行多步引导信号传播
2. 设计密集奖励（如代码编译通过给部分奖励）
3. 使用专家演示进行热启动（行为克隆）
4. 分层 RL：学习高层策略分解任务

### Q: 防止灾难性遗忘的方法？

**A:**
1. 经验回放：保留旧任务的样本
2. KL 约束：防止策略偏离太远
3. EWC (Elastic Weight Consolidation)：保护重要参数
4. 模块化：不同任务用不同子网络

---

## 附录：关键公式速查

**PPO 目标：**
```
L^{CLIP}(θ) = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]
```

**GAE：**
```
Â_t = Σ_{l=0}^∞ (γλ)^l δ_{t+l}
```

**总损失：**
```
L_total = L^{CLIP} - c1 * L^{VF} + c2 * H(π) + β * KL(π || π_ref)
```

**LoRA：**
```
W' = W + BA,  B ∈ R^{d×r}, A ∈ R^{r×k}
```

---

*Document Version: 1.0*
*Last Updated: 2026-04-13*
