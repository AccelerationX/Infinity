# 基于 RLHF 的 Agent 自我改进系统 v2.0

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **这是一个严谨、完整的RLHF实现，追求研究质量而非花架子。**

标准的RLHF三阶段训练流程，包含Bradley-Terry奖励模型和Actor-Critic PPO实现。

## 核心特性

- ✅ **标准RLHF三阶段流程**
  - Stage 1: SFT (Supervised Fine-Tuning)
  - Stage 2: Reward Model (Bradley-Terry Pairwise Training)
  - Stage 3: RL Fine-tuning (Actor-Critic PPO)

- ✅ **严谨的奖励模型**
  - Bradley-Terry模型: P(y1 > y2) = σ(r(y1) - r(y2))
  - Pairwise Ranking Loss: -log σ(r(y_w) - r(y_l))
  - 完整的人类偏好学习

- ✅ **标准的Actor-Critic PPO**
  - 独立的Actor（策略）和Critic（价值）网络
  - GAE (Generalized Advantage Estimation)
  - PPO裁剪目标函数
  - KL散度约束

- ✅ **LoRA高效微调**
  - 参数高效微调（< 1%可训练参数）
  - 显存优化

- ✅ **真实的Agent环境**
  - 代码执行沙箱
  - 工具调用支持
  - 安全的执行环境

## 与简版实现的区别

| 组件 | 简版实现 (v1) | 严谨实现 (v2) |
|------|--------------|---------------|
| **奖励模型** | 简单规则/通用模型 | **Bradley-Terry + Pairwise训练** |
| **PPO** | 简化版，无Value Network | **标准Actor-Critic + GAE** |
| **RLHF流程** | 单阶段 | **完整三阶段流程** |
| **Agent环境** | 简单正则匹配 | **真实代码执行沙箱** |
| **理论基础** | 概念性实现 | **符合论文标准** |

## 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                     RLHF Training Pipeline                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Stage 1: SFT                                                   │
│  ├─ Input: Demonstration Data (prompt, completion)              │
│  ├─ Model: CausalLM + LoRA                                      │
│  ├─ Loss: CrossEntropy on completion tokens                     │
│  └─ Output: SFT Model (policy initialization)                   │
│                              ↓                                  │
│  Stage 2: Reward Model                                          │
│  ├─ Input: Preference Data (prompt, chosen, rejected)           │
│  ├─ Model: BradleyTerryRewardModel                              │
│  │   ├─ Base: CausalLM + LoRA                                   │
│  │   └─ Head: RewardHead (hidden → 1)                           │
│  ├─ Loss: -log σ(r(x,y_w) - r(x,y_l))                           │
│  └─ Output: Reward Model (r_φ)                                  │
│                              ↓                                  │
│  Stage 3: PPO                                                   │
│  ├─ Input: Task Prompts                                         │
│  ├─ Model: ActorCriticModel                                     │
│  │   ├─ Actor: CausalLM + LoRA (policy π_θ)                    │
│  │   ├─ Critic: CriticHead (value V)                           │
│  │   └─ Ref: Frozen reference model (KL constraint)             │
│  ├─ Algorithm:                                                  │
│  │   1. Collect rollouts (query → response → reward)            │
│  │   2. Compute GAE advantages                                  │
│  │   3. Update with PPO clipped objective                       │
│  │   4. KL penalty: β·KL(π_θ || π_ref)                          │
│  └─ Output: RL Policy (optimized π_θ*)                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 快速开始

### 安装依赖

```bash
pip install torch transformers peft accelerate
```

### 运行示例

```bash
python examples/code_generation_example.py
```

## 核心组件详解

### 1. Bradley-Terry 奖励模型

基于Bradley-Terry模型学习人类偏好：

```python
from rlhf.models.reward_model import BradleyTerryRewardModel

reward_model = BradleyTerryRewardModel(
    base_model_name="Qwen/Qwen2.5-0.5B-Instruct",
    use_lora=True,
)

# Pairwise loss
loss, metrics = reward_model.compute_pairwise_loss(
    chosen_input_ids,
    chosen_attention_mask,
    rejected_input_ids,
    rejected_attention_mask,
)
```

**数学原理：**

```
P(y_w > y_l | x) = σ(r_φ(x, y_w) - r_φ(x, y_l))

Loss = -E[log σ(r_φ(x, y_w) - r_φ(x, y_l))]
```

### 2. Actor-Critic PPO

标准的Actor-Critic架构：

```python
from rlhf.models.actor_critic import ActorCriticModel

actor_critic = ActorCriticModel(
    base_model_name="Qwen/Qwen2.5-0.5B-Instruct",
    use_lora=True,
)

# Actor: generates actions (text)
# Critic: estimates state value V(s)
# Ref: frozen reference model for KL constraint
```

**PPO目标函数：**

```
L^{CLIP}(θ) = E[min(r_t(θ)·A_t, clip(r_t(θ), 1-ε, 1+ε)·A_t)]

其中:
- r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)
- A_t: GAE advantage estimate
- ε: clip parameter (default 0.2)
```

### 3. GAE 优势估计

```
Â_t = Σ_{l=0}^∞ (γλ)^l δ_{t+l}

δ_t = r_t + γV(s_{t+1}) - V(s_t)
```

### 4. 代码执行环境

安全的Python代码执行沙箱：

```python
from rlhf.environment.code_env import CodeExecutionEnv

env = CodeExecutionEnv(
    timeout=5.0,
    allow_imports=['math', 'random'],
)

result = env.execute("""
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)

result = factorial(5)
""")

# result: {success, output, error, execution_time}
```

## 完整训练流程

```python
from rlhf import RLHFConfig, RLHFPipeline
from rlhf.data.schemas import DemonstrationData, PreferenceData

# 准备数据
demonstration_data = [
    DemonstrationData(prompt="...", completion="..."),
    # ...
]

preference_data = [
    PreferenceData(prompt="...", chosen="...", rejected="..."),
    # ...
]

# 配置
config = RLHFConfig(
    model=ModelConfig(model_name="Qwen/Qwen2.5-0.5B-Instruct"),
    sft=SFTConfig(num_train_epochs=3),
    reward=RewardConfig(num_train_epochs=2),
    ppo=PPOConfig(num_rollouts=512),
)

# 执行三阶段训练
pipeline = RLHFPipeline(config)
results = pipeline.run(
    demonstration_data=demonstration_data,
    preference_data=preference_data,
)
```

## 项目结构

```
04_llm_rlhf_self_improvement/
├── rlhf/                       # 核心包
│   ├── config.py              # 配置系统
│   ├── data/                  # 数据管理
│   │   ├── schemas.py         # Demonstration/Preference/Trajectory
│   │   ├── datasets.py        # PyTorch Datasets
│   │   └── collators.py       # Data collators
│   ├── models/                # 模型定义
│   │   ├── reward_model.py    # BradleyTerryRewardModel
│   │   ├── actor_critic.py    # ActorCriticModel
│   │   └── lora_utils.py      # LoRA工具
│   ├── training/              # 训练流程
│   │   ├── sft_trainer.py     # Stage 1: SFT
│   │   ├── reward_trainer.py  # Stage 2: Reward Model
│   │   └── ppo_trainer.py     # Stage 3: PPO
│   ├── environment/           # Agent环境
│   │   └── code_env.py        # 代码执行沙箱
│   └── pipeline.py            # 完整Pipeline
├── examples/                  # 示例
│   └── code_generation_example.py
├── DESIGN.md                  # 详细设计文档
└── README.md                  # 本文档
```

## 关键算法公式速查

**Bradley-Terry Loss:**
```
L_BT = -log σ(r_φ(x, y_w) - r_φ(x, y_l))
```

**PPO Clipped Objective:**
```
L^{CLIP} = min(r·A, clip(r, 1-ε, 1+ε)·A)
r = π_θ / π_old
```

**GAE:**
```
Â_t = Σ (γλ)^l δ_{t+l}
δ_t = r_t + γV_{t+1} - V_t
```

**Total PPO Loss:**
```
L_total = L^{CLIP} - c_1·L^{VF} + c_2·H(π) + β·KL(π||π_ref)
```

## 参考论文

1. [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (InstructGPT, Ouyang et al., 2022)
2. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) (PPO, Schulman et al., 2017)
3. [Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593) (Ziegler et al., 2019)
4. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (Hu et al., 2021)

## License

MIT License

---

**郑重声明**：这是一个追求研究质量的严谨实现，不是花架子。所有核心算法均符合论文标准，可用于真实研究。
