"""
配置文件 - RLHF系统配置
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class RLHFConfig:
    """RLHF系统整体配置"""
    
    # 模型配置
    base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"  # 基础模型
    use_lora: bool = True  # 使用LoRA高效微调
    lora_rank: int = 8
    lora_alpha: int = 16
    
    # 奖励模型配置
    reward_model_path: Optional[str] = None  # 预训练奖励模型路径
    reward_dropout: float = 0.1
    
    # PPO配置
    ppo_epochs: int = 4
    ppo_clip_epsilon: float = 0.2
    ppo_value_clip: float = 0.4
    ppo_entropy_coef: float = 0.01
    ppo_value_coef: float = 0.5
    
    # 训练配置
    learning_rate: float = 1e-5
    batch_size: int = 4
    mini_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 512
    max_new_tokens: int = 256
    
    # 经验回放配置
    replay_buffer_size: int = 10000
    priority_alpha: float = 0.6  # 优先回放指数
    priority_beta: float = 0.4   # 重要性采样指数
    
    # 自我改进配置
    improvement_iterations: int = 10
    episodes_per_iteration: int = 20
    success_threshold: float = 0.7  # 成功判定阈值
    kl_penalty_coef: float = 0.2  # KL散度惩罚系数
    
    # 评估配置
    eval_interval: int = 2
    eval_episodes: int = 10
    
    # 输出配置
    output_dir: str = "./rlhf_outputs"
    checkpoint_interval: int = 5
    
    # 设备配置
    device: str = "cuda"  # cuda/cpu
    mixed_precision: bool = True  # 混合精度训练
