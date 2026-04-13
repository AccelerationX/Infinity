"""
严谨的配置系统 - RLHF三阶段配置
"""
from dataclasses import dataclass, field
from typing import Optional, List
import torch


@dataclass
class ModelConfig:
    """基础模型配置"""
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    trust_remote_code: bool = True
    torch_dtype: str = "float16"  # float16, bfloat16, float32
    
    # LoRA配置
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    lora_bias: str = "none"
    
    # 设备配置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    device_map: Optional[str] = "auto"
    
    @property
    def torch_dtype_enum(self):
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.torch_dtype, torch.float16)


@dataclass  
class SFTConfig:
    """
    Stage 1: 监督微调配置
    
    目标：学习专家示范，建立基础能力
    """
    # 训练配置
    output_dir: str = "./outputs/stage1_sft"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    
    # 优化器配置
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    
    # 序列配置
    max_seq_length: int = 512
    padding_side: str = "right"
    
    # 评估配置
    evaluation_strategy: str = "steps"
    eval_steps: int = 100
    save_steps: int = 100
    logging_steps: int = 10
    
    # 保存配置
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    
    # 其他
    seed: int = 42
    fp16: bool = True
    bf16: bool = False
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False


@dataclass
class RewardConfig:
    """
    Stage 2: 奖励模型训练配置
    
    目标：学习人类偏好，建立Bradley-Terry奖励模型
    """
    output_dir: str = "./outputs/stage2_reward"
    
    # 训练配置
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 2  # Pairwise需要更大显存
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    
    # 优化器配置
    learning_rate: float = 1e-5  # 通常比SFT更低
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    
    # 序列配置
    max_length: int = 512  # prompt + chosen/rejected
    
    # 评估配置
    evaluation_strategy: str = "steps"
    eval_steps: int = 50
    save_steps: int = 50
    logging_steps: int = 5
    
    # Bradley-Terry模型配置
    bt_margin: float = 0.0  # 偏好强度阈值
    bt_temperature: float = 1.0
    
    # 保存配置
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_accuracy"
    
    seed: int = 42
    fp16: bool = True


@dataclass
class PPOConfig:
    """
    Stage 3: PPO训练配置
    
    目标：基于奖励模型优化策略
    """
    output_dir: str = "./outputs/stage3_ppo"
    
    # PPO算法配置
    ppo_epochs: int = 1  # 每次采集数据后更新的轮数
    num_rollouts: int = 512  # 总采集轨迹数
    rollout_batch_size: int = 16  # 每次采集的批量
    step_batch_size: int = 8  # PPO更新的批量
    
    # PPO超参数
    gamma: float = 0.99  # 折扣因子
    lam: float = 0.95  # GAE参数
    clip_eps: float = 0.2  # PPO裁剪阈值
    kl_coef: float = 0.2  # KL散度惩罚系数
    kl_target: Optional[float] = 6.0  # 自适应KL目标（可选）
    
    # 价值函数配置
    vf_coef: float = 0.5  # 价值损失系数
    vf_clip: float = 0.2  # 价值函数裁剪
    
    # 熵配置
    entropy_coef: float = 0.01  # 熵奖励系数
    
    # 优化器配置
    learning_rate: float = 1e-6  # PPO通常使用更低的学习率
    lr_scheduler_type: str = "constant"
    
    # 生成配置
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    
    # 序列配置
    max_seq_length: int = 512
    
    # 日志配置
    logging_steps: int = 1
    save_steps: int = 50
    
    seed: int = 42


@dataclass
class RLHFConfig:
    """
    完整的RLHF配置
    
    包含三阶段的所有配置
    """
    # 基础模型配置（三阶段共享）
    model: ModelConfig = field(default_factory=ModelConfig)
    
    # 三阶段配置
    sft: SFTConfig = field(default_factory=SFTConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    
    # 阶段控制
    run_sft: bool = True
    run_reward: bool = True
    run_ppo: bool = True
    
    # 阶段间依赖
    sft_model_path: Optional[str] = None  # SFT输出路径（用于Reward和PPO初始化）
    reward_model_path: Optional[str] = None  # Reward模型路径（用于PPO）
    
    # 全局输出目录
    output_root: str = "./rlhf_outputs"
    
    def __post_init__(self):
        """自动设置阶段间的依赖路径"""
        if self.sft_model_path is None:
            self.sft_model_path = self.sft.output_dir
        if self.reward_model_path is None:
            self.reward_model_path = self.reward.output_dir
