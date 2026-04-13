"""
Bradley-Terry奖励模型

基于Bradley-Terry模型的Pairwise奖励学习

核心公式：
P(y1 > y2 | x) = σ(r(x, y1) - r(x, y2))

损失函数：
L = -log σ(r(x, y_w) - r(x, y_l))
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, PreTrainedModel
from typing import Dict, Optional, Tuple


class RewardHead(nn.Module):
    """奖励头：将hidden state映射到标量奖励"""
    
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.dense = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )
        
        # 初始化（参考InstructGPT）
        for module in self.dense.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.zeros_(module.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
        Returns:
            rewards: [batch_size, 1]
        """
        # 取最后一个token的hidden state
        # 使用最后一个非padding位置的hidden state
        last_hidden = hidden_states[:, -1, :]  # [batch_size, hidden_size]
        reward = self.dense(last_hidden)  # [batch_size, 1]
        return reward


class BradleyTerryRewardModel(nn.Module):
    """
    Bradley-Terry奖励模型
    
    基于预训练语言模型，添加奖励头
    使用Pairwise Ranking Loss训练
    """
    
    def __init__(
        self,
        base_model_name: str,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # 加载基础模型
        from transformers import AutoConfig
        from peft import LoraConfig, get_peft_model, TaskType
        
        self.config = AutoConfig.from_pretrained(
            base_model_name,
            trust_remote_code=True,
        )
        
        # 修改配置添加dropout
        self.config.hidden_dropout_prob = dropout
        self.config.attention_probs_dropout_prob = dropout
        
        # 加载基础模型
        self.base_model = AutoModel.from_pretrained(
            base_model_name,
            config=self.config,
            trust_remote_code=True,
        )
        
        # 应用LoRA
        if use_lora:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=dropout,
                bias="none",
                task_type=TaskType.SEQ_CLS,
            )
            self.base_model = get_peft_model(self.base_model, lora_config)
        
        # 奖励头
        self.reward_head = RewardHead(self.config.hidden_size, dropout)
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_dict: bool = True,
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            rewards: [batch_size, 1]
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        
        hidden_states = outputs.last_hidden_state
        rewards = self.reward_head(hidden_states)
        
        return rewards
    
    def compute_pairwise_loss(
        self,
        chosen_input_ids: torch.Tensor,
        chosen_attention_mask: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        rejected_attention_mask: torch.Tensor,
        margin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算Pairwise Ranking Loss
        
        L = -log σ(r(x, y_w) - r(x, y_l) - margin)
        
        Args:
            chosen_input_ids: [batch_size, seq_len]
            chosen_attention_mask: [batch_size, seq_len]
            rejected_input_ids: [batch_size, seq_len]
            rejected_attention_mask: [batch_size, seq_len]
            margin: [batch_size] 偏好强度
            
        Returns:
            loss: 标量
            metrics: 训练指标
        """
        # 计算chosen的奖励
        chosen_rewards = self.forward(
            chosen_input_ids,
            chosen_attention_mask,
        )  # [batch_size, 1]
        
        # 计算rejected的奖励
        rejected_rewards = self.forward(
            rejected_input_ids,
            rejected_attention_mask,
        )  # [batch_size, 1]
        
        # Bradley-Terry loss
        reward_diff = chosen_rewards - rejected_rewards  # [batch_size, 1]
        
        # 应用margin
        if margin is not None:
            reward_diff = reward_diff - margin.unsqueeze(1)
        
        # Loss: -log σ(diff)
        loss = -F.logsigmoid(reward_diff).mean()
        
        # 计算准确率（chosen奖励 > rejected奖励的比例）
        with torch.no_grad():
            accuracy = (chosen_rewards > rejected_rewards).float().mean().item()
            chosen_reward_mean = chosen_rewards.mean().item()
            rejected_reward_mean = rejected_rewards.mean().item()
            reward_margin = (chosen_rewards - rejected_rewards).mean().item()
        
        metrics = {
            "loss": loss.item(),
            "accuracy": accuracy,
            "chosen_reward": chosen_reward_mean,
            "rejected_reward": rejected_reward_mean,
            "reward_margin": reward_margin,
        }
        
        return loss, metrics
    
    def get_reward(
        self,
        texts: list,
        batch_size: int = 8,
    ) -> torch.Tensor:
        """
        批量计算文本的奖励值
        
        Args:
            texts: 文本列表
            batch_size: 批大小
            
        Returns:
            rewards: [len(texts)]
        """
        self.eval()
        all_rewards = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                
                inputs = {k: v.to(self.base_model.device) for k, v in inputs.items()}
                
                rewards = self.forward(
                    inputs["input_ids"],
                    inputs["attention_mask"],
                )
                
                all_rewards.append(rewards.cpu())
        
        return torch.cat(all_rewards, dim=0).squeeze(-1)
    
    def save_pretrained(self, save_path: str):
        """保存模型"""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # 保存基础模型
        self.base_model.save_pretrained(save_path)
        
        # 保存奖励头
        torch.save(
            self.reward_head.state_dict(),
            os.path.join(save_path, "reward_head.pt")
        )
        
        # 保存tokenizer
        self.tokenizer.save_pretrained(save_path)
    
    @classmethod
    def load_pretrained(cls, load_path: str, **kwargs):
        """加载模型"""
        # 从config获取参数
        import os
        import json
        
        config_path = os.path.join(load_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            base_model_name = config.get("_name_or_path", load_path)
        else:
            base_model_name = load_path
        
        # 创建模型
        model = cls(base_model_name, **kwargs)
        
        # 加载奖励头
        reward_head_path = os.path.join(load_path, "reward_head.pt")
        if os.path.exists(reward_head_path):
            state_dict = torch.load(reward_head_path, map_location="cpu")
            model.reward_head.load_state_dict(state_dict)
        
        return model
