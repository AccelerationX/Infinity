"""
奖励模型 (Reward Model)
评估Agent输出的质量，为PPO训练提供奖励信号
"""
import torch
import torch.nn as nn
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class RewardConfig:
    """奖励模型配置"""
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dropout: float = 0.1
    use_lora: bool = True
    lora_rank: int = 8
    lora_alpha: int = 16
    device: str = "cuda"


class RewardModel(nn.Module):
    """
    奖励模型
    
    基于预训练语言模型，在最后一层添加价值头
    输出标量奖励值
    """
    
    def __init__(self, config: RewardConfig):
        super().__init__()
        self.config = config
        
        # 加载预训练模型
        from transformers import AutoModel, AutoTokenizer, AutoConfig
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型配置
        model_config = AutoConfig.from_pretrained(
            config.model_name,
            trust_remote_code=True
        )
        model_config.hidden_dropout_prob = config.dropout
        model_config.attention_probs_dropout_prob = config.dropout
        
        # 加载基础模型
        self.base_model = AutoModel.from_pretrained(
            config.model_name,
            config=model_config,
            trust_remote_code=True
        )
        
        # 添加LoRA
        if config.use_lora:
            from peft import get_peft_model, LoraConfig, TaskType
            
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=0.1,
                bias="none",
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            )
            self.base_model = get_peft_model(self.base_model, lora_config)
        
        # 奖励头 - 将hidden state映射到标量奖励
        hidden_size = model_config.hidden_size
        self.reward_head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_size // 2, 1),
        )
        
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            rewards: [batch_size, 1]
        """
        # 获取最后一层hidden state
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        
        # 使用最后一个token的hidden state
        # outputs.last_hidden_state: [batch_size, seq_len, hidden_size]
        last_hidden = outputs.last_hidden_state
        
        # 找到每个序列的实际最后一个token（考虑padding）
        # 使用attention_mask找到最后一个非padding位置
        seq_lengths = attention_mask.sum(dim=1) - 1  # [batch_size]
        batch_size = input_ids.size(0)
        
        # 提取每个序列最后一个token的hidden state
        last_token_hidden = last_hidden[
            torch.arange(batch_size, device=last_hidden.device),
            seq_lengths
        ]  # [batch_size, hidden_size]
        
        # 通过奖励头得到标量奖励
        rewards = self.reward_head(last_token_hidden)  # [batch_size, 1]
        
        return rewards
    
    def get_rewards(
        self,
        queries: List[str],
        responses: List[str],
    ) -> torch.Tensor:
        """
        获取奖励分数
        
        Args:
            queries: 查询列表
            responses: 回答列表
            
        Returns:
            rewards: 奖励值tensor [batch_size]
        """
        # 拼接query和response
        texts = [f"{q}\n{r}" for q, r in zip(queries, responses)]
        
        # Tokenize
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        
        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)
        
        with torch.no_grad():
            rewards = self.forward(input_ids, attention_mask)
        
        return rewards.squeeze(-1)  # [batch_size]
    
    def save_pretrained(self, save_path: str):
        """保存模型"""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # 保存基础模型
        self.base_model.save_pretrained(save_path)
        
        # 保存奖励头
        torch.save(self.reward_head.state_dict(), os.path.join(save_path, "reward_head.pt"))
        
        # 保存tokenizer
        self.tokenizer.save_pretrained(save_path)
        
        # 保存配置
        torch.save(self.config, os.path.join(save_path, "reward_config.pt"))
    
    @classmethod
    def load_pretrained(cls, load_path: str):
        """加载模型"""
        config = torch.load(os.path.join(load_path, "reward_config.pt"))
        model = cls(config)
        
        # 加载奖励头
        reward_head_state = torch.load(os.path.join(load_path, "reward_head.pt"))
        model.reward_head.load_state_dict(reward_head_state)
        
        return model


class RuleBasedRewardModel:
    """
    基于规则的奖励模型（用于冷启动）
    
    在没有训练好的奖励模型时，使用规则计算奖励
    """
    
    def __init__(self):
        self.rules = []
    
    def add_rule(self, rule_func, weight: float = 1.0):
        """添加规则"""
        self.rules.append((rule_func, weight))
    
    def get_rewards(
        self,
        queries: List[str],
        responses: List[str],
        task_results: Optional[List[Dict]] = None,
    ) -> torch.Tensor:
        """
        基于规则计算奖励
        
        Args:
            queries: 查询列表
            responses: 回答列表
            task_results: 任务执行结果（可选）
            
        Returns:
            rewards: 奖励值tensor
        """
        rewards = []
        
        for i, (query, response) in enumerate(zip(queries, responses)):
            reward = 0.0
            task_result = task_results[i] if task_results else None
            
            # 应用所有规则
            for rule_func, weight in self.rules:
                r = rule_func(query, response, task_result)
                reward += weight * r
            
            rewards.append(reward)
        
        return torch.tensor(rewards, dtype=torch.float32)


# 预定义规则函数
def format_correctness_rule(query: str, response: str, task_result: Dict) -> float:
    """格式正确性规则"""
    # 检查是否有明确的输出格式
    if "```" in response or "###" in response:
        return 1.0
    return 0.5


def task_success_rule(query: str, response: str, task_result: Dict) -> float:
    """任务成功规则"""
    if task_result is None:
        return 0.0
    
    success = task_result.get("success", False)
    return 1.0 if success else 0.0


def execution_time_rule(query: str, response: str, task_result: Dict) -> float:
    """执行时间规则（越快越好）"""
    if task_result is None:
        return 0.0
    
    execution_time = task_result.get("execution_time", 10.0)
    # 归一化到0-1，假设10秒是正常，1秒是优秀
    score = max(0.0, 1.0 - execution_time / 10.0)
    return score


def create_default_rule_based_reward() -> RuleBasedRewardModel:
    """创建默认的规则奖励模型"""
    model = RuleBasedRewardModel()
    model.add_rule(task_success_rule, weight=0.6)  # 任务成功最重要
    model.add_rule(format_correctness_rule, weight=0.2)
    model.add_rule(execution_time_rule, weight=0.2)
    return model
