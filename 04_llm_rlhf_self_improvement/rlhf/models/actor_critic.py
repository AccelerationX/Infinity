"""
Actor-Critic模型

标准的Actor-Critic架构用于PPO训练
- Actor: 策略网络，生成动作
- Critic: 价值网络，估计状态价值
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Tuple, Optional
import copy


class CriticHead(nn.Module):
    """
    价值头：估计状态价值V(s)
    
    输出标量价值估计
    """
    
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.dense = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )
        
        # 初始化
        for module in self.dense.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.zeros_(module.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
        Returns:
            values: [batch_size, 1]
        """
        # 取最后一个token的hidden state
        last_hidden = hidden_states[:, -1, :]  # [batch_size, hidden_size]
        value = self.dense(last_hidden)  # [batch_size, 1]
        return value


class ActorCriticModel(nn.Module):
    """
    Actor-Critic模型
    
    Actor: 语言模型，输出动作分布
    Critic: 价值头，估计状态价值
    
    同时维护一个参考模型（ref_model）用于KL约束
    """
    
    def __init__(
        self,
        base_model_name: str,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        dropout: float = 0.1,
        device: str = "cuda",
    ):
        super().__init__()
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # 加载Actor（策略网络）
        from .lora_utils import load_model_with_lora
        
        self.actor, self.tokenizer = load_model_with_lora(
            base_model_name,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=dropout,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        
        # 创建Critic（价值网络）
        # Critic独立训练，不共享LoRA参数
        hidden_size = self.actor.config.hidden_size
        self.critic = CriticHead(hidden_size, dropout)
        
        # 参考模型（用于KL约束）
        self.ref_model = None
        
        self.to(self.device)
    
    def init_reference_model(self):
        """初始化参考模型（冻结，用于KL约束）"""
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.actor.config._name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
        )
        self.ref_model.to(self.device)
        self.ref_model.eval()
        
        # 冻结参考模型
        for param in self.ref_model.parameters():
            param.requires_grad = False
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Actor和Critic的前向传播
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            {
                "logits": [batch_size, seq_len, vocab_size]
                "values": [batch_size, 1]
            }
        """
        # Actor前向（获取hidden states和logits）
        actor_outputs = self.actor(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        
        logits = actor_outputs.logits  # [batch_size, seq_len, vocab_size]
        hidden_states = actor_outputs.hidden_states[-1]  # 最后一层
        
        # Critic前向（估计价值）
        values = self.critic(hidden_states)  # [batch_size, 1]
        
        return {
            "logits": logits,
            "values": values,
        }
    
    def get_action_and_value(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_ref_logprob: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        获取动作和价值
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            return_ref_logprob: 是否返回参考模型的logprob
            
        Returns:
            {
                "action_logits": [...]
                "values": [...]
                "ref_logprobs": [...] (optional)
            }
        """
        # 前向传播
        outputs = self.forward(input_ids, attention_mask)
        logits = outputs["logits"]
        values = outputs["values"]
        
        result = {
            "logits": logits,
            "values": values,
        }
        
        # 获取参考模型的logprob（用于KL约束）
        if return_ref_logprob and self.ref_model is not None:
            with torch.no_grad():
                ref_outputs = self.ref_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                ref_logits = ref_outputs.logits
                
                # 计算log概率
                ref_logprobs = torch.log_softmax(ref_logits, dim=-1)
                
                # 收集实际token的logprob
                # shift: logits对应下一个token的预测
                ref_logprobs = ref_logprobs[:, :-1, :].gather(
                    dim=-1,
                    index=input_ids[:, 1:].unsqueeze(-1)
                ).squeeze(-1)
                
                result["ref_logprobs"] = ref_logprobs
        
        return result
    
    def generate(
        self,
        prompts: list,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> Dict[str, list]:
        """
        生成响应
        
        Args:
            prompts: prompt列表
            max_new_tokens: 最大生成token数
            temperature: 温度
            top_p: top-p采样
            
        Returns:
            {
                "responses": [...],
                "sequences": [...],
            }
        """
        # Tokenize prompts
        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 生成
        with torch.no_grad():
            outputs = self.actor.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        
        sequences = outputs.sequences
        
        # 解码响应
        responses = self.tokenizer.batch_decode(
            sequences,
            skip_special_tokens=True
        )
        
        return {
            "responses": responses,
            "sequences": sequences,
            "prompt_length": inputs["input_ids"].shape[1],
        }
    
    def compute_logprobs(
        self,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算序列的log概率
        
        Args:
            sequences: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            logprobs: [batch_size, seq_len-1]
        """
        outputs = self.actor(
            input_ids=sequences,
            attention_mask=attention_mask,
        )
        
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        
        # log softmax
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        
        # 收集实际token的logprob
        target_ids = sequences[:, 1:]
        token_logprobs = log_probs.gather(
            dim=-1,
            index=target_ids.unsqueeze(-1)
        ).squeeze(-1)
        
        return token_logprobs
    
    def save_pretrained(self, save_path: str):
        """保存模型"""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # 保存Actor
        self.actor.save_pretrained(os.path.join(save_path, "actor"))
        
        # 保存Critic
        torch.save(
            self.critic.state_dict(),
            os.path.join(save_path, "critic.pt")
        )
        
        # 保存tokenizer
        self.tokenizer.save_pretrained(save_path)
    
    def load_pretrained(self, load_path: str):
        """加载模型"""
        import os
        from transformers import AutoModelForCausalLM
        
        # 加载Actor
        actor_path = os.path.join(load_path, "actor")
        self.actor = AutoModelForCausalLM.from_pretrained(
            actor_path,
            trust_remote_code=True,
        )
        self.actor.to(self.device)
        
        # 加载Critic
        critic_path = os.path.join(load_path, "critic.pt")
        if os.path.exists(critic_path):
            critic_state = torch.load(critic_path, map_location=self.device)
            self.critic.load_state_dict(critic_state)
