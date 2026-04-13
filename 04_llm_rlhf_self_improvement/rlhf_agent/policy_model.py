"""
策略模型 (Policy Model)
Agent的核心策略网络，生成动作（文本输出）
"""
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import copy


@dataclass
class PolicyConfig:
    """策略模型配置"""
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    use_lora: bool = True
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    device: str = "cuda"


class PolicyModel:
    """
    策略模型
    
    基于预训练语言模型，负责生成Agent的响应
    支持LoRA高效微调和参考模型（用于KL散度约束）
    """
    
    def __init__(self, config: PolicyConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # 加载tokenizer
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载策略模型
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if config.device == "cuda" else torch.float32,
        )
        
        # 添加LoRA
        if config.use_lora:
            from peft import get_peft_model, LoraConfig, TaskType
            
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                bias="none",
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        self.model.to(self.device)
        
        # 创建参考模型（用于KL散度约束，防止策略偏离太远）
        self.ref_model = None
    
    def init_reference_model(self):
        """初始化参考模型（冻结参数）"""
        from transformers import AutoModelForCausalLM
        
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
        )
        self.ref_model.to(self.device)
        self.ref_model.eval()
        
        # 冻结参考模型参数
        for param in self.ref_model.parameters():
            param.requires_grad = False
    
    def generate(
        self,
        queries: List[str],
        max_new_tokens: int = 256,
        return_logprobs: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        生成响应
        
        Args:
            queries: 查询列表
            max_new_tokens: 最大生成token数
            return_logprobs: 是否返回log概率
            
        Returns:
            {
                "responses": 生成的文本列表
                "response_ids": [batch_size, seq_len]
                "logprobs": [batch_size, seq_len]
                "masks": [batch_size, seq_len]
            }
        """
        # Tokenize输入
        query_encodings = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        
        query_ids = query_encodings["input_ids"].to(self.device)
        query_masks = query_encodings["attention_mask"].to(self.device)
        
        # 生成响应
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=query_ids,
                attention_mask=query_masks,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        
        # 分离query和response
        generated_ids = outputs.sequences  # [batch_size, total_len]
        
        # response_ids = generated_ids[:, query_ids.size(1):]
        response_ids = generated_ids
        
        # 解码响应
        responses = self.tokenizer.batch_decode(
            response_ids,
            skip_special_tokens=True
        )
        
        result = {
            "responses": responses,
            "response_ids": response_ids,
            "query_ids": query_ids,
        }
        
        # 计算log概率
        if return_logprobs:
            logprobs, masks = self._compute_logprobs(
                query_ids, response_ids, query_masks
            )
            result["logprobs"] = logprobs
            result["masks"] = masks
        
        return result
    
    def _compute_logprobs(
        self,
        query_ids: torch.Tensor,
        response_ids: torch.Tensor,
        query_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算生成token的log概率
        
        Returns:
            logprobs: [batch_size, response_len]
            masks: [batch_size, response_len]
        """
        # 拼接query和response
        full_ids = response_ids
        full_masks = (full_ids != self.tokenizer.pad_token_id).long()
        
        # 前向传播获取logits
        outputs = self.model(input_ids=full_ids, attention_mask=full_masks)
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        
        # 获取response部分的log概率
        # logits[:, :-1]对应token 1到n的预测
        # full_ids[:, 1:]是实际的token 1到n
        log_probs = torch.log_softmax(logits[:, :-1], dim=-1)  # [batch_size, seq_len-1, vocab_size]
        
        # 取出实际token的log概率
        target_ids = full_ids[:, 1:]  # [batch_size, seq_len-1]
        token_logprobs = log_probs.gather(
            dim=-1,
            index=target_ids.unsqueeze(-1)
        ).squeeze(-1)  # [batch_size, seq_len-1]
        
        # 创建mask（只保留response部分）
        query_len = query_ids.size(1)
        masks = full_masks[:, 1:].clone()  # [batch_size, seq_len-1]
        masks[:, :query_len-1] = 0  # 标记query部分为0
        
        return token_logprobs, masks
    
    def forward_pass(
        self,
        query_ids: torch.Tensor,
        response_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        训练前向传播
        
        Returns:
            logprobs: 当前策略的log概率
            ref_logprobs: 参考策略的log概率（用于KL约束）
        """
        # 拼接query和response
        full_ids = response_ids
        full_masks = (full_ids != self.tokenizer.pad_token_id).long()
        
        # 当前策略的log概率
        outputs = self.model(input_ids=full_ids, attention_mask=full_masks)
        logits = outputs.logits[:, :-1]  # [batch_size, seq_len-1, vocab_size]
        log_probs = torch.log_softmax(logits, dim=-1)
        
        target_ids = full_ids[:, 1:]
        token_logprobs = log_probs.gather(
            dim=-1,
            index=target_ids.unsqueeze(-1)
        ).squeeze(-1)
        
        # 参考策略的log概率
        ref_logprobs = None
        if self.ref_model is not None:
            with torch.no_grad():
                ref_outputs = self.ref_model(input_ids=full_ids, attention_mask=full_masks)
                ref_logits = ref_outputs.logits[:, :-1]
                ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
                ref_token_logprobs = ref_log_probs.gather(
                    dim=-1,
                    index=target_ids.unsqueeze(-1)
                ).squeeze(-1)
            ref_logprobs = ref_token_logprobs
        
        return token_logprobs, ref_logprobs
    
    def save_pretrained(self, save_path: str):
        """保存模型"""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
    
    def load_pretrained(self, load_path: str):
        """加载模型"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.model = AutoModelForCausalLM.from_pretrained(
            load_path,
            trust_remote_code=True,
        )
        self.model.to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            load_path,
            trust_remote_code=True
        )
