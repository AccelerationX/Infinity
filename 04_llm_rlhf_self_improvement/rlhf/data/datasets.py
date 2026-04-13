"""
PyTorch Dataset实现

为三阶段分别实现Dataset
"""
import torch
from torch.utils.data import Dataset
from typing import List, Optional, Dict, Any
from transformers import PreTrainedTokenizer

from .schemas import DemonstrationData, PreferenceData, TrajectoryData


class SFTDataset(Dataset):
    """
    Stage 1: 监督微调数据集
    
    将DemonstrationData转换为模型输入
    """
    
    def __init__(
        self,
        data: List[DemonstrationData],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # 构建完整文本: prompt + completion
        full_text = item.prompt + item.completion
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        # 构建labels（只对completion部分计算loss）
        prompt_encoding = self.tokenizer(
            item.prompt,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_len = prompt_encoding["input_ids"].shape[1]
        
        labels = encoding["input_ids"].clone()
        # prompt部分mask掉（设为-100，不计算loss）
        labels[:, :prompt_len] = -100
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }


class PreferenceDataset(Dataset):
    """
    Stage 2: 偏好数据集
    
    将PreferenceData转换为pairwise训练格式
    """
    
    def __init__(
        self,
        data: List[PreferenceData],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        
        # 构建chosen和rejected的完整文本
        chosen_text = item.prompt + item.chosen
        rejected_text = item.prompt + item.rejected
        
        # Tokenize chosen
        chosen_encoding = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        # Tokenize rejected
        rejected_encoding = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "chosen_input_ids": chosen_encoding["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_encoding["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_encoding["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_encoding["attention_mask"].squeeze(0),
            "margin": torch.tensor(item.margin, dtype=torch.float32),
        }


class RolloutDataset(Dataset):
    """
    Stage 3: PPO Rollout数据集
    
    存储采集的trajectory，支持采样和GAE计算
    """
    
    def __init__(
        self,
        trajectories: List[TrajectoryData],
        gamma: float = 0.99,
        lam: float = 0.95,
    ):
        self.trajectories = trajectories
        self.gamma = gamma
        self.lam = lam
        
        # 预处理：计算所有trajectory的advantages和returns
        self._preprocess()
    
    def _preprocess(self):
        """预处理所有trajectory，计算advantages和returns"""
        self.samples = []
        
        for traj in self.trajectories:
            advantages = traj.compute_advantages(self.gamma, self.lam)
            returns = traj.compute_returns(self.gamma)
            
            for i, step in enumerate(traj.steps):
                self.samples.append({
                    "observation": step.observation,
                    "action": step.action,
                    "reward": step.reward,
                    "value": step.value,
                    "logprob": step.logprob,
                    "advantage": advantages[i],
                    "return": returns[i],
                    "done": step.done,
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]
    
    def get_batch(self, indices: List[int]) -> Dict[str, List]:
        """获取一个batch"""
        batch = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "values": [],
            "logprobs": [],
            "advantages": [],
            "returns": [],
        }
        
        for idx in indices:
            sample = self.samples[idx]
            for key in batch.keys():
                batch[key].append(sample[key])
        
        return batch


class QueryDataset(Dataset):
    """
    用于PPO采集的查询数据集
    
    只包含prompt，用于生成rollout
    """
    
    def __init__(
        self,
        queries: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 256,
    ):
        self.queries = queries
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        query = self.queries[idx]
        
        encoding = self.tokenizer(
            query,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "query": query,
        }
