"""
经验回放缓冲区 (Replay Buffer)
存储和采样训练经验，支持优先回放
"""
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import random


@dataclass
class Experience:
    """单条经验"""
    query: str
    response: str
    query_ids: torch.Tensor
    response_ids: torch.Tensor
    logprobs: torch.Tensor  # 旧策略的log概率
    rewards: float
    masks: torch.Tensor
    priority: float = 1.0  # 优先回放优先级
    success: bool = False  # 是否成功
    task_result: Optional[Dict] = None
    timestamp: float = 0.0


class ReplayBuffer:
    """
    经验回放缓冲区
    
    功能：
    1. 存储经验（成功/失败案例）
    2. 优先回放（基于TD误差或奖励）
    3. 成功经验蒸馏（保留高质量案例）
    4. 数据平衡（防止成功/失败样本不平衡）
    """
    
    def __init__(
        self,
        capacity: int = 10000,
        priority_alpha: float = 0.6,
        priority_beta: float = 0.4,
        success_ratio: float = 0.5,  # 成功样本的目标比例
    ):
        self.capacity = capacity
        self.priority_alpha = priority_alpha
        self.priority_beta = priority_beta
        self.success_ratio = success_ratio
        
        # 存储缓冲区
        self.buffer: deque = deque(maxlen=capacity)
        
        # 成功经验单独存储（用于蒸馏）
        self.success_buffer: deque = deque(maxlen=capacity // 2)
        
        # 优先级（用于优先回放）
        self.priorities = np.ones(capacity, dtype=np.float32)
        
        # 统计
        self.success_count = 0
        self.failure_count = 0
        
        self.position = 0
    
    def add(self, experience: Experience):
        """添加经验"""
        # 计算优先级（基于奖励的绝对值）
        priority = (abs(experience.rewards) + 1e-6) ** self.priority_alpha
        experience.priority = priority
        
        # 添加到主缓冲区
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            # 替换旧经验
            old_exp = self.buffer[self.position]
            if old_exp.success:
                self.success_count -= 1
            else:
                self.failure_count -= 1
            
            self.buffer[self.position] = experience
        
        # 更新统计
        if experience.success:
            self.success_count += 1
            # 同时添加到成功缓冲区
            self.success_buffer.append(experience)
        else:
            self.failure_count += 1
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(
        self,
        batch_size: int,
        device: str = "cuda",
        balanced: bool = True,
    ) -> Tuple[Dict[str, torch.Tensor], List[int]]:
        """
        采样批次
        
        Args:
            batch_size: 批次大小
            device: 设备
            balanced: 是否平衡成功/失败样本
            
        Returns:
            batch: 批次数据
            indices: 采样的索引
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        buffer_list = list(self.buffer)
        
        if balanced and self.success_count > 0 and self.failure_count > 0:
            # 平衡采样
            success_samples = min(batch_size // 2, self.success_count)
            failure_samples = batch_size - success_samples
            
            # 分离成功和失败样本
            success_exps = [e for e in buffer_list if e.success]
            failure_exps = [e for e in buffer_list if not e.success]
            
            # 按优先级采样
            success_batch = self._sample_by_priority(success_exps, success_samples)
            failure_batch = self._sample_by_priority(failure_exps, failure_samples)
            
            sampled_experiences = success_batch + failure_batch
            random.shuffle(sampled_experiences)
        else:
            # 普通优先回放采样
            sampled_experiences = self._sample_by_priority(buffer_list, batch_size)
        
        # 转换为tensor批次
        batch = self._collate_experiences(sampled_experiences, device)
        
        return batch, list(range(len(sampled_experiences)))
    
    def _sample_by_priority(
        self,
        experiences: List[Experience],
        sample_size: int,
    ) -> List[Experience]:
        """基于优先级采样"""
        if len(experiences) <= sample_size:
            return experiences
        
        # 获取优先级
        priorities = np.array([e.priority for e in experiences])
        probs = priorities / priorities.sum()
        
        # 采样
        indices = np.random.choice(
            len(experiences),
            size=sample_size,
            replace=False,
            p=probs,
        )
        
        return [experiences[i] for i in indices]
    
    def _collate_experiences(
        self,
        experiences: List[Experience],
        device: str,
    ) -> Dict[str, torch.Tensor]:
        """将经验列表转换为批次tensor"""
        # 找到最大长度
        max_query_len = max(e.query_ids.size(0) for e in experiences)
        max_response_len = max(e.response_ids.size(0) for e in experiences)
        
        batch_size = len(experiences)
        
        # 初始化tensor
        query_ids = torch.full((batch_size, max_query_len), 0, dtype=torch.long)
        response_ids = torch.full((batch_size, max_response_len), 0, dtype=torch.long)
        logprobs = torch.zeros((batch_size, max_response_len - 1))  # -1因为shift
        masks = torch.zeros((batch_size, max_response_len - 1))
        rewards = torch.zeros(batch_size)
        
        # 填充数据
        for i, exp in enumerate(experiences):
            q_len = exp.query_ids.size(0)
            r_len = exp.response_ids.size(0)
            
            query_ids[i, :q_len] = exp.query_ids
            response_ids[i, :r_len] = exp.response_ids
            
            lp_len = min(exp.logprobs.size(0), max_response_len - 1)
            logprobs[i, :lp_len] = exp.logprobs[:lp_len]
            
            m_len = min(exp.masks.size(0), max_response_len - 1)
            masks[i, :m_len] = exp.masks[:m_len]
            
            rewards[i] = exp.rewards
        
        return {
            "query_ids": query_ids.to(device),
            "response_ids": response_ids.to(device),
            "logprobs": logprobs.to(device),
            "masks": masks.to(device),
            "rewards": rewards.to(device),
        }
    
    def sample_success_cases(
        self,
        n: int = 10,
        similarity_threshold: float = 0.8,
    ) -> List[Experience]:
        """
        采样成功案例（用于蒸馏）
        
        Args:
            n: 采样数量
            similarity_threshold: 相似度阈值（去重）
            
        Returns:
            成功案例列表
        """
        if len(self.success_buffer) == 0:
            return []
        
        # 按奖励排序，取top n
        sorted_success = sorted(
            self.success_buffer,
            key=lambda e: e.rewards,
            reverse=True,
        )
        
        # 简单去重（基于query相似度）
        selected = []
        for exp in sorted_success:
            if len(selected) >= n:
                break
            
            # 检查是否与已选样本相似
            is_duplicate = False
            for sel in selected:
                # 简单字符串相似度
                similarity = self._string_similarity(exp.query, sel.query)
                if similarity > similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                selected.append(exp)
        
        return selected
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """计算字符串相似度（简化版Jaccard）"""
        set1 = set(s1.split())
        set2 = set(s2.split())
        
        if len(set1) == 0 and len(set2) == 0:
            return 1.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def update_priorities(self, indices: List[int], td_errors: List[float]):
        """更新优先级（基于TD误差）"""
        for idx, td_error in zip(indices, td_errors):
            if idx < len(self.buffer):
                priority = (abs(td_error) + 1e-6) ** self.priority_alpha
                list(self.buffer)[idx].priority = priority
    
    def get_statistics(self) -> Dict[str, float]:
        """获取缓冲区统计信息"""
        if len(self.buffer) == 0:
            return {
                "size": 0,
                "success_rate": 0.0,
                "mean_reward": 0.0,
                "success_count": 0,
                "failure_count": 0,
            }
        
        rewards = [e.rewards for e in self.buffer]
        
        return {
            "size": len(self.buffer),
            "success_rate": self.success_count / len(self.buffer),
            "mean_reward": np.mean(rewards),
            "max_reward": np.max(rewards),
            "min_reward": np.min(rewards),
            "success_count": self.success_count,
            "failure_count": self.failure_count,
        }
    
    def save(self, save_path: str):
        """保存缓冲区"""
        import os
        import pickle
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        data = {
            "buffer": list(self.buffer),
            "success_buffer": list(self.success_buffer),
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "position": self.position,
        }
        
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
    
    def load(self, load_path: str):
        """加载缓冲区"""
        import pickle
        
        with open(load_path, "rb") as f:
            data = pickle.load(f)
        
        self.buffer = deque(data["buffer"], maxlen=self.capacity)
        self.success_buffer = deque(data["success_buffer"], maxlen=self.capacity // 2)
        self.success_count = data["success_count"]
        self.failure_count = data["failure_count"]
        self.position = data["position"]
    
    def __len__(self):
        return len(self.buffer)
