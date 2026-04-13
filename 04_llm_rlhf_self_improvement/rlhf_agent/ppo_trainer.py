"""
PPO Trainer
使用近端策略优化算法训练策略模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class PPOConfig:
    """PPO训练配置"""
    ppo_epochs: int = 4
    clip_epsilon: float = 0.2
    value_clip: float = 0.4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    kl_penalty_coef: float = 0.2
    learning_rate: float = 1e-5
    batch_size: int = 4
    mini_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    mixed_precision: bool = True


class PPOTrainer:
    """
    PPO训练器
    
    核心算法:
    1. 收集经验（查询-响应-奖励）
    2. 计算优势函数 (GAE)
    3. 策略更新（带裁剪的目标函数）
    4. 价值函数更新
    """
    
    def __init__(
        self,
        policy_model,
        reward_model,
        config: PPOConfig,
    ):
        self.policy = policy_model
        self.reward_model = reward_model
        self.config = config
        
        # 初始化参考模型（用于KL约束）
        if self.policy.ref_model is None:
            self.policy.init_reference_model()
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.policy.model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=1000,
            eta_min=1e-6,
        )
        
        # 混合精度
        self.scaler = None
        if config.mixed_precision and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
    
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        masks: torch.Tensor,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算优势函数（使用GAE）
        
        Args:
            rewards: [batch_size, seq_len]
            values: [batch_size, seq_len] 价值估计
            masks: [batch_size, seq_len]
            gamma: 折扣因子
            gae_lambda: GAE lambda参数
            
        Returns:
            advantages: 优势估计
            returns: 回报估计（用于价值函数更新）
        """
        batch_size, seq_len = rewards.shape
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        # 逆向计算GAE
        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                next_value = 0
            else:
                next_value = values[:, t + 1]
            
            delta = rewards[:, t] + gamma * next_value * masks[:, t] - values[:, t]
            advantages[:, t] = last_gae = delta + gamma * gae_lambda * masks[:, t] * last_gae
        
        # 回报 = 优势 + 价值
        returns = advantages + values
        
        return advantages, returns
    
    def ppo_loss(
        self,
        old_logprobs: torch.Tensor,
        new_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        masks: torch.Tensor,
        ref_logprobs: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算PPO损失
        
        Args:
            old_logprobs: 旧策略的log概率 [batch_size, seq_len]
            new_logprobs: 新策略的log概率 [batch_size, seq_len]
            advantages: 优势估计 [batch_size, seq_len]
            masks: 掩码 [batch_size, seq_len]
            ref_logprobs: 参考策略的log概率（用于KL约束）
            
        Returns:
            losses: 包含各项损失的字典
        """
        # 重要性采样比率
        ratio = torch.exp(new_logprobs - old_logprobs)
        
        # 裁剪目标
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2) * masks
        policy_loss = policy_loss.sum() / masks.sum()
        
        # KL散度惩罚（防止策略偏离参考模型太远）
        kl_loss = 0
        if ref_logprobs is not None:
            kl = new_logprobs - ref_logprobs  # log(π/π_ref)
            kl_loss = self.config.kl_penalty_coef * (kl * masks).sum() / masks.sum()
        
        # 熵奖励（鼓励探索）
        # 这里简化处理，实际应该计算策略熵
        entropy_loss = 0
        
        # 总损失
        total_loss = policy_loss + kl_loss - self.config.entropy_coef * entropy_loss
        
        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "kl_loss": kl_loss,
            "entropy_loss": entropy_loss,
            "approx_kl": ((new_logprobs - old_logprobs) * masks).sum() / masks.sum(),
            "clip_frac": ((ratio - 1).abs() > self.config.clip_epsilon).float().mean(),
        }
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        单步训练
        
        Args:
            batch: 包含以下字段
                - query_ids: [batch_size, query_len]
                - response_ids: [batch_size, response_len]
                - old_logprobs: [batch_size, response_len]
                - rewards: [batch_size]
                - masks: [batch_size, response_len]
                
        Returns:
            metrics: 训练指标
        """
        query_ids = batch["query_ids"].to(self.policy.device)
        response_ids = batch["response_ids"].to(self.policy.device)
        old_logprobs = batch["logprobs"].to(self.policy.device)
        rewards = batch["rewards"].to(self.policy.device)
        masks = batch["masks"].to(self.policy.device)
        
        # 扩展rewards到每个token（简化处理）
        seq_len = response_ids.size(1)
        token_rewards = rewards.unsqueeze(1).expand(-1, seq_len - 1)  # -1因为shift
        
        # PPO更新循环
        all_metrics = []
        
        for ppo_epoch in range(self.config.ppo_epochs):
            # 前向传播获取新策略的log概率
            new_logprobs, ref_logprobs = self.policy.forward_pass(query_ids, response_ids)
            
            # 简化的价值估计（使用奖励的均值）
            values = torch.zeros_like(new_logprobs)
            
            # 计算优势
            advantages, returns = self.compute_advantages(
                token_rewards[:, :new_logprobs.size(1)],
                values,
                masks[:, :new_logprobs.size(1)],
            )
            
            # 归一化优势
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # 计算PPO损失
            losses = self.ppo_loss(
                old_logprobs[:, :new_logprobs.size(1)],
                new_logprobs,
                advantages,
                masks[:, :new_logprobs.size(1)],
                ref_logprobs,
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                self.scaler.scale(losses["total_loss"]).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.policy.model.parameters(),
                    self.config.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses["total_loss"].backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.model.parameters(),
                    self.config.max_grad_norm
                )
                self.optimizer.step()
            
            # 记录指标
            metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
            metrics["mean_advantage"] = advantages.mean().item()
            metrics["mean_reward"] = rewards.mean().item()
            all_metrics.append(metrics)
        
        self.scheduler.step()
        
        # 平均指标
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        return avg_metrics
    
    def save_checkpoint(self, save_path: str, iteration: int):
        """保存检查点"""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        checkpoint = {
            "iteration": iteration,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
        }
        
        if self.scaler is not None:
            checkpoint["scaler_state"] = self.scaler.state_dict()
        
        torch.save(checkpoint, os.path.join(save_path, f"checkpoint_{iteration}.pt"))
        
        # 保存策略模型
        self.policy.save_pretrained(os.path.join(save_path, f"policy_{iteration}"))
    
    def load_checkpoint(self, load_path: str):
        """加载检查点"""
        checkpoint = torch.load(load_path)
        
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        
        if self.scaler is not None and "scaler_state" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state"])
        
        return checkpoint["iteration"]
