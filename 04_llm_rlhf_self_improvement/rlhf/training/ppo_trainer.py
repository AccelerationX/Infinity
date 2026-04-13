"""
Stage 3: PPO Trainer

标准Actor-Critic PPO实现
- GAE优势估计
- 裁剪目标函数
- KL散度约束
"""
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm

from ..models.actor_critic import ActorCriticModel
from ..models.reward_model import BradleyTerryRewardModel
from ..data.schemas import TrajectoryData, StepData


class PPOTrainer:
    """
    PPO Trainer
    
    标准PPO算法流程：
    1. 采集Rollout数据
    2. 计算GAE优势
    3. 更新策略（PPO裁剪目标）
    4. 更新价值函数
    """
    
    def __init__(self, config):
        self.config = config
        self.actor_critic = None
        self.reward_model = None
        self.optimizer = None
        
        # 训练状态
        self.global_step = 0
        self.best_reward = -float('inf')
        
        # 统计
        self.kl_ctl = AdaptiveKLController(
            config.ppo.kl_coef,
            config.ppo.kl_target,
        )
        
        # 创建输出目录
        os.makedirs(config.ppo.output_dir, exist_ok=True)
    
    def setup(
        self,
        actor_model_name: str,
        reward_model_path: str,
    ):
        """初始化模型"""
        print(f"Loading Actor-Critic model: {actor_model_name}")
        
        self.actor_critic = ActorCriticModel(
            base_model_name=actor_model_name,
            use_lora=self.config.model.use_lora,
            lora_r=self.config.model.lora_r,
            lora_alpha=self.config.model.lora_alpha,
            device=self.config.model.device,
        )
        
        # 初始化参考模型（用于KL约束）
        self.actor_critic.init_reference_model()
        
        # 加载奖励模型
        print(f"Loading reward model: {reward_model_path}")
        self.reward_model = BradleyTerryRewardModel.load_pretrained(
            reward_model_path,
            use_lora=False,  # 奖励模型冻结
        )
        self.reward_model.eval()
        for param in self.reward_model.parameters():
            param.requires_grad = False
        
        # 优化器（优化Actor和Critic）
        self.optimizer = torch.optim.AdamW(
            list(self.actor_critic.actor.parameters()) + 
            list(self.actor_critic.critic.parameters()),
            lr=self.config.ppo.learning_rate,
            betas=(0.9, 0.999),
        )
        
        print("Models loaded successfully")
    
    def collect_rollouts(
        self,
        queries: List[str],
    ) -> List[TrajectoryData]:
        """
        采集rollout数据
        
        Args:
            queries: 查询列表
            
        Returns:
            trajectories: 轨迹列表
        """
        self.actor_critic.eval()
        
        trajectories = []
        batch_size = self.config.ppo.rollout_batch_size
        
        print(f"Collecting rollouts for {len(queries)} queries...")
        
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i + batch_size]
            
            # 生成响应
            with torch.no_grad():
                gen_outputs = self.actor_critic.generate(
                    prompts=batch_queries,
                    max_new_tokens=self.config.ppo.max_new_tokens,
                    temperature=self.config.ppo.temperature,
                    top_p=self.config.ppo.top_p,
                )
            
            responses = gen_outputs["responses"]
            sequences = gen_outputs["sequences"]
            prompt_len = gen_outputs["prompt_length"]
            
            # 计算奖励
            full_texts = [q + r for q, r in zip(batch_queries, responses)]
            rewards = self.reward_model.get_reward(full_texts)
            
            # 获取价值和logprob
            with torch.no_grad():
                # Tokenize
                inputs = self.actor_critic.tokenizer(
                    full_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.config.ppo.max_seq_length,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.actor_critic.device) for k, v in inputs.items()}
                
                # 前向传播
                outputs = self.actor_critic.get_action_and_value(
                    inputs["input_ids"],
                    inputs["attention_mask"],
                    return_ref_logprob=True,
                )
                
                values = outputs["values"].squeeze(-1).cpu().numpy()
                ref_logprobs = outputs.get("ref_logprobs", None)
                
                # 计算序列的logprob
                seq_logprobs = self.actor_critic.compute_logprobs(
                    inputs["input_ids"],
                    inputs["attention_mask"],
                ).cpu().numpy()
            
            # 构建trajectories
            for j, query in enumerate(batch_queries):
                traj = TrajectoryData(prompt=query)
                
                # 简化为单步（实际可以细粒度）
                step = StepData(
                    observation=query,
                    action=responses[j],
                    reward=rewards[j].item(),
                    value=values[j],
                    logprob=seq_logprobs[j].mean(),  # 平均logprob
                    done=True,
                    ref_logprob=ref_logprobs[j].mean().item() if ref_logprobs is not None else None,
                )
                
                traj.add_step(step)
                traj.total_reward = rewards[j].item()
                traj.success = rewards[j].item() > 0.5
                
                trajectories.append(traj)
        
        self.actor_critic.train()
        return trajectories
    
    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算GAE优势
        
        Args:
            rewards: [T]
            values: [T]
            
        Returns:
            advantages: [T]
            returns: [T]
        """
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value - values[t]
            advantages[t] = last_gae = delta + gamma * lam * last_gae
        
        returns = advantages + values
        return advantages, returns
    
    def ppo_update(
        self,
        trajectories: List[TrajectoryData],
    ) -> Dict[str, float]:
        """
        PPO更新
        
        Args:
            trajectories: 轨迹列表
            
        Returns:
            metrics: 训练指标
        """
        # 准备数据
        all_queries = []
        all_responses = []
        all_old_logprobs = []
        all_rewards = []
        all_values = []
        all_advantages = []
        all_returns = []
        all_ref_logprobs = []
        
        for traj in trajectories:
            for step in traj.steps:
                all_queries.append(traj.prompt)
                all_responses.append(step.action)
                all_old_logprobs.append(step.logprob)
                all_rewards.append(step.reward)
                all_values.append(step.value)
                all_ref_logprobs.append(step.ref_logprob or 0)
        
        # 计算GAE
        advantages, returns = self.compute_gae(
            np.array(all_rewards),
            np.array(all_values),
            self.config.ppo.gamma,
            self.config.ppo.lam,
        )
        
        all_advantages = advantages.tolist()
        all_returns = returns.tolist()
        
        # 归一化优势
        advantages = np.array(all_advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        all_advantages = advantages.tolist()
        
        # 转换为tensor
        old_logprobs_tensor = torch.tensor(all_old_logprobs, dtype=torch.float32)
        advantages_tensor = torch.tensor(all_advantages, dtype=torch.float32)
        returns_tensor = torch.tensor(all_returns, dtype=torch.float32)
        ref_logprobs_tensor = torch.tensor(all_ref_logprobs, dtype=torch.float32)
        
        # 准备batch
        batch_size = self.config.ppo.step_batch_size
        num_samples = len(all_queries)
        
        # PPO epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_kl_loss = 0
        total_entropy = 0
        
        for ppo_epoch in range(self.config.ppo.ppo_epochs):
            # 随机打乱
            indices = np.random.permutation(num_samples)
            
            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                # 获取batch数据
                batch_queries = [all_queries[i] for i in batch_indices]
                batch_responses = [all_responses[i] for i in batch_indices]
                batch_old_logprobs = old_logprobs_tensor[batch_indices].to(self.actor_critic.device)
                batch_advantages = advantages_tensor[batch_indices].to(self.actor_critic.device)
                batch_returns = returns_tensor[batch_indices].to(self.actor_critic.device)
                batch_ref_logprobs = ref_logprobs_tensor[batch_indices].to(self.actor_critic.device)
                
                # 前向传播
                full_texts = [q + r for q, r in zip(batch_queries, batch_responses)]
                inputs = self.actor_critic.tokenizer(
                    full_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.config.ppo.max_seq_length,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.actor_critic.device) for k, v in inputs.items()}
                
                outputs = self.actor_critic.forward(
                    inputs["input_ids"],
                    inputs["attention_mask"],
                )
                
                logits = outputs["logits"]
                values = outputs["values"].squeeze(-1)
                
                # 计算新的logprobs
                log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
                target_ids = inputs["input_ids"][:, 1:]
                new_logprobs = log_probs.gather(
                    dim=-1,
                    index=target_ids.unsqueeze(-1)
                ).squeeze(-1).mean(dim=1)
                
                # PPO损失
                ratio = torch.exp(new_logprobs - batch_old_logprobs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio,
                    1 - self.config.ppo.clip_eps,
                    1 + self.config.ppo.clip_eps
                ) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # KL损失
                kl = new_logprobs - batch_ref_logprobs
                kl_loss = self.kl_ctl.value * kl.mean()
                
                # 价值损失
                value_loss = F.mse_loss(values, batch_returns)
                
                # 总损失
                loss = (
                    policy_loss +
                    self.config.ppo.vf_coef * value_loss +
                    kl_loss
                )
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor_critic.actor.parameters()) +
                    list(self.actor_critic.critic.parameters()),
                    max_norm=1.0
                )
                self.optimizer.step()
                
                # 累计
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_kl_loss += kl_loss.item()
        
        # 更新KL控制器
        mean_kl = (new_logprobs - batch_ref_logprobs).mean().item()
        self.kl_ctl.update(mean_kl)
        
        num_updates = self.config.ppo.ppo_epochs * (num_samples // batch_size + 1)
        
        return {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "kl_loss": total_kl_loss / num_updates,
            "mean_kl": mean_kl,
            "mean_reward": np.mean(all_rewards),
            "mean_advantage": np.mean(all_advantages),
        }
    
    def train(self, queries: List[str]):
        """
        执行PPO训练
        
        Args:
            queries: 训练查询列表
        """
        print(f"\n{'='*60}")
        print("Stage 3: PPO Training")
        print(f"{'='*60}")
        print(f"Total rollouts: {self.config.ppo.num_rollouts}")
        print(f"Rollout batch size: {self.config.ppo.rollout_batch_size}")
        print(f"PPO epochs: {self.config.ppo.ppo_epochs}")
        print(f"{'='*60}\n")
        
        num_iterations = self.config.ppo.num_rollouts // self.config.ppo.rollout_batch_size
        
        for iteration in range(num_iterations):
            print(f"\nIteration {iteration + 1}/{num_iterations}")
            
            # 1. 采集rollout
            print("Collecting rollouts...")
            sample_queries = np.random.choice(
                queries,
                size=self.config.ppo.rollout_batch_size,
                replace=False
            ).tolist()
            
            trajectories = self.collect_rollouts(sample_queries)
            
            # 统计
            rewards = [t.total_reward for t in trajectories]
            mean_reward = np.mean(rewards)
            success_rate = sum(1 for t in trajectories if t.success) / len(trajectories)
            
            print(f"Rollout stats:")
            print(f"  Mean reward: {mean_reward:.4f}")
            print(f"  Success rate: {success_rate:.2%}")
            
            # 2. PPO更新
            print("Updating policy...")
            metrics = self.ppo_update(trajectories)
            
            print(f"Update stats:")
            print(f"  Policy loss: {metrics['policy_loss']:.4f}")
            print(f"  Value loss: {metrics['value_loss']:.4f}")
            print(f"  KL loss: {metrics['kl_loss']:.4f}")
            print(f"  Mean KL: {metrics['mean_kl']:.4f}")
            
            self.global_step += 1
            
            # 保存检查点
            if mean_reward > self.best_reward:
                self.best_reward = mean_reward
                self.save_checkpoint(f"{self.config.ppo.output_dir}/best")
                print(f"New best model saved! (reward={self.best_reward:.4f})")
            
            if (iteration + 1) % 10 == 0:
                self.save_checkpoint(f"{self.config.ppo.output_dir}/checkpoint-{iteration+1}")
        
        # 保存最终模型
        self.save_checkpoint(self.config.ppo.output_dir)
        print(f"\nPPO training completed!")
        print(f"Best reward: {self.best_reward:.4f}")
    
    def save_checkpoint(self, save_path: str):
        """保存检查点"""
        os.makedirs(save_path, exist_ok=True)
        self.actor_critic.save_pretrained(save_path)


class AdaptiveKLController:
    """自适应KL控制器"""
    
    def __init__(self, init_kl_coef: float, target_kl: Optional[float] = None):
        self.value = init_kl_coef
        self.target = target_kl
        self.alpha = 1.5  # 调整因子
        self.beta = 0.5   # 调整因子
    
    def update(self, current_kl: float):
        """根据当前KL更新系数"""
        if self.target is None:
            return
        
        if current_kl > self.target * 1.5:
            self.value *= self.alpha
        elif current_kl < self.target / 1.5:
            self.value *= self.beta
        
        # 限制范围
        self.value = max(0.01, min(1.0, self.value))
