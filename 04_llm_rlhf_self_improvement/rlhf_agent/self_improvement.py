"""
自我改进循环 (Self-Improvement Loop)
整合所有组件，实现Agent的持续自我改进
"""
import os
import time
import json
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, asdict
import numpy as np

from .config import RLHFConfig
from .policy_model import PolicyModel, PolicyConfig
from .reward_model import RewardModel, RewardConfig, RuleBasedRewardModel, create_default_rule_based_reward
from .ppo_trainer import PPOTrainer, PPOConfig
from .replay_buffer import ReplayBuffer, Experience


@dataclass
class ImprovementConfig:
    """自我改进配置"""
    iterations: int = 10
    episodes_per_iteration: int = 20
    success_threshold: float = 0.7
    kl_penalty_coef: float = 0.2
    eval_interval: int = 2
    eval_episodes: int = 10
    checkpoint_interval: int = 5
    output_dir: str = "./rlhf_outputs"
    use_rule_based_reward: bool = True  # 冷启动时使用规则奖励


class SelfImprovementLoop:
    """
    自我改进循环
    
    核心流程：
    1. 收集经验（与环境交互）
    2. 计算奖励
    3. 存储到回放缓冲区
    4. 采样批次进行PPO训练
    5. 评估改进效果
    6. 重复直到收敛
    """
    
    def __init__(
        self,
        config: RLHFConfig,
        improvement_config: ImprovementConfig,
        task_executor: Optional[Callable] = None,
    ):
        self.config = config
        self.improvement_config = improvement_config
        self.task_executor = task_executor
        
        # 初始化组件
        self._init_components()
        
        # 创建输出目录
        os.makedirs(improvement_config.output_dir, exist_ok=True)
        
        # 训练历史
        self.training_history = []
        self.iteration = 0
    
    def _init_components(self):
        """初始化组件"""
        # 策略模型
        policy_config = PolicyConfig(
            model_name=self.config.base_model_name,
            use_lora=self.config.use_lora,
            lora_rank=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            temperature=0.7,
            device=self.config.device,
        )
        self.policy = PolicyModel(policy_config)
        
        # 奖励模型
        if self.improvement_config.use_rule_based_reward:
            # 冷启动：使用规则奖励
            self.reward_model = create_default_rule_based_reward()
            self.using_rule_reward = True
        else:
            # 使用学习的奖励模型
            reward_config = RewardConfig(
                model_name=self.config.base_model_name,
                use_lora=self.config.use_lora,
                device=self.config.device,
            )
            self.reward_model = RewardModel(reward_config)
            self.using_rule_reward = False
        
        # PPO训练器
        ppo_config = PPOConfig(
            ppo_epochs=self.config.ppo_epochs,
            clip_epsilon=self.config.ppo_clip_epsilon,
            value_clip=self.config.ppo_value_clip,
            entropy_coef=self.config.ppo_entropy_coef,
            value_coef=self.config.ppo_value_coef,
            kl_penalty_coef=self.config.kl_penalty_coef,
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            mini_batch_size=self.config.mini_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            mixed_precision=self.config.mixed_precision,
        )
        
        # 只有使用神经网络奖励模型时才需要PPO
        if not self.improvement_config.use_rule_based_reward:
            self.trainer = PPOTrainer(self.policy, self.reward_model, ppo_config)
        else:
            self.trainer = None
        
        # 回放缓冲区
        self.replay_buffer = ReplayBuffer(
            capacity=self.config.replay_buffer_size,
            priority_alpha=self.config.priority_alpha,
            priority_beta=self.config.priority_beta,
        )
    
    def collect_episodes(
        self,
        queries: List[str],
        iteration: int,
    ) -> List[Experience]:
        """
        收集训练经验
        
        Args:
            queries: 查询列表
            iteration: 当前迭代轮次
            
        Returns:
            经验列表
        """
        experiences = []
        
        # 生成响应
        print(f"  Generating responses for {len(queries)} queries...")
        results = self.policy.generate(
            queries,
            max_new_tokens=self.config.max_new_tokens,
            return_logprobs=True,
        )
        
        responses = results["responses"]
        response_ids = results["response_ids"]
        query_ids = results["query_ids"]
        logprobs = results["logprobs"]
        masks = results["masks"]
        
        # 执行任务（如果有执行器）
        task_results = []
        if self.task_executor:
            for query, response in zip(queries, responses):
                try:
                    result = self.task_executor(query, response)
                    task_results.append(result)
                except Exception as e:
                    task_results.append({"success": False, "error": str(e)})
        else:
            task_results = [None] * len(queries)
        
        # 计算奖励
        print("  Computing rewards...")
        if self.using_rule_reward:
            rewards = self.reward_model.get_rewards(queries, responses, task_results)
        else:
            rewards = self.reward_model.get_rewards(queries, responses)
        
        # 创建经验
        for i in range(len(queries)):
            exp = Experience(
                query=queries[i],
                response=responses[i],
                query_ids=query_ids[i],
                response_ids=response_ids[i],
                logprobs=logprobs[i],
                rewards=rewards[i].item(),
                masks=masks[i],
                success=rewards[i].item() > self.improvement_config.success_threshold,
                task_result=task_results[i],
                timestamp=time.time(),
            )
            experiences.append(exp)
        
        return experiences
    
    def train_iteration(self) -> Dict[str, float]:
        """单次训练迭代"""
        print(f"\n{'='*60}")
        print(f"Iteration {self.iteration + 1}/{self.improvement_config.iterations}")
        print(f"{'='*60}")
        
        # 1. 生成查询（这里使用示例查询，实际应该从任务分布采样）
        queries = self._generate_queries(self.improvement_config.episodes_per_iteration)
        
        # 2. 收集经验
        print("\n[1/4] Collecting experiences...")
        experiences = self.collect_episodes(queries, self.iteration)
        
        # 3. 添加到回放缓冲区
        print(f"\n[2/4] Adding to replay buffer...")
        for exp in experiences:
            self.replay_buffer.add(exp)
        
        buffer_stats = self.replay_buffer.get_statistics()
        print(f"  Buffer size: {buffer_stats['size']}")
        print(f"  Success rate: {buffer_stats['success_rate']:.2%}")
        print(f"  Mean reward: {buffer_stats['mean_reward']:.4f}")
        
        # 4. 训练（如果不是使用纯规则奖励）
        metrics = {}
        if self.trainer is not None and len(self.replay_buffer) >= self.config.batch_size:
            print(f"\n[3/4] Training PPO...")
            
            # 采样批次
            batch, _ = self.replay_buffer.sample(
                self.config.batch_size,
                device=self.config.device,
                balanced=True,
            )
            
            # PPO训练
            metrics = self.trainer.train_step(batch)
            
            print(f"  Total loss: {metrics['total_loss']:.4f}")
            print(f"  Policy loss: {metrics['policy_loss']:.4f}")
            print(f"  Mean reward: {metrics['mean_reward']:.4f}")
        
        # 5. 评估
        print(f"\n[4/4] Evaluating...")
        eval_metrics = self.evaluate()
        
        # 合并指标
        metrics.update(eval_metrics)
        metrics["buffer_size"] = buffer_stats["size"]
        metrics["success_rate"] = buffer_stats["success_rate"]
        
        # 记录历史
        self.training_history.append({
            "iteration": self.iteration,
            "metrics": metrics,
            "timestamp": time.time(),
        })
        
        # 保存检查点
        if (self.iteration + 1) % self.improvement_config.checkpoint_interval == 0:
            self._save_checkpoint()
        
        self.iteration += 1
        
        return metrics
    
    def evaluate(self) -> Dict[str, float]:
        """评估当前策略"""
        eval_queries = self._generate_queries(
            self.improvement_config.eval_episodes,
            seed=42  # 固定种子保证可复现
        )
        
        # 生成响应
        results = self.policy.generate(
            eval_queries,
            max_new_tokens=self.config.max_new_tokens,
            return_logprobs=False,
        )
        
        responses = results["responses"]
        
        # 执行任务
        task_results = []
        if self.task_executor:
            for query, response in zip(eval_queries, responses):
                try:
                    result = self.task_executor(query, response)
                    task_results.append(result)
                except Exception as e:
                    task_results.append({"success": False, "error": str(e)})
        else:
            task_results = [None] * len(eval_queries)
        
        # 计算奖励
        if self.using_rule_reward:
            rewards = self.reward_model.get_rewards(eval_queries, responses, task_results)
        else:
            rewards = self.reward_model.get_rewards(eval_queries, responses)
        
        # 计算指标
        reward_list = rewards.cpu().numpy() if hasattr(rewards, 'cpu') else rewards
        
        successes = sum(1 for r in reward_list if r > self.improvement_config.success_threshold)
        
        metrics = {
            "eval_mean_reward": float(np.mean(reward_list)),
            "eval_success_rate": successes / len(eval_queries),
            "eval_max_reward": float(np.max(reward_list)),
            "eval_min_reward": float(np.min(reward_list)),
        }
        
        print(f"  Eval mean reward: {metrics['eval_mean_reward']:.4f}")
        print(f"  Eval success rate: {metrics['eval_success_rate']:.2%}")
        
        return metrics
    
    def train(self) -> List[Dict]:
        """完整训练循环"""
        print("\n" + "="*60)
        print("Starting Self-Improvement Training")
        print("="*60)
        print(f"Total iterations: {self.improvement_config.iterations}")
        print(f"Episodes per iteration: {self.improvement_config.episodes_per_iteration}")
        print(f"Success threshold: {self.improvement_config.success_threshold}")
        print(f"Device: {self.config.device}")
        print("="*60 + "\n")
        
        for i in range(self.improvement_config.iterations):
            metrics = self.train_iteration()
            
            # 检查收敛
            if metrics.get("eval_success_rate", 0) >= 0.95:
                print(f"\n🎉 Converged at iteration {i+1}!")
                break
        
        # 保存最终结果
        self._save_final_model()
        self._save_training_history()
        
        return self.training_history
    
    def _generate_queries(self, n: int, seed: Optional[int] = None) -> List[str]:
        """生成训练查询（示例实现）"""
        if seed is not None:
            np.random.seed(seed)
        
        # 示例查询模板
        templates = [
            "Calculate the sum of {a} and {b}",
            "What is {a} multiplied by {b}?",
            "Find the result of {a} minus {b}",
            "Divide {a} by {b}",
            "What is {a} to the power of {b}?",
        ]
        
        queries = []
        for _ in range(n):
            template = np.random.choice(templates)
            a = np.random.randint(1, 100)
            b = np.random.randint(1, 100)
            query = template.format(a=a, b=b)
            queries.append(query)
        
        return queries
    
    def _save_checkpoint(self):
        """保存检查点"""
        if self.trainer is not None:
            checkpoint_dir = os.path.join(
                self.improvement_config.output_dir,
                "checkpoints"
            )
            self.trainer.save_checkpoint(checkpoint_dir, self.iteration)
            print(f"  Checkpoint saved to {checkpoint_dir}")
    
    def _save_final_model(self):
        """保存最终模型"""
        final_dir = os.path.join(self.improvement_config.output_dir, "final_model")
        self.policy.save_pretrained(final_dir)
        print(f"\nFinal model saved to {final_dir}")
    
    def _save_training_history(self):
        """保存训练历史"""
        history_path = os.path.join(
            self.improvement_config.output_dir,
            "training_history.json"
        )
        
        with open(history_path, "w") as f:
            json.dump(self.training_history, f, indent=2)
        
        print(f"Training history saved to {history_path}")
        
        # 生成训练曲线图
        self._plot_training_curves()
    
    def _plot_training_curves(self):
        """绘制训练曲线"""
        try:
            import matplotlib.pyplot as plt
            
            iterations = [h["iteration"] for h in self.training_history]
            rewards = [h["metrics"].get("eval_mean_reward", 0) for h in self.training_history]
            success_rates = [h["metrics"].get("eval_success_rate", 0) for h in self.training_history]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # 奖励曲线
            ax1.plot(iterations, rewards, "b-", label="Mean Reward")
            ax1.set_xlabel("Iteration")
            ax1.set_ylabel("Reward")
            ax1.set_title("Reward over Iterations")
            ax1.legend()
            ax1.grid(True)
            
            # 成功率曲线
            ax2.plot(iterations, success_rates, "g-", label="Success Rate")
            ax2.axhline(y=self.improvement_config.success_threshold, color="r", linestyle="--", label="Threshold")
            ax2.set_xlabel("Iteration")
            ax2.set_ylabel("Success Rate")
            ax2.set_title("Success Rate over Iterations")
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            plot_path = os.path.join(self.improvement_config.output_dir, "training_curves.png")
            plt.savefig(plot_path, dpi=150)
            print(f"Training curves saved to {plot_path}")
        except ImportError:
            print("matplotlib not installed, skipping plot generation")
