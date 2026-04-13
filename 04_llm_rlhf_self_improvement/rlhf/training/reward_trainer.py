"""
Stage 2: 奖励模型训练（Reward Model Trainer）

基于Bradley-Terry模型进行Pairwise训练
学习人类偏好
"""
import os
import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
from typing import Dict, List, Optional

from ..data.datasets import PreferenceDataset
from ..data.schemas import PreferenceData
from ..models.reward_model import BradleyTerryRewardModel


class RewardTrainer:
    """
    奖励模型Trainer
    
    核心：Pairwise Ranking Loss
    L = -log σ(r(x, y_w) - r(x, y_l))
    """
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        # 训练状态
        self.global_step = 0
        self.best_accuracy = 0.0
        
        # 创建输出目录
        os.makedirs(config.reward.output_dir, exist_ok=True)
    
    def setup(self, model_name: str):
        """初始化模型"""
        print(f"Loading reward model: {model_name}")
        
        # 如果有SFT模型，从SFT初始化
        if hasattr(self.config, 'sft_model_path') and self.config.sft_model_path:
            print(f"Initializing from SFT model: {self.config.sft_model_path}")
            # 可以在这里加载SFT权重作为初始化
        
        self.model = BradleyTerryRewardModel(
            base_model_name=model_name,
            use_lora=self.config.model.use_lora,
            lora_r=self.config.model.lora_r,
            lora_alpha=self.config.model.lora_alpha,
            dropout=self.config.reward.reward_dropout,
        )
        
        # 优化器（只优化奖励头和LoRA参数）
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.reward.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=self.config.reward.weight_decay,
        )
        
        print("Reward model loaded successfully")
    
    def train(
        self,
        train_data: List[PreferenceData],
        eval_data: Optional[List[PreferenceData]] = None,
    ):
        """
        执行训练
        
        Args:
            train_data: 偏好对比数据
            eval_data: 评估数据
        """
        print(f"\n{'='*60}")
        print("Stage 2: Reward Model Training (Bradley-Terry)")
        print(f"{'='*60}")
        print(f"Training samples: {len(train_data)}")
        print(f"Eval samples: {len(eval_data) if eval_data else 0}")
        print(f"Epochs: {self.config.reward.num_train_epochs}")
        print(f"Learning rate: {self.config.reward.learning_rate}")
        print(f"{'='*60}\n")
        
        # 创建数据集
        train_dataset = PreferenceDataset(
            train_data,
            self.model.tokenizer,
            max_length=self.config.reward.max_length,
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.reward.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
        )
        
        # 学习率调度器
        num_training_steps = len(train_loader) * self.config.reward.num_train_epochs
        num_warmup_steps = int(num_training_steps * self.config.reward.warmup_ratio)
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        
        # 训练循环
        self.model.train()
        global_step = 0
        
        for epoch in range(self.config.reward.num_train_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.reward.num_train_epochs}")
            
            epoch_metrics = {
                "loss": 0,
                "accuracy": 0,
                "chosen_reward": 0,
                "rejected_reward": 0,
                "reward_margin": 0,
            }
            
            progress_bar = tqdm(train_loader, desc="Training")
            
            for step, batch in enumerate(progress_bar):
                # 移动到设备
                batch = {k: v.to(self.model.base_model.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # 计算损失
                loss, metrics = self.model.compute_pairwise_loss(
                    chosen_input_ids=batch["chosen_input_ids"],
                    chosen_attention_mask=batch["chosen_attention_mask"],
                    rejected_input_ids=batch["rejected_input_ids"],
                    rejected_attention_mask=batch["rejected_attention_mask"],
                    margin=batch.get("margin"),
                )
                
                # 反向传播
                loss.backward()
                
                # 梯度累积
                if (step + 1) % self.config.reward.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=1.0
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    global_step += 1
                
                # 累计指标
                for key in epoch_metrics:
                    if key in metrics:
                        epoch_metrics[key] += metrics[key]
                
                # 更新进度条
                progress_bar.set_postfix({
                    "loss": metrics["loss"],
                    "acc": metrics["accuracy"],
                    "margin": metrics["reward_margin"],
                })
                
                # 日志
                if global_step % self.config.reward.logging_steps == 0:
                    avg_metrics = {k: v / (step + 1) for k, v in epoch_metrics.items()}
                    print(f"\nStep {global_step}:")
                    print(f"  Loss: {avg_metrics['loss']:.4f}")
                    print(f"  Accuracy: {avg_metrics['accuracy']:.4f}")
                    print(f"  Reward Margin: {avg_metrics['reward_margin']:.4f}")
            
            # 每个epoch结束，评估
            if eval_data:
                eval_metrics = self.evaluate(eval_data)
                print(f"\nEpoch {epoch + 1} - Eval Results:")
                print(f"  Accuracy: {eval_metrics['accuracy']:.4f}")
                print(f"  Reward Margin: {eval_metrics['reward_margin']:.4f}")
                
                # 保存最佳模型
                if eval_metrics['accuracy'] > self.best_accuracy:
                    self.best_accuracy = eval_metrics['accuracy']
                    self.save_checkpoint(f"{self.config.reward.output_dir}/best")
                    print(f"New best model saved! (accuracy={self.best_accuracy:.4f})")
            
            # 保存检查点
            self.save_checkpoint(f"{self.config.reward.output_dir}/checkpoint-{epoch+1}")
        
        # 保存最终模型
        self.save_checkpoint(self.config.reward.output_dir)
        print(f"\nReward model training completed!")
        print(f"Model saved to {self.config.reward.output_dir}")
        print(f"Best accuracy: {self.best_accuracy:.4f}")
    
    def evaluate(self, eval_data: List[PreferenceData]) -> Dict[str, float]:
        """评估"""
        eval_dataset = PreferenceDataset(
            eval_data,
            self.model.tokenizer,
            max_length=self.config.reward.max_length,
        )
        
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.config.reward.per_device_eval_batch_size,
            collate_fn=self._collate_fn,
        )
        
        self.model.eval()
        
        total_metrics = {
            "loss": 0,
            "accuracy": 0,
            "chosen_reward": 0,
            "rejected_reward": 0,
            "reward_margin": 0,
        }
        
        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(self.model.base_model.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                _, metrics = self.model.compute_pairwise_loss(
                    batch["chosen_input_ids"],
                    batch["chosen_attention_mask"],
                    batch["rejected_input_ids"],
                    batch["rejected_attention_mask"],
                    batch.get("margin"),
                )
                
                for key in total_metrics:
                    if key in metrics:
                        total_metrics[key] += metrics[key]
        
        # 平均
        num_batches = len(eval_loader)
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        
        self.model.train()
        return avg_metrics
    
    def _collate_fn(self, batch):
        """数据collate函数"""
        return {
            "chosen_input_ids": torch.stack([item["chosen_input_ids"] for item in batch]),
            "chosen_attention_mask": torch.stack([item["chosen_attention_mask"] for item in batch]),
            "rejected_input_ids": torch.stack([item["rejected_input_ids"] for item in batch]),
            "rejected_attention_mask": torch.stack([item["rejected_attention_mask"] for item in batch]),
            "margin": torch.stack([item["margin"] for item in batch]) if "margin" in batch[0] else None,
        }
    
    def save_checkpoint(self, save_path: str):
        """保存检查点"""
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
    
    def predict_reward(self, texts: List[str]) -> List[float]:
        """预测奖励值"""
        self.model.eval()
        rewards = self.model.get_reward(texts)
        return rewards.tolist()
