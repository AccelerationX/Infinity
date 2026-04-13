"""
Stage 1: 监督微调（SFT）Trainer

基于专家示范数据进行监督学习
为后续的RL训练提供良好的初始化
"""
import os
import torch
from torch.utils.data import DataLoader
from transformers import (
    get_cosine_schedule_with_warmup,
    default_data_collator,
)
from tqdm import tqdm
import json
from typing import Dict, List, Optional

from ..data.datasets import SFTDataset
from ..data.schemas import DemonstrationData
from ..models.lora_utils import load_model_with_lora


class SFTTrainer:
    """
    监督微调Trainer
    
    目标：学习专家示范，建立基础能力
    """
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        
        # 训练状态
        self.global_step = 0
        self.best_eval_loss = float('inf')
        
        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)
    
    def setup(self, model_name: str):
        """初始化模型和优化器"""
        print(f"Loading model: {model_name}")
        
        # 加载模型
        self.model, self.tokenizer = load_model_with_lora(
            model_name,
            use_lora=self.config.model.use_lora,
            lora_r=self.config.model.lora_r,
            lora_alpha=self.config.model.lora_alpha,
            lora_dropout=self.config.model.lora_dropout,
            torch_dtype=self.config.model.torch_dtype_enum,
        )
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.sft.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=self.config.sft.weight_decay,
        )
        
        print("Model loaded successfully")
    
    def train(
        self,
        train_data: List[DemonstrationData],
        eval_data: Optional[List[DemonstrationData]] = None,
    ):
        """
        执行训练
        
        Args:
            train_data: 训练数据
            eval_data: 评估数据（可选）
        """
        print(f"\n{'='*60}")
        print("Stage 1: Supervised Fine-Tuning (SFT)")
        print(f"{'='*60}")
        print(f"Training samples: {len(train_data)}")
        print(f"Eval samples: {len(eval_data) if eval_data else 0}")
        print(f"Epochs: {self.config.sft.num_train_epochs}")
        print(f"Batch size: {self.config.sft.per_device_train_batch_size}")
        print(f"Learning rate: {self.config.sft.learning_rate}")
        print(f"{'='*60}\n")
        
        # 创建数据集
        train_dataset = SFTDataset(
            train_data,
            self.tokenizer,
            max_length=self.config.sft.max_seq_length,
        )
        
        # 创建DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.sft.per_device_train_batch_size,
            shuffle=True,
            collate_fn=default_data_collator,
            num_workers=self.config.sft.dataloader_num_workers,
        )
        
        # 学习率调度器
        num_training_steps = len(train_loader) * self.config.sft.num_train_epochs
        num_warmup_steps = int(num_training_steps * self.config.sft.warmup_ratio)
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        
        # 训练循环
        self.model.train()
        global_step = 0
        
        for epoch in range(self.config.sft.num_train_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.sft.num_train_epochs}")
            
            epoch_loss = 0
            progress_bar = tqdm(train_loader, desc="Training")
            
            for step, batch in enumerate(progress_bar):
                # 移动到设备
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                
                # 前向传播
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # 反向传播
                loss.backward()
                
                # 梯度累积
                if (step + 1) % self.config.sft.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=1.0
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    global_step += 1
                
                # 更新进度条
                epoch_loss += loss.item()
                progress_bar.set_postfix({
                    "loss": loss.item(),
                    "lr": self.scheduler.get_last_lr()[0],
                })
                
                # 日志
                if global_step % self.config.sft.logging_steps == 0:
                    avg_loss = epoch_loss / (step + 1)
                    print(f"Step {global_step}: loss={avg_loss:.4f}")
            
            # 每个epoch结束，评估
            if eval_data:
                eval_loss = self.evaluate(eval_data)
                print(f"Epoch {epoch + 1} - Eval loss: {eval_loss:.4f}")
                
                # 保存最佳模型
                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    self.save_checkpoint(f"{self.config.sft.output_dir}/best")
                    print(f"New best model saved!")
            
            # 保存检查点
            if (epoch + 1) % 1 == 0:  # 每个epoch都保存
                self.save_checkpoint(f"{self.config.sft.output_dir}/checkpoint-{epoch+1}")
        
        # 保存最终模型
        self.save_checkpoint(self.config.sft.output_dir)
        print(f"\nSFT completed! Model saved to {self.config.sft.output_dir}")
    
    def evaluate(self, eval_data: List[DemonstrationData]) -> float:
        """评估"""
        eval_dataset = SFTDataset(
            eval_data,
            self.tokenizer,
            max_length=self.config.sft.max_seq_length,
        )
        
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.config.sft.per_device_eval_batch_size,
            collate_fn=default_data_collator,
        )
        
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()
        
        avg_loss = total_loss / len(eval_loader)
        self.model.train()
        
        return avg_loss
    
    def save_checkpoint(self, save_path: str):
        """保存检查点"""
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
    
    def generate_sample(self, prompt: str, max_new_tokens: int = 256) -> str:
        """生成示例"""
        self.model.eval()
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        self.model.train()
        return response
