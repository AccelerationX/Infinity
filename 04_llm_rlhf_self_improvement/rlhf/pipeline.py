"""
RLHF完整Pipeline

整合三阶段训练流程：
1. SFT: 监督微调
2. Reward Model: 奖励模型训练
3. PPO: 强化学习训练
"""
import os
import json
from typing import List, Optional, Dict
from dataclasses import asdict

from .config import RLHFConfig
from .training.sft_trainer import SFTTrainer
from .training.reward_trainer import RewardTrainer
from .training.ppo_trainer import PPOTrainer
from .data.schemas import DemonstrationData, PreferenceData


class RLHFPipeline:
    """
    RLHF完整训练Pipeline
    
    执行标准的三阶段训练：
    Stage 1: SFT (Supervised Fine-Tuning)
    Stage 2: Reward Model Training
    Stage 3: PPO (Proximal Policy Optimization)
    """
    
    def __init__(self, config: RLHFConfig):
        self.config = config
        
        # 创建输出根目录
        os.makedirs(config.output_root, exist_ok=True)
        
        # 保存配置
        self._save_config()
    
    def _save_config(self):
        """保存配置"""
        config_path = os.path.join(self.config.output_root, "config.json")
        with open(config_path, "w") as f:
            json.dump(asdict(self.config), f, indent=2, default=str)
    
    def run(
        self,
        demonstration_data: List[DemonstrationData],
        preference_data: List[PreferenceData],
        eval_demonstration_data: Optional[List[DemonstrationData]] = None,
        eval_preference_data: Optional[List[PreferenceData]] = None,
        ppo_queries: Optional[List[str]] = None,
    ) -> Dict[str, any]:
        """
        执行完整的RLHF训练
        
        Args:
            demonstration_data: SFT训练数据
            preference_data: 奖励模型训练数据
            eval_demonstration_data: SFT评估数据
            eval_preference_data: 奖励模型评估数据
            ppo_queries: PPO训练查询
            
        Returns:
            results: 各阶段训练结果
        """
        results = {}
        
        print("\n" + "="*70)
        print(" "*20 + "RLHF TRAINING PIPELINE")
        print("="*70)
        print(f"\nConfiguration:")
        print(f"  Base model: {self.config.model.model_name}")
        print(f"  Use LoRA: {self.config.model.use_lora}")
        print(f"  Device: {self.config.model.device}")
        print(f"\nData:")
        print(f"  Demonstration: {len(demonstration_data)}")
        print(f"  Preference: {len(preference_data)}")
        print("="*70 + "\n")
        
        # Stage 1: SFT
        if self.config.run_sft:
            print("\n" + "="*70)
            print("STAGE 1: SUPERVISED FINE-TUNING (SFT)")
            print("="*70)
            
            sft_trainer = SFTTrainer(self.config)
            sft_trainer.setup(self.config.model.model_name)
            sft_trainer.train(
                demonstration_data,
                eval_data=eval_demonstration_data,
            )
            
            results["sft"] = {
                "output_dir": self.config.sft.output_dir,
                "status": "completed",
            }
            
            print("\n" + "="*70)
            print("SFT COMPLETED")
            print(f"Model saved to: {self.config.sft.output_dir}")
            print("="*70)
        
        # Stage 2: Reward Model
        if self.config.run_reward:
            print("\n" + "="*70)
            print("STAGE 2: REWARD MODEL TRAINING")
            print("="*70)
            
            reward_trainer = RewardTrainer(self.config)
            reward_trainer.setup(
                self.config.sft_model_path if self.config.run_sft else self.config.model.model_name
            )
            reward_trainer.train(
                preference_data,
                eval_data=eval_preference_data,
            )
            
            results["reward"] = {
                "output_dir": self.config.reward.output_dir,
                "status": "completed",
            }
            
            print("\n" + "="*70)
            print("REWARD MODEL TRAINING COMPLETED")
            print(f"Model saved to: {self.config.reward.output_dir}")
            print("="*70)
        
        # Stage 3: PPO
        if self.config.run_ppo:
            print("\n" + "="*70)
            print("STAGE 3: PPO TRAINING")
            print("="*70)
            
            # 如果没有提供PPO查询，从demonstration数据中提取
            if ppo_queries is None:
                ppo_queries = [d.prompt for d in demonstration_data]
            
            ppo_trainer = PPOTrainer(self.config)
            ppo_trainer.setup(
                actor_model_name=self.config.sft_model_path if self.config.run_sft else self.config.model.model_name,
                reward_model_path=self.config.reward_model_path,
            )
            ppo_trainer.train(ppo_queries)
            
            results["ppo"] = {
                "output_dir": self.config.ppo.output_dir,
                "status": "completed",
            }
            
            print("\n" + "="*70)
            print("PPO TRAINING COMPLETED")
            print(f"Model saved to: {self.config.ppo.output_dir}")
            print("="*70)
        
        # 保存结果
        results_path = os.path.join(self.config.output_root, "results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "="*70)
        print("RLHF PIPELINE COMPLETED")
        print(f"Results saved to: {results_path}")
        print("="*70)
        
        return results
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_dir: str):
        """从检查点加载Pipeline配置"""
        config_path = os.path.join(checkpoint_dir, "config.json")
        with open(config_path) as f:
            config_dict = json.load(f)
        
        config = RLHFConfig(**config_dict)
        return cls(config)
