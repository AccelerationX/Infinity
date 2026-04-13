"""
代码生成任务示例

使用RLHF训练一个代码生成Agent，让它从自己的错误中学习
"""
import sys
sys.path.insert(0, "D:\\ResearchProjects\\04_llm_rlhf_self_improvement")

from rlhf import RLHFConfig, SFTConfig, RewardConfig, PPOConfig, ModelConfig
from rlhf.pipeline import RLHFPipeline
from rlhf.data.schemas import DemonstrationData, PreferenceData
from rlhf.environment.code_env import CodeExecutionEnv
import random


def generate_synthetic_data(num_samples: int = 100):
    """生成合成训练数据"""
    
    # 简单的Python编程任务
    tasks = [
        {
            "prompt": "Write a function to calculate the factorial of a number n.",
            "completion": """def factorial(n):
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)""",
        },
        {
            "prompt": "Write a function to check if a number is prime.",
            "completion": """def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True""",
        },
        {
            "prompt": "Write a function to reverse a string.",
            "completion": """def reverse_string(s):
    return s[::-1]""",
        },
        {
            "prompt": "Write a function to calculate the sum of a list of numbers.",
            "completion": """def sum_list(numbers):
    total = 0
    for num in numbers:
        total += num
    return total""",
        },
        {
            "prompt": "Write a function to find the maximum number in a list.",
            "completion": """def find_max(numbers):
    if not numbers:
        return None
    max_num = numbers[0]
    for num in numbers:
        if num > max_num:
            max_num = num
    return max_num""",
        },
    ]
    
    # 生成demonstration数据
    demonstrations = []
    for i in range(num_samples):
        task = random.choice(tasks)
        demonstrations.append(DemonstrationData(
            prompt=task["prompt"],
            completion=task["completion"],
            metadata={"task_type": "code_generation"},
        ))
    
    # 生成preference数据（模拟人类偏好）
    preferences = []
    for i in range(num_samples // 2):
        task = random.choice(tasks)
        
        # chosen是好的回答
        chosen = task["completion"]
        
        # rejected是有问题的回答（模拟）
        # 例如：缺少边界检查
        if "factorial" in task["prompt"]:
            rejected = """def factorial(n):
    return n * factorial(n - 1)"""  # 缺少base case
        elif "prime" in task["prompt"]:
            rejected = """def is_prime(n):
    for i in range(2, n):
        if n % i == 0:
            return False
    return True"""  # 效率低
        else:
            rejected = task["completion"].replace("return", "print")  # 错误用法
        
        preferences.append(PreferenceData(
            prompt=task["prompt"],
            chosen=chosen,
            rejected=rejected,
            metadata={"task_type": "code_generation"},
        ))
    
    return demonstrations, preferences


def main():
    """主函数"""
    print("="*70)
    print("RLHF Code Generation Example")
    print("="*70)
    
    # 生成训练数据
    print("\nGenerating synthetic training data...")
    demonstration_data, preference_data = generate_synthetic_data(num_samples=50)
    print(f"Demonstration samples: {len(demonstration_data)}")
    print(f"Preference samples: {len(preference_data)}")
    
    # 分割训练/评估数据
    split_idx = int(len(demonstration_data) * 0.9)
    train_demos = demonstration_data[:split_idx]
    eval_demos = demonstration_data[split_idx:]
    
    split_idx = int(len(preference_data) * 0.9)
    train_prefs = preference_data[:split_idx]
    eval_prefs = preference_data[split_idx:]
    
    # 配置
    config = RLHFConfig(
        model=ModelConfig(
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            use_lora=True,
            lora_r=8,
            lora_alpha=16,
        ),
        sft=SFTConfig(
            output_dir="./outputs_code/sft",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            learning_rate=2e-5,
        ),
        reward=RewardConfig(
            output_dir="./outputs_code/reward",
            num_train_epochs=2,
            per_device_train_batch_size=2,
            learning_rate=1e-5,
        ),
        ppo=PPOConfig(
            output_dir="./outputs_code/ppo",
            num_rollouts=128,
            rollout_batch_size=16,
            ppo_epochs=2,
            learning_rate=1e-6,
        ),
        output_root="./outputs_code",
        run_sft=True,
        run_reward=True,
        run_ppo=True,
    )
    
    # 创建pipeline
    pipeline = RLHFPipeline(config)
    
    # 执行训练
    results = pipeline.run(
        demonstration_data=train_demos,
        preference_data=train_prefs,
        eval_demonstration_data=eval_demos,
        eval_preference_data=eval_prefs,
    )
    
    print("\n" + "="*70)
    print("Training completed!")
    print("="*70)
    
    # 测试最终模型
    print("\nTesting final model...")
    
    # 创建代码执行环境
    env = CodeExecutionEnv()
    
    test_prompts = [
        "Write a function to calculate the factorial of a number n.",
        "Write a function to check if a number is prime.",
        "Write a function to reverse a string.",
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 50)
        # 这里可以加载训练好的模型并生成代码
        # 为了演示，暂时跳过实际生成
        print("[Model output would be generated here]")
    
    print("\n" + "="*70)
    print("Example completed!")
    print("="*70)


if __name__ == "__main__":
    main()
