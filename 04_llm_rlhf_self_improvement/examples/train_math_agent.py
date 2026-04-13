"""
示例：训练数学Agent的自我改进

这个示例展示了如何使用RLHF框架训练一个数学计算Agent，
让它从自己的错误中学习，逐步提升计算准确率。
"""
import sys
import re
sys.path.insert(0, "D:\\ResearchProjects\\04_llm_rlhf_self_improvement")

from rlhf_agent import SelfImprovementLoop, ImprovementConfig
from rlhf_agent.config import RLHFConfig


def math_task_executor(query: str, response: str) -> dict:
    """
    数学任务执行器
    
    解析Agent的回答，计算正确性
    """
    # 从query中提取数字和运算符
    numbers = re.findall(r'\d+', query)
    if len(numbers) < 2:
        return {"success": False, "error": "Could not parse numbers"}
    
    a, b = int(numbers[0]), int(numbers[1])
    
    # 计算正确答案
    if "sum" in query.lower() or "plus" in query.lower():
        correct_answer = a + b
    elif "multiplied" in query.lower() or "product" in query.lower():
        correct_answer = a * b
    elif "minus" in query.lower() or "subtract" in query.lower():
        correct_answer = a - b
    elif "divide" in query.lower():
        correct_answer = a / b if b != 0 else float('inf')
    elif "power" in query.lower():
        correct_answer = a ** b
    else:
        return {"success": False, "error": "Unknown operation"}
    
    # 从response中提取数字
    response_numbers = re.findall(r'-?\d+\.?\d*', response.replace(",", ""))
    if not response_numbers:
        return {"success": False, "error": "No number in response"}
    
    try:
        agent_answer = float(response_numbers[-1])  # 取最后一个数字
    except ValueError:
        return {"success": False, "error": "Could not parse answer"}
    
    # 检查正确性
    is_correct = abs(agent_answer - correct_answer) < 0.01
    
    return {
        "success": is_correct,
        "correct_answer": correct_answer,
        "agent_answer": agent_answer,
        "query": query,
    }


def main():
    """主函数"""
    print("="*60)
    print("RLHF Self-Improvement: Math Agent")
    print("="*60)
    
    # 配置
    rlhf_config = RLHFConfig(
        # 使用小模型便于演示
        base_model_name="Qwen/Qwen2.5-0.5B-Instruct",
        
        # LoRA配置
        use_lora=True,
        lora_rank=8,
        lora_alpha=16,
        
        # PPO配置
        ppo_epochs=2,
        ppo_clip_epsilon=0.2,
        kl_penalty_coef=0.1,
        
        # 训练配置
        learning_rate=5e-5,
        batch_size=4,
        mini_batch_size=2,
        gradient_accumulation_steps=2,
        max_seq_length=256,
        max_new_tokens=128,
        
        # 经验回放
        replay_buffer_size=1000,
        
        # 设备
        device="cuda",  # 如果没有GPU会自动使用CPU
        mixed_precision=True,
        
        # 输出
        output_dir="./outputs/math_agent",
    )
    
    improvement_config = ImprovementConfig(
        iterations=5,  # 演示用，实际可以设置更大
        episodes_per_iteration=10,
        success_threshold=0.7,
        eval_interval=1,
        eval_episodes=10,
        checkpoint_interval=2,
        output_dir="./outputs/math_agent",
        use_rule_based_reward=True,  # 使用规则奖励冷启动
    )
    
    # 创建自我改进循环
    loop = SelfImprovementLoop(
        config=rlhf_config,
        improvement_config=improvement_config,
        task_executor=math_task_executor,
    )
    
    # 开始训练
    print("\nStarting training...")
    print("Note: This is a demonstration with limited iterations.")
    print("For full training, increase iterations and episodes.\n")
    
    history = loop.train()
    
    # 打印最终结果
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    
    if history:
        final_metrics = history[-1]["metrics"]
        print(f"\nFinal Results:")
        print(f"  Iterations: {len(history)}")
        print(f"  Success Rate: {final_metrics.get('eval_success_rate', 0):.2%}")
        print(f"  Mean Reward: {final_metrics.get('eval_mean_reward', 0):.4f}")
        
        # 计算改进幅度
        if len(history) > 1:
            initial_success = history[0]["metrics"].get("eval_success_rate", 0)
            final_success = final_metrics.get("eval_success_rate", 0)
            improvement = final_success - initial_success
            print(f"  Improvement: +{improvement:.2%}")
    
    print(f"\nOutput directory: {improvement_config.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
