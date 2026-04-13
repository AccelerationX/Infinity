#!/usr/bin/env python3
"""
LLM Agent 能力评测框架 - 主入口
支持通用评测和 AI Companion 专属评测
支持模型: OpenAI GPT, Kimi, MiniMax, DeepSeek

使用方法:
    python run_benchmark.py --model kimi-k2.5
    python run_benchmark.py --model MiniMax-M2.7 --suite companion
    python run_benchmark.py --model deepseek-chat --category companion_emotion
"""
import argparse
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from benchmark.runner import BenchmarkRunner
from models import OpenAIModel, KimiModel, MiniMaxModel, DeepSeekModel

# 导入通用任务
from tasks.tool_use.calculator_task import SimpleCalculatorTask, MultiStepCalculatorTask
from tasks.planning.travel_planning_task import TravelPlanningTask, TaskOrderingTask
from tasks.context.memory_task import LongContextMemoryTask, MultiTurnConsistencyTask

# 导入 Companion 专属任务
from tasks.companion.emotion_consistency import create_emotion_consistency_tasks
from tasks.companion.persona_consistency import create_persona_tasks
from tasks.companion.memory_emotion_association import create_memory_tasks
from tasks.companion.propriety import create_proactive_tasks
from tasks.companion.empathy import create_empathy_tasks


# 任务注册表
ALL_TASKS = {
    "general": [
        SimpleCalculatorTask,
        MultiStepCalculatorTask,
        TravelPlanningTask,
        TaskOrderingTask,
        LongContextMemoryTask,
        MultiTurnConsistencyTask,
    ],
}


def get_all_tasks():
    """获取所有任务实例"""
    tasks = []
    
    # 通用任务
    for task_class in ALL_TASKS["general"]:
        tasks.append(task_class())
    
    # Companion 专属任务
    tasks.extend(create_emotion_consistency_tasks())
    tasks.extend(create_persona_tasks())
    tasks.extend(create_memory_tasks())
    tasks.extend(create_proactive_tasks())
    tasks.extend(create_empathy_tasks())
    
    return tasks


def get_tasks(category: str = None, difficulty: str = None, suite: str = None):
    """
    获取任务列表，支持按类别、难度、套件筛选
    """
    all_tasks = get_all_tasks()
    filtered_tasks = []
    
    for task in all_tasks:
        # 套件筛选
        if suite == "general" and task.category.startswith("companion"):
            continue
        if suite == "companion" and not task.category.startswith("companion"):
            continue
        
        # 类别筛选
        if category and task.category != category:
            continue
        
        # 难度筛选
        if difficulty and task.difficulty != difficulty:
            continue
        
        filtered_tasks.append(task)
    
    return filtered_tasks


def get_model(model_name: str, api_key: str = None):
    """
    根据模型名称创建对应的模型实例
    
    Args:
        model_name: 模型名称 (如 kimi-k2.5, MiniMax-M2.7, deepseek-chat, gpt-4o)
        api_key: 可选，直接使用提供的 API key
    """
    model_name_lower = model_name.lower()
    
    # Kimi 模型
    if "kimi" in model_name_lower:
        key = api_key or os.getenv("KIMI_API_KEY") or os.getenv("MOONSHOT_API_KEY")
        return KimiModel(model_name=model_name, api_key=key)
    
    # MiniMax 模型
    elif "minimax" in model_name_lower:
        key = api_key or os.getenv("MINIMAX_API_KEY")
        return MiniMaxModel(model_name=model_name, api_key=key)
    
    # DeepSeek 模型
    elif "deepseek" in model_name_lower:
        key = api_key or os.getenv("DEEPSEEK_API_KEY")
        return DeepSeekModel(model_name=model_name, api_key=key)
    
    # OpenAI 模型 (GPT 系列)
    elif "gpt" in model_name_lower or "openai" in model_name_lower:
        key = api_key or os.getenv("OPENAI_API_KEY")
        return OpenAIModel(model_name=model_name, api_key=key)
    
    else:
        # 默认尝试作为 OpenAI 兼容模型处理
        print(f"[WARN] Unknown model '{model_name}', trying as OpenAI-compatible...")
        key = api_key or os.getenv("OPENAI_API_KEY")
        return OpenAIModel(model_name=model_name, api_key=key)


def check_api_key(model_name: str) -> bool:
    """检查 API key 是否配置"""
    model_name_lower = model_name.lower()
    
    if "kimi" in model_name_lower:
        key = os.getenv("KIMI_API_KEY") or os.getenv("MOONSHOT_API_KEY")
        if not key:
            print("[ERROR] KIMI_API_KEY or MOONSHOT_API_KEY not set!")
            return False
    elif "minimax" in model_name_lower:
        key = os.getenv("MINIMAX_API_KEY")
        if not key:
            print("[ERROR] MINIMAX_API_KEY not set!")
            return False
    elif "deepseek" in model_name_lower:
        key = os.getenv("DEEPSEEK_API_KEY")
        if not key:
            print("[ERROR] DEEPSEEK_API_KEY not set!")
            return False
    elif "gpt" in model_name_lower:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            print("[ERROR] OPENAI_API_KEY not set!")
            return False
    
    return True


def print_summary_table(results_by_category: dict):
    """打印分类别的结果摘要"""
    print(f"\n{'='*70}")
    print("DETAILED SUMMARY BY CATEGORY")
    print(f"{'='*70}")
    print(f"{'Category':<25} {'Tasks':<8} {'Success':<10} {'Avg Score':<12} {'Avg Time'}")
    print("-"*70)
    
    for cat, data in sorted(results_by_category.items()):
        print(f"{cat:<25} {data['total']:<8} {data['success_rate']:<10.1%} {data['avg_score']:<12.2f} {data['avg_time']:.2f}s")
    
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Agent Companion Benchmark (ACB)")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name to evaluate (e.g., kimi-k2.5, MiniMax-M2.7, deepseek-chat, gpt-4o)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="Directly provide API key (optional, can use env var)"
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=["tool_use", "planning", "context", 
                 "companion_emotion", "companion_persona", "companion_memory", 
                 "companion_proactive", "companion_empathy"],
        help="Filter tasks by category"
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        choices=["easy", "medium", "hard"],
        help="Filter tasks by difficulty"
    )
    parser.add_argument(
        "--suite",
        type=str,
        choices=["general", "companion", "all"],
        default="all",
        help="Task suite to run (default: all)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results",
        help="Output directory for results (default: ./results)"
    )
    
    args = parser.parse_args()
    
    # 检查 API key
    if not args.api_key and not check_api_key(args.model):
        print("\nPlease set the appropriate API key environment variable or use --api-key")
        print("Examples:")
        print("  export KIMI_API_KEY='your-key'")
        print("  export MINIMAX_API_KEY='your-key'")
        print("  export DEEPSEEK_API_KEY='your-key'")
        return 1
    
    # 获取任务列表
    tasks = get_tasks(
        category=args.category, 
        difficulty=args.difficulty,
        suite=args.suite
    )
    
    if not tasks:
        print("No tasks match the given criteria!")
        return 1
    
    print(f"Found {len(tasks)} tasks to run")
    
    # 初始化模型
    print(f"\nInitializing model: {args.model}")
    model = get_model(args.model, api_key=args.api_key)
    
    if not model.is_available():
        print(f"Model {args.model} is not available! Check API key.")
        return 1
    
    print(f"Model ready: {model.model_name}")
    print(f"Base URL: {getattr(model, 'base_url', 'default')}")
    
    # 创建 runner 并运行评测
    runner = BenchmarkRunner(model=model, output_dir=args.output)
    results = runner.run_tasks(tasks, verbose=args.verbose)
    
    # 按类别统计
    by_category = {}
    for r in results:
        cat = r.task_category
        if cat not in by_category:
            by_category[cat] = {"total": 0, "success": 0, "score": 0, "time": 0}
        by_category[cat]["total"] += 1
        if r.success:
            by_category[cat]["success"] += 1
        by_category[cat]["score"] += r.score
        by_category[cat]["time"] += r.execution_time
    
    # 计算平均值
    for cat in by_category:
        data = by_category[cat]
        data["success_rate"] = data["success"] / data["total"]
        data["avg_score"] = data["score"] / data["total"]
        data["avg_time"] = data["time"] / data["total"]
    
    # 打印摘要
    summary = runner._get_summary()
    print(f"\n{'='*70}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*70}")
    print(f"Model: {args.model}")
    print(f"Total tasks: {summary['total_tasks']}")
    print(f"Success rate: {summary['success_rate']:.1%}")
    print(f"Average score: {summary['average_score']:.2f}")
    print(f"Avg execution time: {summary['average_execution_time']:.2f}s")
    
    # 打印详细分类结果
    print_summary_table(by_category)
    
    # 保存结果
    output_file = runner.save_results()
    
    print(f"\n[OK] Benchmark completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
