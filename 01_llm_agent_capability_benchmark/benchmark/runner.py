"""
评测任务执行器
负责任务的批量执行和结果收集
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# 添加项目根目录到路径
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from models.base import BaseModel
    from tasks.base import BaseTask, TaskResult
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Python path: {sys.path}")
    raise


class BenchmarkRunner:
    """评测任务运行器"""
    
    def __init__(self, model: BaseModel, output_dir: str = "./results"):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results: List[TaskResult] = []
    
    def run_task(self, task: BaseTask, verbose: bool = False) -> TaskResult:
        """运行单个任务"""
        if verbose:
            print(f"\n[Task] {task.name} ({task.difficulty})")
        
        # 预处理
        task.pre_process()
        
        # 获取提示词和工具
        prompt = task.get_prompt()
        tools = task.get_available_tools()
        
        if verbose:
            print(f"Prompt length: {len(prompt)} chars")
        
        # 调用模型
        model_output = self.model.generate(prompt, tools=tools)
        
        if "error" in model_output:
            result = TaskResult(
                task_id=task.task_id,
                task_name=task.name,
                task_category=task.category,
                success=False,
                score=0.0,
                execution_time=model_output.get("latency", 0),
                model_output="",
                error_message=model_output["error"],
                metadata={"model_info": self.model.get_model_info()},
            )
        else:
            # 评估输出
            raw_output = model_output["content"]
            if model_output.get("tool_calls"):
                raw_output += f"\n[Tool Calls: {json.dumps(model_output['tool_calls'], ensure_ascii=False)}]"
            
            result = task.evaluate(raw_output)
            result.execution_time = model_output.get("latency", 0)
            
            # 计算细化的评分维度
            detailed_scores = self._calculate_detailed_scores(
                task, model_output, result
            )
            
            result.metadata = {
                "model_info": self.model.get_model_info(),
                "token_usage": model_output.get("usage", {}),
                "tool_calls": model_output.get("tool_calls", []),
                "detailed_scores": detailed_scores,
            }
        
        # 后处理
        result = task.post_process(result)
        
        if verbose:
            print(f"Score: {result.score:.2f} | Time: {result.execution_time:.2f}s | Success: {result.success}")
        
        return result
    
    def _calculate_detailed_scores(self, task: BaseTask, model_output: Dict, result: TaskResult) -> Dict:
        """计算细化的评分维度"""
        detailed = {
            "accuracy": result.score,  # 准确性
            "efficiency": 0.0,  # 效率（响应时间）
            "tool_usage": 0.0,  # 工具使用能力
            "completeness": 0.0,  # 回答完整性
            "latency_score": 0.0,  # 延迟评分
        }
        
        # 效率评分（基于 token 使用效率）
        usage = model_output.get("usage", {})
        if usage:
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
            
            # 如果 completion_tokens 很少但得分高，说明效率高
            if completion_tokens > 0 and result.score > 0:
                # 期望每 100 token 产生一定质量的输出
                efficiency = min(1.0, (result.score * 500) / completion_tokens)
                detailed["efficiency"] = round(efficiency, 2)
        
        # 工具使用评分
        tool_calls = model_output.get("tool_calls", [])
        if task.get_available_tools():
            # 任务需要工具
            if tool_calls:
                # 正确使用了工具
                detailed["tool_usage"] = 1.0
                # 检查工具参数是否正确
                for tc in tool_calls:
                    if tc.get("name") == "calculate":
                        args = tc.get("arguments", "")
                        if "123" in args and "456" in args:
                            detailed["tool_usage"] = 1.0
                        elif "expression" in args:
                            detailed["tool_usage"] = 0.8
                        else:
                            detailed["tool_usage"] = 0.6
            else:
                # 应该使用工具但没有用
                detailed["tool_usage"] = 0.0
        else:
            # 任务不需要工具
            detailed["tool_usage"] = 1.0 if not tool_calls else 0.5  # 不需要却用了扣分
        
        # 完整性评分（基于输出长度和内容）
        content = model_output.get("content", "")
        if len(content) > 100:
            detailed["completeness"] = min(1.0, len(content) / 500)
        elif len(content) > 0:
            detailed["completeness"] = len(content) / 200
        else:
            detailed["completeness"] = 0.0
        
        # 延迟评分（越快越好）
        latency = model_output.get("latency", 10)
        if latency < 2:
            detailed["latency_score"] = 1.0
        elif latency < 5:
            detailed["latency_score"] = 0.8
        elif latency < 10:
            detailed["latency_score"] = 0.6
        elif latency < 20:
            detailed["latency_score"] = 0.4
        else:
            detailed["latency_score"] = 0.2
        
        return detailed
    
    def run_tasks(self, tasks: List[BaseTask], verbose: bool = False) -> List[TaskResult]:
        """批量运行任务"""
        self.results = []
        
        print(f"\n{'='*60}")
        print(f"Running {len(tasks)} tasks with model: {self.model.model_name}")
        print(f"{'='*60}")
        
        for i, task in enumerate(tasks, 1):
            print(f"\n[{i}/{len(tasks)}]", end=" ")
            result = self.run_task(task, verbose=verbose)
            self.results.append(result)
        
        return self.results
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """保存结果到文件"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.model.model_name}_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        data = {
            "model": self.model.get_model_info(),
            "timestamp": datetime.now().isoformat(),
            "total_tasks": len(self.results),
            "summary": self._get_summary(),
            "results": [
                {
                    "task_id": r.task_id,
                    "task_name": r.task_name,
                    "task_category": r.task_category,
                    "success": r.success,
                    "score": r.score,
                    "execution_time": r.execution_time,
                    "model_output": r.model_output,
                    "error_message": r.error_message,
                    "metadata": r.metadata,
                }
                for r in self.results
            ],
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"\nResults saved to: {filepath}")
        return str(filepath)
    
    def _get_summary(self) -> Dict:
        """生成结果摘要"""
        if not self.results:
            return {}
        
        total = len(self.results)
        successful = sum(1 for r in self.results if r.success)
        avg_score = sum(r.score for r in self.results) / total
        avg_time = sum(r.execution_time for r in self.results) / total
        
        # 按类别统计
        by_category = {}
        for r in self.results:
            cat = r.task_category
            if cat not in by_category:
                by_category[cat] = {
                    "total": 0, "success": 0, "score": 0, 
                    "time": 0, "detailed_scores": {}
                }
            by_category[cat]["total"] += 1
            if r.success:
                by_category[cat]["success"] += 1
            by_category[cat]["score"] += r.score
            by_category[cat]["time"] += r.execution_time
            
            # 累加细化评分
            detailed = r.metadata.get("detailed_scores", {})
            for key, value in detailed.items():
                if key not in by_category[cat]["detailed_scores"]:
                    by_category[cat]["detailed_scores"][key] = 0
                by_category[cat]["detailed_scores"][key] += value
        
        for cat in by_category:
            data = by_category[cat]
            data["avg_score"] = data["score"] / data["total"]
            data["success_rate"] = data["success"] / data["total"]
            data["avg_time"] = data["time"] / data["total"]
            # 平均细化评分
            for key in data["detailed_scores"]:
                data["detailed_scores"][key] /= data["total"]
        
        return {
            "total_tasks": total,
            "successful_tasks": successful,
            "success_rate": successful / total,
            "average_score": avg_score,
            "average_execution_time": avg_time,
            "by_category": by_category,
        }
