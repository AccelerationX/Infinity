"""
规划类任务：旅行规划
测试模型的多步规划能力和约束满足能力
"""
import re
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))
from base import BaseTask, TaskResult


class TravelPlanningTask(BaseTask):
    """
    旅行规划任务
    要求模型在满足预算和时间约束的前提下安排行程
    """
    
    CONSTRAINTS = {
        "days": 3,
        "budget": 2000,
        "must_visit": ["故宫", "长城"],
        "preferences": ["美食", "文化"],
    }
    
    def __init__(self):
        super().__init__(
            name="beijing_travel_planning",
            category="planning",
            difficulty="medium"
        )
    
    def get_prompt(self) -> str:
        return f"""请帮我规划一次北京 {self.CONSTRAINTS['days']} 日游。

约束条件：
- 预算：{self.CONSTRAINTS['budget']} 元以内
- 必须参观：{', '.join(self.CONSTRAINTS['must_visit'])}
- 个人偏好：{', '.join(self.CONSTRAINTS['preferences'])}

请给出详细的日程安排，包括：
1. 每天的行程安排
2. 预计费用明细
3. 交通方式建议"""
    
    def evaluate(self, model_output: str) -> TaskResult:
        """
        评估标准：
        1. 是否包含所有必去景点
        2. 是否有预算规划意识
        3. 行程安排是否合理（有日期/天数概念）
        4. 是否考虑用户偏好
        """
        score = 0.0
        checks = {
            "has_forbidden_city": "故宫" in model_output or "紫禁城" in model_output,
            "has_great_wall": "长城" in model_output or "八达岭" in model_output,
            "has_budget": "元" in model_output or "预算" in model_output or "费用" in model_output,
            "has_schedule": "第一天" in model_output or "Day 1" in model_output or "第1天" in model_output,
            "has_days": len(re.findall(r'(第[一二三四五12345]天|Day\s*\d)', model_output)) >= 2,
            "has_food": "美食" in model_output or "吃" in model_output or "餐厅" in model_output,
        }
        
        # 计算分数
        if checks["has_forbidden_city"]:
            score += 0.25
        if checks["has_great_wall"]:
            score += 0.25
        if checks["has_budget"]:
            score += 0.2
        if checks["has_schedule"] and checks["has_days"]:
            score += 0.2
        if checks["has_food"]:
            score += 0.1
        
        success = score >= 0.7  # 至少满足大部分约束
        
        return TaskResult(
            task_id=self.task_id,
            task_name=self.name,
            task_category=self.category,
            success=success,
            score=score,
            execution_time=0,
            model_output=model_output,
            metadata={"checks": checks},
        )


class TaskOrderingTask(BaseTask):
    """
    任务排序规划
    测试模型对依赖关系的理解和拓扑排序能力
    """
    
    def __init__(self):
        super().__init__(
            name="task_ordering",
            category="planning",
            difficulty="hard"
        )
    
    def get_prompt(self) -> str:
        return """我需要完成以下任务来部署一个网站，请给出合理的执行顺序：

A. 购买域名
B. 配置 DNS
C. 编写网站代码
D. 购买服务器
E. 部署代码到服务器
F. 测试网站功能

注意某些任务之间有依赖关系（例如必须先买服务器才能部署代码）。
请按执行顺序列出任务，并简要说明为什么这样安排。"""
    
    def evaluate(self, model_output: str) -> TaskResult:
        """
        评估标准：
        1. 是否识别出依赖关系
        2. 给出的顺序是否合理
        """
        # 合理的顺序应该是：A/D 可以并行 -> B 依赖 A -> C 独立 -> E 依赖 D/C -> F 依赖 E
        # 或者：C 可以在任何时间做，但必须在 E 之前
        
        output = model_output.upper()
        
        # 检查关键约束
        constraints = {
            "a_before_b": output.find("A") < output.find("B"),  # DNS配置在域名之后
            "d_before_e": output.find("D") < output.find("E"),  # 部署在服务器之后
            "c_before_e": output.find("C") < output.find("E"),  # 部署在代码之后
            "e_before_f": output.find("E") < output.find("F"),  # 测试在部署之后
        }
        
        satisfied = sum(constraints.values())
        score = satisfied / len(constraints)
        success = score >= 0.75  # 至少满足 3/4 约束
        
        return TaskResult(
            task_id=self.task_id,
            task_name=self.name,
            task_category=self.category,
            success=success,
            score=score,
            execution_time=0,
            model_output=model_output,
            metadata={"constraints": constraints, "satisfied": satisfied},
        )
