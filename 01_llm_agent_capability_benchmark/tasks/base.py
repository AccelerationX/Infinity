"""
任务基类定义
所有评测任务都需要继承 BaseTask 并实现相应方法
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import uuid
import time


@dataclass
class TaskResult:
    """任务执行结果"""
    task_id: str
    task_name: str
    task_category: str
    success: bool
    score: float  # 0-1
    execution_time: float  # 秒
    model_output: str
    expected_output: Optional[str] = None
    metadata: Dict[str, Any] = None
    error_message: Optional[str] = None


class BaseTask(ABC):
    """评测任务基类"""
    
    def __init__(self, name: str, category: str, difficulty: str = "easy"):
        self.task_id = str(uuid.uuid4())[:8]
        self.name = name
        self.category = category  # tool_use, planning, context, robustness
        self.difficulty = difficulty  # easy, medium, hard
        self.metadata = {}
    
    @abstractmethod
    def get_prompt(self) -> str:
        """获取任务的输入提示词"""
        pass
    
    @abstractmethod
    def evaluate(self, model_output: str) -> TaskResult:
        """
        评估模型输出
        返回 TaskResult 包含评分和元数据
        """
        pass
    
    def get_available_tools(self) -> Optional[List[Dict]]:
        """
        返回该任务可用的工具列表
        如果返回 None，表示该任务不需要工具
        """
        return None
    
    def pre_process(self) -> None:
        """任务执行前的预处理（可选）"""
        pass
    
    def post_process(self, result: TaskResult) -> TaskResult:
        """任务执行后的后处理（可选）"""
        return result
    
    def to_dict(self) -> Dict:
        """序列化为字典"""
        return {
            "task_id": self.task_id,
            "name": self.name,
            "category": self.category,
            "difficulty": self.difficulty,
        }
