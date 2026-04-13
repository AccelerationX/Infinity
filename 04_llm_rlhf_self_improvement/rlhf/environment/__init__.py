"""
环境模块

提供真实的Agent交互环境：
- 代码执行环境
- 工具调用环境
- 多轮对话管理
"""

from .code_env import CodeExecutionEnv
from .task_env import TaskEnvironment

__all__ = [
    "CodeExecutionEnv",
    "TaskEnvironment",
]
