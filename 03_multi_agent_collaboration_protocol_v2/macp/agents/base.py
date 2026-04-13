"""
Agent基类
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from ..templates.base import AgentRole


class BaseAgent(ABC):
    """
    Agent基类
    
    所有Agent必须实现此接口
    """
    
    @abstractmethod
    def get_id(self) -> str:
        """获取Agent唯一ID"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """获取Agent名称"""
        pass
    
    @abstractmethod
    def get_role(self) -> AgentRole:
        """获取Agent角色"""
        pass
    
    @abstractmethod
    def get_role_name(self) -> str:
        """获取角色名称"""
        pass
    
    @abstractmethod
    def get_skills(self) -> List[str]:
        """获取技能列表"""
        pass
    
    @abstractmethod
    def get_status(self) -> str:
        """获取当前状态"""
        pass
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """获取详细状态（用于监控）"""
        pass
    
    @abstractmethod
    def can_handle(self, task: Dict[str, Any]) -> bool:
        """检查是否能处理任务"""
        pass
    
    @abstractmethod
    def execute(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行任务
        
        Args:
            task: 任务定义
            context: 上下文信息
            
        Returns:
            执行结果
        """
        pass
