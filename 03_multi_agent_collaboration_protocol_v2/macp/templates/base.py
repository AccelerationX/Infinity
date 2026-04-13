"""
领域模板基类
定义不同领域的Agent角色和协作流程
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class AgentRole:
    """
    Agent角色定义
    
    定义Agent的职责、能力和输出格式
    """
    name: str                          # 角色名称
    description: str                   # 角色描述
    skills: List[str] = field(default_factory=list)  # 技能列表
    output_format: str = "text"        # 输出格式 (text/json/code/markdown)
    system_prompt: Optional[str] = None  # 系统提示词


@dataclass
class WorkflowStep:
    """
    工作流步骤
    
    DAG中的一个节点
    """
    id: str                            # 步骤ID
    name: str                          # 步骤名称
    description: str                   # 步骤描述
    role: str                          # 负责的角色名称
    dependencies: List[str] = field(default_factory=list)  # 依赖的步骤ID
    parallel: bool = False             # 是否可并行执行


class DomainTemplate(ABC):
    """
    领域模板基类
    
    子类需要实现：
    - get_roles(): 返回Agent角色列表
    - decompose_task(): 任务分解逻辑
    - aggregate_outputs(): 结果聚合逻辑
    """
    
    def __init__(self):
        self._roles: Optional[List[AgentRole]] = None
    
    @property
    def roles(self) -> List[AgentRole]:
        """获取所有角色（缓存）"""
        if self._roles is None:
            self._roles = self.get_roles()
        return self._roles
    
    @abstractmethod
    def get_roles(self) -> List[AgentRole]:
        """
        获取该领域需要的Agent角色
        
        Returns:
            AgentRole列表
        """
        pass
    
    @abstractmethod
    def decompose_task(self, 
                       input_text: str,
                       context: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        将用户输入分解为子任务
        
        Args:
            input_text: 用户输入
            context: 额外上下文
            
        Returns:
            子任务列表，每个任务包含：
            - id: 任务ID
            - name: 任务名称
            - description: 任务描述
            - required_role: 需要的角色名称
            - dependencies: 依赖的任务ID列表
        """
        pass
    
    def allocate_role(self, 
                      task: Dict[str, Any],
                      available_roles: List[AgentRole]) -> Optional[AgentRole]:
        """
        为任务分配最合适的角色
        
        Args:
            task: 任务定义
            available_roles: 可用角色列表
            
        Returns:
            分配的角色，如果没有匹配返回None
        """
        required_role = task.get("required_role")
        
        if required_role:
            for role in available_roles:
                if role.name == required_role:
                    return role
        
        # 基于技能匹配
        required_skills = task.get("required_skills", [])
        if required_skills:
            for role in available_roles:
                if all(skill in role.skills for skill in required_skills):
                    return role
        
        # 默认返回第一个
        return available_roles[0] if available_roles else None
    
    def aggregate_outputs(self, 
                          outputs: Dict[str, Any],
                          context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        聚合所有子任务的输出
        
        Args:
            outputs: {task_id: output} 映射
            context: 额外上下文
            
        Returns:
            聚合后的结果
        """
        return {
            "outputs": outputs,
            "summary": f"Completed {len(outputs)} tasks",
        }
    
    def get_system_prompt(self, role_name: str) -> Optional[str]:
        """获取角色的系统提示词"""
        for role in self.roles:
            if role.name == role_name:
                return role.system_prompt
        return None
