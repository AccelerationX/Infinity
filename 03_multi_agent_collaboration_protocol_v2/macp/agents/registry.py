"""
Agent注册表
管理所有Agent的注册和发现
"""
import threading
from typing import Dict, List, Optional, Any

from .base import BaseAgent


class AgentRegistry:
    """
    Agent注册中心
    
    管理Agent的生命周期和状态
    """
    
    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}
        self._by_role: Dict[str, List[str]] = {}
        self._states: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
    
    def register(self, agent: BaseAgent):
        """注册Agent"""
        with self._lock:
            agent_id = agent.get_id()
            self._agents[agent_id] = agent
            
            # 按角色索引
            role_name = agent.get_role_name()
            if role_name not in self._by_role:
                self._by_role[role_name] = []
            self._by_role[role_name].append(agent_id)
            
            # 初始化状态
            self._states[agent_id] = agent.get_state()
    
    def unregister(self, agent_id: str) -> bool:
        """注销Agent"""
        with self._lock:
            if agent_id not in self._agents:
                return False
            
            agent = self._agents[agent_id]
            role_name = agent.get_role_name()
            
            # 移除索引
            if role_name in self._by_role:
                self._by_role[role_name].remove(agent_id)
            
            # 移除Agent和状态
            del self._agents[agent_id]
            del self._states[agent_id]
            
            return True
    
    def get(self, agent_id: str) -> Optional[BaseAgent]:
        """获取Agent"""
        return self._agents.get(agent_id)
    
    def get_all(self) -> List[BaseAgent]:
        """获取所有Agent"""
        return list(self._agents.values())
    
    def find_by_role(self, role_name: str) -> Optional[BaseAgent]:
        """按角色查找Agent（返回第一个可用）"""
        with self._lock:
            agent_ids = self._by_role.get(role_name, [])
            for agent_id in agent_ids:
                agent = self._agents.get(agent_id)
                if agent and agent.get_status() != "error":
                    return agent
            return None
    
    def find_all_by_role(self, role_name: str) -> List[BaseAgent]:
        """按角色查找所有Agent"""
        with self._lock:
            agent_ids = self._by_role.get(role_name, [])
            return [self._agents[aid] for aid in agent_ids if aid in self._agents]
    
    def update_state(self, agent_id: str, state: Dict[str, Any]):
        """更新Agent状态"""
        with self._lock:
            self._states[agent_id] = state
    
    def get_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """获取Agent状态"""
        with self._lock:
            return self._states.get(agent_id)
    
    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """获取所有Agent状态"""
        with self._lock:
            return self._states.copy()
    
    def get_available_agents(self, role_name: Optional[str] = None) -> List[BaseAgent]:
        """获取可用Agent"""
        agents = self.get_all()
        
        if role_name:
            agents = [a for a in agents if a.get_role_name() == role_name]
        
        # 过滤掉错误状态的Agent
        return [a for a in agents if a.get_status() != "error"]
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取注册表指标"""
        with self._lock:
            total = len(self._agents)
            by_role = {role: len(ids) for role, ids in self._by_role.items()}
            
            status_counts = {"idle": 0, "busy": 0, "error": 0}
            for state in self._states.values():
                status = state.get("status", "idle")
                status_counts[status] = status_counts.get(status, 0) + 1
            
            return {
                "total_agents": total,
                "by_role": by_role,
                "status_distribution": status_counts,
            }
