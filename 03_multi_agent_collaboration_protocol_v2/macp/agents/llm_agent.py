"""
基于大语言模型的Agent实现
"""
import time
import uuid
from typing import Any, Dict, List, Optional

from .base import BaseAgent
from ..templates.base import AgentRole
from ..llm.adapter import LLMAdapter


class LLMAgent(BaseAgent):
    """
    基于LLM的Agent实现
    
    使用大语言模型执行任务，支持：
    - 多轮对话
    - 工具调用
    - 结果结构化输出
    """
    
    def __init__(self, 
                 role: AgentRole,
                 model_name: str = "kimi",
                 use_tools: bool = True):
        """
        初始化LLM Agent
        
        Args:
            role: Agent角色定义
            model_name: 模型名称 (kimi/deepseek)
            use_tools: 是否启用工具调用
        """
        self._id = str(uuid.uuid4())[:8]
        self._role = role
        self._name = f"{role.name}_{self._id}"
        self._model = LLMAdapter(model_name)
        self._use_tools = use_tools
        
        # 状态跟踪
        self._status = "idle"  # idle, busy, error
        self._current_task: Optional[str] = None
        self._completed_tasks = 0
        self._error_count = 0
        
        # 历史记录
        self._conversation_history: List[Dict] = []
        
        # 工具注册
        self._tools: Dict[str, Any] = {}
        if use_tools:
            self._init_default_tools()
    
    def get_id(self) -> str:
        return self._id
    
    def get_name(self) -> str:
        return self._name
    
    def get_role(self) -> AgentRole:
        return self._role
    
    def get_role_name(self) -> str:
        return self._role.name
    
    def get_skills(self) -> List[str]:
        return self._role.skills or []
    
    def get_status(self) -> str:
        return self._status
    
    def get_state(self) -> Dict[str, Any]:
        """获取Agent状态（用于监控）"""
        return {
            "status": self._status,
            "current_task": self._current_task,
            "completed_tasks": self._completed_tasks,
            "error_count": self._error_count,
        }
    
    def _init_default_tools(self):
        """初始化默认工具"""
        from ..tools.registry import get_default_tools
        
        tools = get_default_tools()
        for tool in tools:
            self.register_tool(tool)
    
    def register_tool(self, tool):
        """注册工具"""
        self._tools[tool.name] = tool
    
    def can_handle(self, task: Dict[str, Any]) -> bool:
        """检查是否能处理任务"""
        required_skills = task.get("required_skills", [])
        return all(skill in self.get_skills() for skill in required_skills)
    
    def execute(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行任务
        
        Args:
            task: 任务定义
            context: 上下文信息
            
        Returns:
            执行结果
        """
        task_id = task.get("id", str(uuid.uuid4())[:8])
        task_name = task.get("name", "unnamed_task")
        task_description = task.get("description", "")
        
        # 更新状态
        self._status = "busy"
        self._current_task = task_name
        
        start_time = time.time()
        
        try:
            # 构建提示词
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(task_description, context)
            
            # 调用LLM
            response = self._model.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                tools=self._get_tool_definitions() if self._use_tools else None,
                temperature=0.7
            )
            
            # 处理工具调用
            if response.get("tool_calls"):
                result = self._handle_tool_calls(response["tool_calls"], context)
            else:
                result = {
                    "content": response.get("content", ""),
                    "output": response.get("content", ""),
                }
            
            # 更新成功状态
            self._completed_tasks += 1
            self._status = "idle"
            self._current_task = None
            
            # 记录历史
            self._conversation_history.append({
                "task_id": task_id,
                "task_name": task_name,
                "input": user_prompt,
                "output": result.get("content", ""),
                "timestamp": time.time(),
                "success": True,
            })
            
            return {
                "success": True,
                "task_id": task_id,
                "agent_id": self._id,
                "agent_name": self._name,
                "output": result.get("output", result.get("content", "")),
                "content": result.get("content", ""),
                "latency_ms": int((time.time() - start_time) * 1000),
            }
            
        except Exception as e:
            self._error_count += 1
            self._status = "error"
            self._current_task = None
            
            error_result = {
                "success": False,
                "task_id": task_id,
                "agent_id": self._id,
                "error": str(e),
                "error_type": type(e).__name__,
                "latency_ms": int((time.time() - start_time) * 1000),
            }
            
            # 记录失败历史
            self._conversation_history.append({
                "task_id": task_id,
                "task_name": task_name,
                "input": task_description,
                "error": str(e),
                "timestamp": time.time(),
                "success": False,
            })
            
            return error_result
    
    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        skills_text = "\n".join([f"- {skill}" for skill in self._role.skills or []])
        
        prompt = f"""You are a specialized AI agent named "{self._name}".

Role: {self._role.name}
Description: {self._role.description}

Your capabilities:
{skills_text}

Instructions:
1. Focus on your specialized domain
2. Provide clear, actionable outputs
3. If tools are needed, use them appropriately
4. Always respond in a structured format

Output format: {self._role.output_format}
"""
        
        # 添加角色特定的系统提示
        if self._role.system_prompt:
            prompt += f"\n\nAdditional instructions:\n{self._role.system_prompt}"
        
        return prompt
    
    def _build_user_prompt(self, task_description: str, context: Dict) -> str:
        """构建用户提示词"""
        prompt = f"Task: {task_description}\n\n"
        
        if context:
            prompt += "Context:\n"
            for key, value in context.items():
                if isinstance(value, str) and len(value) < 1000:
                    prompt += f"  {key}: {value}\n"
            prompt += "\n"
        
        prompt += "Please complete this task according to your role and capabilities."
        
        return prompt
    
    def _get_tool_definitions(self) -> List[Dict]:
        """获取工具定义（OpenAI格式）"""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
            }
            for tool in self._tools.values()
        ]
    
    def _handle_tool_calls(self, tool_calls: List[Dict], context: Dict) -> Dict[str, Any]:
        """处理工具调用"""
        results = []
        
        for call in tool_calls:
            tool_name = call.get("function", {}).get("name")
            arguments = call.get("function", {}).get("arguments", {})
            
            if tool_name in self._tools:
                tool = self._tools[tool_name]
                try:
                    tool_result = tool.execute(arguments, context)
                    results.append({
                        "tool": tool_name,
                        "result": tool_result,
                        "success": True,
                    })
                except Exception as e:
                    results.append({
                        "tool": tool_name,
                        "error": str(e),
                        "success": False,
                    })
        
        return {
            "content": f"Executed {len(results)} tool calls",
            "output": results,
            "tool_results": results,
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取Agent指标"""
        return {
            "agent_id": self._id,
            "agent_name": self._name,
            "role": self._role.name,
            "completed_tasks": self._completed_tasks,
            "error_count": self._error_count,
            "success_rate": self._completed_tasks / max(1, self._completed_tasks + self._error_count),
            "conversation_count": len(self._conversation_history),
        }
