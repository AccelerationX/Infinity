"""
MACP 核心框架 - 通用多Agent协作引擎
"""
import uuid
import threading
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
from dataclasses import dataclass, field

from .job import Job, JobStatus, Task, TaskStatus
from .scheduler import Scheduler
from .workspace import Workspace
from .event_bus import EventBus
from ..templates.base import DomainTemplate
from ..agents.registry import AgentRegistry


@dataclass
class FrameworkState:
    """框架状态"""
    is_running: bool = False
    active_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    alerts: List[Dict] = field(default_factory=list)  # 告警信息


class CollaborationFramework:
    """
    多Agent协作框架 - 核心引擎
    
    使用示例:
        framework = CollaborationFramework(template=SoftwareDevelopmentTemplate())
        job = framework.execute("创建一个博客网站")
        
        # 实时监控
        for state in framework.monitor_job(job.id):
            print(f"进度: {state['progress']}%, 当前任务: {state['current_task']}")
    """
    
    def __init__(self, 
                 template: DomainTemplate,
                 workspace_path: str = "./macp_workspace",
                 max_workers: int = 3):
        """
        初始化框架
        
        Args:
            template: 领域模板（定义角色和流程）
            workspace_path: 工作目录
            max_workers: 最大并行任务数
        """
        self.template = template
        self.workspace = Workspace(workspace_path)
        self.registry = AgentRegistry()
        self.scheduler = Scheduler(self.registry, self.workspace, max_workers)
        self.event_bus = EventBus()
        
        # 初始化模板中的角色
        self._init_agents_from_template()
        
        # 框架状态
        self.state = FrameworkState()
        self._state_lock = threading.RLock()
        
        # 注册事件监听
        self._register_event_handlers()
        
        # 确保目录存在
        self.workspace.ensure_exists()
    
    def _init_agents_from_template(self):
        """根据模板初始化Agent"""
        from ..agents.llm_agent import LLMAgent
        
        for role in self.template.roles:
            agent = LLMAgent(role=role)
            self.registry.register(agent)
    
    def _register_event_handlers(self):
        """注册事件处理器"""
        # 任务失败事件
        self.event_bus.subscribe("task_failed", self._on_task_failed)
        # 任务阻塞事件
        self.event_bus.subscribe("task_blocked", self._on_task_blocked)
        # Agent错误事件
        self.event_bus.subscribe("agent_error", self._on_agent_error)
    
    def execute(self, 
                input_text: str,
                job_name: Optional[str] = None,
                context: Optional[Dict] = None,
                on_progress: Optional[Callable[[Dict], None]] = None) -> Job:
        """
        执行协作任务
        
        Args:
            input_text: 用户输入的任务描述
            job_name: 任务名称（可选）
            context: 额外上下文
            on_progress: 进度回调函数
            
        Returns:
            Job 任务对象
        """
        # 创建任务
        job_id = str(uuid.uuid4())[:8]
        job = Job(
            id=job_id,
            name=job_name or f"任务-{job_id}",
            description=input_text,
            status=JobStatus.PENDING,
            created_at=datetime.now()
        )
        
        # 创建工作目录
        project_path = self.workspace.create_project(job_id, job.name)
        job.workspace_path = str(project_path)
        
        # 使用模板分解任务
        try:
            job.add_log("开始分析任务...")
            task_dicts = self.template.decompose_task(input_text, context)
            
            # 转换为Task对象
            for i, task_dict in enumerate(task_dicts):
                task = Task(
                    id=task_dict.get("id", f"task_{i}"),
                    name=task_dict["name"],
                    description=task_dict["description"],
                    dependencies=task_dict.get("dependencies", []),
                )
                
                # 分配角色
                role = self.template.allocate_role(task_dict, self.template.roles)
                if role:
                    task.agent_role = role.name
                    # 找到对应的Agent
                    agent = self.registry.find_by_role(role.name)
                    if agent:
                        task.agent_id = agent.get_id()
                
                job.tasks.append(task)
            
            job.add_log(f"任务分解完成，共 {len(job.tasks)} 个子任务")
            self.workspace.save_job(job)
            
            # 提交到调度器
            self.scheduler.submit_job(job, on_progress=on_progress)
            
            # 更新状态
            with self._state_lock:
                self.state.active_jobs += 1
            
            return job
            
        except Exception as e:
            job.status = JobStatus.FAILED
            job.add_log(f"任务初始化失败: {str(e)}")
            self.workspace.save_job(job)
            
            # 添加告警
            self._add_alert("error", f"任务 {job_id} 初始化失败", str(e))
            raise
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """获取任务状态"""
        return self.workspace.load_job(job_id)
    
    def list_jobs(self, status: Optional[JobStatus] = None) -> List[Job]:
        """列出所有任务"""
        jobs = self.workspace.list_jobs()
        if status:
            jobs = [j for j in jobs if j.status == status]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        获取任务详细状态（用于Web界面）
        
        Returns:
            {
                "job_id": str,
                "name": str,
                "status": str,
                "progress_percent": int,
                "current_task": str,
                "tasks": [...],
                "logs": [...],
                "alerts": [...]
            }
        """
        job = self.get_job(job_id)
        if not job:
            return None
        
        # 计算总体进度
        if job.tasks:
            completed = len([t for t in job.tasks if t.status == TaskStatus.COMPLETED])
            total = len(job.tasks)
            progress = int(completed / total * 100)
        else:
            progress = 0
        
        # 找出当前正在执行的任务
        current = None
        for task in job.tasks:
            if task.status == TaskStatus.RUNNING:
                current = task.name
                break
        
        return {
            "job_id": job.id,
            "name": job.name,
            "status": job.status.value,
            "progress_percent": progress,
            "current_task": current,
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "duration": job.duration_seconds,
            "tasks": [
                {
                    "id": t.id,
                    "name": t.name,
                    "status": t.status.value,
                    "agent_role": t.agent_role,
                    "progress": t.progress_percent,
                    "dependencies": t.dependencies
                }
                for t in job.tasks
            ],
            "logs": job.recent_logs(20),
            "summary": job.summary
        }
    
    def get_agent_status(self) -> List[Dict[str, Any]]:
        """
        获取所有Agent状态（用于Web界面）
        
        Returns:
            [
                {
                    "agent_id": str,
                    "name": str,
                    "role": str,
                    "status": "idle" | "busy" | "error",
                    "current_task": str,
                    "completed_tasks": int,
                    "skills": [...]
                }
            ]
        """
        agents = self.registry.get_all()
        result = []
        
        for agent in agents:
            state = self.registry.get_state(agent.get_id()) or {}
            
            result.append({
                "agent_id": agent.get_id()[:8],
                "name": agent.get_name(),
                "role": agent.get_role_name(),
                "status": state.get("status", "idle"),
                "current_task": state.get("current_task"),
                "completed_tasks": state.get("completed_tasks", 0),
                "skills": agent.get_skills()
            })
        
        return result
    
    def get_overall_progress(self) -> Dict[str, Any]:
        """
        获取整体进度（用于Web界面的总进度条）
        
        Returns:
            {
                "active_jobs": int,
                "completed_jobs": int,
                "failed_jobs": int,
                "total_jobs": int,
                "overall_progress": float  # 所有任务的平均进度
            }
        """
        jobs = self.workspace.list_jobs()
        
        active = len([j for j in jobs if j.status == JobStatus.RUNNING])
        completed = len([j for j in jobs if j.status == JobStatus.COMPLETED])
        failed = len([j for j in jobs if j.status == JobStatus.FAILED])
        
        # 计算平均进度
        if jobs:
            total_progress = sum(j.progress_percent for j in jobs)
            avg_progress = total_progress / len(jobs)
        else:
            avg_progress = 0
        
        return {
            "active_jobs": active,
            "completed_jobs": completed,
            "failed_jobs": failed,
            "total_jobs": len(jobs),
            "overall_progress": round(avg_progress, 1)
        }
    
    def get_alerts(self, level: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取告警信息
        
        Args:
            level: 过滤级别 ("error", "warning", "info")
        """
        with self._state_lock:
            alerts = self.state.alerts.copy()
        
        if level:
            alerts = [a for a in alerts if a["level"] == level]
        
        # 按时间倒序
        alerts.sort(key=lambda a: a["timestamp"], reverse=True)
        return alerts
    
    def cancel_job(self, job_id: str) -> bool:
        """取消任务"""
        return self.scheduler.cancel_job(job_id)
    
    def pause_job(self, job_id: str) -> bool:
        """暂停任务"""
        return self.scheduler.pause_job(job_id)
    
    def resume_job(self, job_id: str) -> bool:
        """恢复任务"""
        return self.scheduler.resume_job(job_id)
    
    def _on_task_failed(self, event: Dict):
        """任务失败处理"""
        job_id = event.get("job_id")
        task_name = event.get("task_name")
        error = event.get("error")
        
        self._add_alert("error", f"任务 {job_id} 中的 '{task_name}' 失败", error)
    
    def _on_task_blocked(self, event: Dict):
        """任务阻塞处理"""
        job_id = event.get("job_id")
        task_name = event.get("task_name")
        reason = event.get("reason")
        
        self._add_alert("warning", f"任务 {job_id} 中的 '{task_name}' 被阻塞", reason)
    
    def _on_agent_error(self, event: Dict):
        """Agent错误处理"""
        agent_id = event.get("agent_id")
        agent_name = event.get("agent_name")
        error = event.get("error")
        
        self._add_alert("error", f"Agent {agent_name} 发生错误", error)
    
    def _add_alert(self, level: str, title: str, message: str):
        """添加告警"""
        with self._state_lock:
            self.state.alerts.append({
                "level": level,
                "title": title,
                "message": message,
                "timestamp": datetime.now().isoformat()
            })
            # 保留最近100条
            self.state.alerts = self.state.alerts[-100:]
    
    def clear_alerts(self):
        """清空告警"""
        with self._state_lock:
            self.state.alerts.clear()
    
    def shutdown(self):
        """关闭框架"""
        self.scheduler.shutdown()
