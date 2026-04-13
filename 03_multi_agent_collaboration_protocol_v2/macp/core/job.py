"""
任务定义和状态管理
"""
import json
from enum import Enum
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional


class JobStatus(Enum):
    """任务状态"""
    PENDING = "pending"      # 等待执行
    RUNNING = "running"      # 执行中
    PAUSED = "paused"        # 暂停
    COMPLETED = "completed"  # 完成
    FAILED = "failed"        # 失败
    CANCELLED = "cancelled"  # 取消


class TaskStatus(Enum):
    """子任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    BLOCKED = "blocked"      # 被阻塞（依赖未满足）
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"      # 被跳过


@dataclass
class Task:
    """子任务"""
    id: str
    name: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    agent_role: Optional[str] = None
    agent_id: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    progress_percent: int = 0
    output: Any = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    @property
    def is_ready(self) -> bool:
        """检查是否准备好执行"""
        return self.status == TaskStatus.PENDING
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "agent_role": self.agent_role,
            "agent_id": self.agent_id,
            "dependencies": self.dependencies,
            "progress_percent": self.progress_percent,
            "output": self.output,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


@dataclass
class Job:
    """
    协作任务
    
    包含多个子任务，支持DAG执行
    """
    id: str
    name: str
    description: str
    status: JobStatus = JobStatus.PENDING
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    tasks: List[Task] = field(default_factory=list)
    logs: List[Dict] = field(default_factory=list)
    workspace_path: Optional[str] = None
    summary: Optional[str] = None
    
    def add_log(self, message: str, level: str = "info"):
        """添加日志"""
        self.logs.append({
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message
        })
    
    def recent_logs(self, n: int = 20) -> List[Dict]:
        """获取最近n条日志"""
        return self.logs[-n:]
    
    @property
    def progress_percent(self) -> int:
        """计算总体进度"""
        if not self.tasks:
            return 0
        
        if self.status == JobStatus.COMPLETED:
            return 100
        
        total_progress = sum(t.progress_percent for t in self.tasks)
        return int(total_progress / len(self.tasks))
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """计算持续时间"""
        if self.started_at:
            end = self.completed_at or datetime.now()
            return (end - self.started_at).total_seconds()
        return None
    
    @property
    def ready_tasks(self) -> List[Task]:
        """获取准备就绪的任务"""
        completed_ids = {t.id for t in self.tasks if t.status == TaskStatus.COMPLETED}
        
        ready = []
        for task in self.tasks:
            if task.status == TaskStatus.PENDING:
                # 检查依赖是否满足
                if all(dep in completed_ids for dep in task.dependencies):
                    ready.append(task)
        
        return ready
    
    @property
    def running_tasks(self) -> List[Task]:
        """获取正在执行的任务"""
        return [t for t in self.tasks if t.status == TaskStatus.RUNNING]
    
    @property
    def failed_tasks(self) -> List[Task]:
        """获取失败的任务"""
        return [t for t in self.tasks if t.status == TaskStatus.FAILED]
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """获取指定任务"""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "tasks": [t.to_dict() for t in self.tasks],
            "logs": self.logs,
            "workspace_path": self.workspace_path,
            "summary": self.summary,
            "progress_percent": self.progress_percent,
            "duration_seconds": self.duration_seconds,
        }
    
    def save_to_file(self, path: str):
        """保存到文件"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_file(cls, path: str) -> "Job":
        """从文件加载"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 解析任务
        tasks = []
        for t_data in data.get("tasks", []):
            task = Task(
                id=t_data["id"],
                name=t_data["name"],
                description=t_data["description"],
                status=TaskStatus(t_data.get("status", "pending")),
                agent_role=t_data.get("agent_role"),
                agent_id=t_data.get("agent_id"),
                dependencies=t_data.get("dependencies", []),
                progress_percent=t_data.get("progress_percent", 0),
                output=t_data.get("output"),
                error=t_data.get("error"),
            )
            if t_data.get("started_at"):
                task.started_at = datetime.fromisoformat(t_data["started_at"])
            if t_data.get("completed_at"):
                task.completed_at = datetime.fromisoformat(t_data["completed_at"])
            tasks.append(task)
        
        # 解析Job
        job = cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            status=JobStatus(data.get("status", "pending")),
            tasks=tasks,
            logs=data.get("logs", []),
            workspace_path=data.get("workspace_path"),
            summary=data.get("summary"),
        )
        
        if data.get("created_at"):
            job.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("started_at"):
            job.started_at = datetime.fromisoformat(data["started_at"])
        if data.get("completed_at"):
            job.completed_at = datetime.fromisoformat(data["completed_at"])
        
        return job
