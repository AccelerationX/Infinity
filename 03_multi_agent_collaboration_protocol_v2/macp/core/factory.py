"""
工厂核心 - 用户的主要入口
"""
import os
import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Callable, Iterator
from datetime import datetime

from .job import Job, JobStatus, Task
from .scheduler import Scheduler
from .workspace import Workspace
from ..agents.registry import AgentRegistry


class Factory:
    """
    工厂 - 多Agent协作系统的主入口
    
    使用示例:
        factory = Factory(workspace="./projects")
        job = factory.submit("设计一个博客网站")
        result = factory.wait_for_completion(job.id)
    """
    
    def __init__(self, workspace: str = "./projects", config: Optional[dict] = None):
        """
        初始化工厂
        
        Args:
            workspace: 项目工作目录
            config: 配置选项
        """
        self.workspace = Workspace(workspace)
        self.registry = AgentRegistry()
        self.scheduler = Scheduler(self.registry, self.workspace)
        self.config = config or {}
        
        # 确保目录存在
        self.workspace.ensure_exists()
        
        # 加载已有项目
        self._load_existing_projects()
    
    def submit(self, project_name: str, description: str, 
               metadata: Optional[dict] = None) -> Job:
        """
        提交新任务
        
        Args:
            project_name: 项目名称
            description: 任务描述
            metadata: 额外元数据
            
        Returns:
            Job 对象
        """
        job_id = str(uuid.uuid4())[:8]
        
        job = Job(
            id=job_id,
            project_name=project_name,
            description=description,
            status=JobStatus.PENDING,
            created_at=datetime.now(),
            metadata=metadata or {}
        )
        
        # 创建工作目录
        project_path = self.workspace.create_project(job_id, project_name)
        job.workspace_path = str(project_path)
        
        # 保存任务
        self.workspace.save_job(job)
        
        # 提交到调度器
        self.scheduler.submit_job(job)
        
        return job
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """获取任务状态"""
        return self.workspace.load_job(job_id)
    
    def list_jobs(self, status: Optional[JobStatus] = None) -> List[Job]:
        """列出所有任务"""
        jobs = self.workspace.list_jobs()
        if status:
            jobs = [j for j in jobs if j.status == status]
        return jobs
    
    def wait_for_completion(self, job_id: str, 
                           timeout: Optional[float] = None,
                           poll_interval: float = 1.0) -> Job:
        """
        等待任务完成
        
        Args:
            job_id: 任务ID
            timeout: 超时时间（秒）
            poll_interval: 轮询间隔
            
        Returns:
            完成后的Job对象
        """
        import time
        start_time = time.time()
        
        while True:
            job = self.get_job(job_id)
            if not job:
                raise ValueError(f"Job {job_id} not found")
            
            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                return job
            
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Job {job_id} timeout")
            
            time.sleep(poll_interval)
    
    def track_progress(self, job_id: str) -> Iterator[dict]:
        """
        实时追踪任务进度（生成器）
        
        使用示例:
            for progress in factory.track_progress(job_id):
                print(f"进度: {progress['percent']}%")
        """
        import time
        last_update = None
        
        while True:
            job = self.get_job(job_id)
            if not job:
                break
            
            # 检查是否有更新
            current_update = job.updated_at
            if current_update != last_update:
                last_update = current_update
                yield {
                    "job_id": job_id,
                    "status": job.status.value,
                    "percent": job.progress_percent,
                    "current_task": job.current_task,
                    "completed_tasks": len([t for t in job.tasks if t.status == "completed"]),
                    "total_tasks": len(job.tasks),
                    "logs": job.recent_logs(5)
                }
            
            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                break
            
            time.sleep(0.5)
    
    def cancel_job(self, job_id: str) -> bool:
        """取消任务"""
        return self.scheduler.cancel_job(job_id)
    
    def pause_job(self, job_id: str) -> bool:
        """暂停任务"""
        return self.scheduler.pause_job(job_id)
    
    def resume_job(self, job_id: str) -> bool:
        """恢复任务"""
        return self.scheduler.resume_job(job_id)
    
    def get_result(self, job_id: str) -> dict:
        """
        获取任务结果
        
        Returns:
            {
                "job_id": str,
                "status": str,
                "output_path": str,
                "artifacts": List[str],  # 产出文件列表
                "summary": str  # 执行摘要
            }
        """
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
        
        artifacts = []
        if job.workspace_path:
            project_dir = Path(job.workspace_path)
            if project_dir.exists():
                # 收集所有产出文件
                for pattern in ["**/*.md", "**/*.json", "**/*.py", "**/*.js", "**/*.ts", "**/*.jsx", "**/*.tsx"]:
                    artifacts.extend([str(f.relative_to(project_dir)) for f in project_dir.glob(pattern)])
        
        return {
            "job_id": job_id,
            "status": job.status.value,
            "output_path": job.workspace_path,
            "artifacts": artifacts,
            "summary": job.summary,
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "duration_seconds": job.duration_seconds
        }
    
    def _load_existing_projects(self):
        """加载已存在的项目"""
        jobs = self.workspace.list_jobs()
        for job in jobs:
            if job.status in [JobStatus.PENDING, JobStatus.RUNNING]:
                # 恢复未完成的任务
                self.scheduler.submit_job(job)
