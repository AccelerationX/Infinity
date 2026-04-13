"""
工作区管理
持久化任务和项目数据
"""
import os
import json
from pathlib import Path
from typing import List, Optional

from .job import Job


class Workspace:
    """
    工作区
    
    管理任务和项目的持久化存储
    """
    
    def __init__(self, path: str):
        self.path = Path(path)
        self.jobs_dir = self.path / "jobs"
        self.projects_dir = self.path / "projects"
    
    def ensure_exists(self):
        """确保目录存在"""
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.projects_dir.mkdir(parents=True, exist_ok=True)
    
    def create_project(self, project_id: str, name: str) -> Path:
        """
        创建项目目录
        
        Returns:
            项目路径
        """
        project_path = self.projects_dir / f"{project_id}_{name[:20]}"
        project_path.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        (project_path / "docs").mkdir(exist_ok=True)
        (project_path / "code").mkdir(exist_ok=True)
        (project_path / "tests").mkdir(exist_ok=True)
        (project_path / "config").mkdir(exist_ok=True)
        
        return project_path
    
    def save_job(self, job: Job):
        """保存任务"""
        self.ensure_exists()
        
        job_path = self.jobs_dir / f"{job.id}.json"
        job.save_to_file(str(job_path))
    
    def load_job(self, job_id: str) -> Optional[Job]:
        """加载任务"""
        job_path = self.jobs_dir / f"{job_id}.json"
        
        if not job_path.exists():
            return None
        
        try:
            return Job.load_from_file(str(job_path))
        except Exception:
            return None
    
    def list_jobs(self) -> List[Job]:
        """列出所有任务"""
        self.ensure_exists()
        
        jobs = []
        for job_file in self.jobs_dir.glob("*.json"):
            try:
                job = Job.load_from_file(str(job_file))
                jobs.append(job)
            except Exception:
                continue
        
        return jobs
    
    def delete_job(self, job_id: str) -> bool:
        """删除任务"""
        job_path = self.jobs_dir / f"{job_id}.json"
        
        if job_path.exists():
            job_path.unlink()
            return True
        return False
    
    def get_project_path(self, project_id: str) -> Optional[Path]:
        """获取项目路径"""
        for project_dir in self.projects_dir.iterdir():
            if project_dir.is_dir() and project_dir.name.startswith(project_id):
                return project_dir
        return None
    
    def clean_old_jobs(self, days: int = 30):
        """清理旧任务"""
        import time
        
        cutoff = time.time() - (days * 24 * 60 * 60)
        
        for job_file in self.jobs_dir.glob("*.json"):
            try:
                if job_file.stat().st_mtime < cutoff:
                    job_file.unlink()
            except Exception:
                pass
