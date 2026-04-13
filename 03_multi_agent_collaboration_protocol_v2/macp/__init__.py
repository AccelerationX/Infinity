"""
MACP - Multi-Agent Collaboration Protocol
通用多Agent协作框架

使用示例:
    from macp import CollaborationFramework
    from macp.templates.software_dev import SoftwareDevelopmentTemplate
    
    # 初始化
    framework = CollaborationFramework(SoftwareDevelopmentTemplate())
    
    # 执行任务
    job = framework.execute("创建一个博客网站")
    
    # 监控状态
    status = framework.get_job_status(job.id)
"""

__version__ = "2.0.0"

from .core.framework import CollaborationFramework
from .core.job import Job, Task, JobStatus, TaskStatus
from .templates.base import DomainTemplate, AgentRole

__all__ = [
    "CollaborationFramework",
    "Job",
    "Task", 
    "JobStatus",
    "TaskStatus",
    "DomainTemplate",
    "AgentRole",
]
