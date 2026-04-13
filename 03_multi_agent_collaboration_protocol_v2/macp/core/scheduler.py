"""
任务调度器
管理DAG执行和Agent分配
"""
import threading
import time
from typing import Dict, Any, Optional, Callable, List
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime

from .job import Job, JobStatus, Task, TaskStatus
from .workspace import Workspace
from ..agents.registry import AgentRegistry


class Scheduler:
    """
    任务调度器
    
    负责任务调度：
    1. 基于DAG依赖执行
    2. Agent动态分配
    3. 并行/串行执行
    4. 失败重试
    """
    
    def __init__(self, 
                 registry: AgentRegistry,
                 workspace: Workspace,
                 max_workers: int = 3):
        self.registry = registry
        self.workspace = workspace
        self.max_workers = max_workers
        
        # 执行器
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # 任务跟踪
        self._active_futures: Dict[str, Future] = {}  # job_id -> Future
        self._paused_jobs: set = set()
        
        # 锁
        self._lock = threading.RLock()
        
        # 运行状态
        self._running = False
        self._scheduler_thread: Optional[threading.Thread] = None
    
    def start(self):
        """启动调度器"""
        with self._lock:
            if self._running:
                return
            
            self._running = True
            self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
            self._scheduler_thread.start()
    
    def shutdown(self):
        """关闭调度器"""
        with self._lock:
            self._running = False
            
        self.executor.shutdown(wait=True)
        
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
    
    def submit_job(self, 
                   job: Job,
                   on_progress: Optional[Callable[[Dict], None]] = None) -> bool:
        """
        提交任务到调度器
        
        Args:
            job: 任务对象
            on_progress: 进度回调
            
        Returns:
            是否成功提交
        """
        with self._lock:
            if job.id in self._active_futures:
                return False
        
        # 提交执行
        future = self.executor.submit(self._execute_job, job, on_progress)
        
        with self._lock:
            self._active_futures[job.id] = future
        
        return True
    
    def cancel_job(self, job_id: str) -> bool:
        """取消任务"""
        with self._lock:
            future = self._active_futures.get(job_id)
            if future and not future.done():
                future.cancel()
                return True
        return False
    
    def pause_job(self, job_id: str) -> bool:
        """暂停任务"""
        with self._lock:
            if job_id in self._active_futures:
                self._paused_jobs.add(job_id)
                return True
        return False
    
    def resume_job(self, job_id: str) -> bool:
        """恢复任务"""
        with self._lock:
            if job_id in self._paused_jobs:
                self._paused_jobs.remove(job_id)
                return True
        return False
    
    def _scheduler_loop(self):
        """调度器主循环"""
        while self._running:
            time.sleep(0.1)
            
            # 清理已完成的任务
            with self._lock:
                completed = [
                    job_id for job_id, future in self._active_futures.items()
                    if future.done()
                ]
                for job_id in completed:
                    del self._active_futures[job_id]
    
    def _execute_job(self, 
                     job: Job,
                     on_progress: Optional[Callable[[Dict], None]] = None):
        """
        执行任务
        
        基于DAG依赖执行所有子任务
        """
        try:
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now()
            job.add_log("任务开始执行")
            self.workspace.save_job(job)
            
            # 执行直到所有任务完成
            completed_tasks = set()
            failed = False
            
            while len(completed_tasks) < len(job.tasks):
                # 检查是否被取消
                if job.id not in self._active_futures:
                    job.status = JobStatus.CANCELLED
                    job.add_log("任务被取消")
                    break
                
                # 检查是否暂停
                if job.id in self._paused_jobs:
                    time.sleep(0.5)
                    continue
                
                # 获取就绪的任务
                ready_tasks = [
                    t for t in job.tasks
                    if t.status == TaskStatus.PENDING
                    and all(dep in completed_tasks for dep in t.dependencies)
                ]
                
                if not ready_tasks:
                    # 检查是否有任务正在执行
                    running = [t for t in job.tasks if t.status == TaskStatus.RUNNING]
                    if not running and len(completed_tasks) < len(job.tasks):
                        # 可能有死锁
                        blocked = [t for t in job.tasks if t.status == TaskStatus.BLOCKED]
                        if blocked:
                            job.add_log(f"警告: {len(blocked)} 个任务被阻塞", "warning")
                    
                    time.sleep(0.1)
                    continue
                
                # 执行任务（限制并行度）
                running_count = len([t for t in job.tasks if t.status == TaskStatus.RUNNING])
                available_slots = self.max_workers - running_count
                
                for task in ready_tasks[:available_slots]:
                    if failed:
                        break
                    
                    # 执行单个任务
                    self._execute_task(job, task, on_progress)
                    
                    # 检查失败
                    if task.status == TaskStatus.FAILED:
                        failed = True
                        job.add_log(f"任务失败: {task.name}", "error")
                
                # 更新进度
                if on_progress:
                    on_progress({
                        "job_id": job.id,
                        "progress": job.progress_percent,
                        "completed": len(completed_tasks),
                        "total": len(job.tasks),
                    })
                
                # 保存状态
                self.workspace.save_job(job)
                
                # 更新已完成集合
                completed_tasks = {
                    t.id for t in job.tasks
                    if t.status == TaskStatus.COMPLETED
                }
            
            # 任务完成
            if job.status == JobStatus.RUNNING:
                if failed:
                    job.status = JobStatus.FAILED
                    job.add_log("任务执行失败")
                else:
                    job.status = JobStatus.COMPLETED
                    job.add_log("任务执行完成")
                
                job.completed_at = datetime.now()
                self.workspace.save_job(job)
                
                # 最终进度回调
                if on_progress:
                    on_progress({
                        "job_id": job.id,
                        "progress": 100,
                        "completed": len(job.tasks),
                        "total": len(job.tasks),
                        "status": job.status.value,
                    })
            
        except Exception as e:
            job.status = JobStatus.FAILED
            job.add_log(f"调度器错误: {str(e)}", "error")
            job.completed_at = datetime.now()
            self.workspace.save_job(job)
    
    def _execute_task(self,
                      job: Job,
                      task: Task,
                      on_progress: Optional[Callable[[Dict], None]] = None):
        """执行单个任务"""
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        task.progress_percent = 10
        
        job.add_log(f"开始执行: {task.name}")
        self.workspace.save_job(job)
        
        try:
            # 查找Agent
            agent = None
            if task.agent_id:
                agent = self.registry.get(task.agent_id)
            
            if not agent and task.agent_role:
                agent = self.registry.find_by_role(task.agent_role)
            
            if not agent:
                raise Exception(f"找不到可用的Agent (role: {task.agent_role})")
            
            # 更新Agent状态
            self.registry.update_state(agent.get_id(), {
                "status": "busy",
                "current_task": task.name,
                "completed_tasks": agent.get_state().get("completed_tasks", 0),
            })
            
            # 准备上下文
            context = {
                "job_id": job.id,
                "job_name": job.name,
                "workspace_path": job.workspace_path,
                "task_dependencies": {},
            }
            
            # 获取依赖任务的输出
            for dep_id in task.dependencies:
                dep_task = job.get_task(dep_id)
                if dep_task and dep_task.output:
                    context["task_dependencies"][dep_id] = dep_task.output
            
            # 执行任务
            task.progress_percent = 50
            result = agent.execute(
                task={
                    "id": task.id,
                    "name": task.name,
                    "description": task.description,
                },
                context=context
            )
            
            # 处理结果
            if result.get("success"):
                task.status = TaskStatus.COMPLETED
                task.output = result.get("output")
                task.progress_percent = 100
                task.completed_at = datetime.now()
                
                job.add_log(f"完成: {task.name}")
                
                # 更新Agent状态
                current_completed = agent.get_state().get("completed_tasks", 0)
                self.registry.update_state(agent.get_id(), {
                    "status": "idle",
                    "current_task": None,
                    "completed_tasks": current_completed + 1,
                })
            else:
                raise Exception(result.get("error", "未知错误"))
                
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.progress_percent = 0
            
            job.add_log(f"失败: {task.name} - {str(e)}", "error")
            
            # 更新Agent状态
            if 'agent' in locals():
                self.registry.update_state(agent.get_id(), {
                    "status": "error",
                    "current_task": None,
                })
        
        self.workspace.save_job(job)
