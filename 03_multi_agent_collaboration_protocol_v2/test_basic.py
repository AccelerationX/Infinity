"""
Basic test for MACP framework
Tests core functionality without LLM calls
"""
import sys
sys.path.insert(0, "D:\\ResearchProjects\\02_multi_agent_collaboration_protocol_v2")

from macp.core.job import Job, Task, JobStatus, TaskStatus
from macp.templates.software_dev import SoftwareDevelopmentTemplate


def test_job_creation():
    """Test job creation"""
    print("Testing job creation...")
    
    job = Job(
        id="test_001",
        name="Test Job",
        description="Test description"
    )
    
    assert job.id == "test_001"
    assert job.status == JobStatus.PENDING
    print("  Job creation: OK")


def test_task_creation():
    """Test task creation"""
    print("Testing task creation...")
    
    task = Task(
        id="task_1",
        name="Test Task",
        description="Test task description",
        dependencies=[]
    )
    
    assert task.id == "task_1"
    assert task.status == TaskStatus.PENDING
    assert task.is_ready == True
    print("  Task creation: OK")


def test_template_roles():
    """Test software dev template roles"""
    print("Testing template roles...")
    
    template = SoftwareDevelopmentTemplate()
    roles = template.get_roles()
    
    assert len(roles) == 7
    
    role_names = [r.name for r in roles]
    expected = ["ProductManager", "Architect", "TechLead", "FrontendDev", 
                "BackendDev", "QAEngineer", "DevOps"]
    
    for name in expected:
        assert name in role_names, f"Missing role: {name}"
    
    print(f"  Template has {len(roles)} roles: OK")


def test_task_decomposition():
    """Test task decomposition"""
    print("Testing task decomposition...")
    
    template = SoftwareDevelopmentTemplate()
    tasks = template.decompose_task("Create a blog website")
    
    assert len(tasks) > 0
    
    # Check task structure
    for task in tasks:
        assert "id" in task
        assert "name" in task
        assert "description" in task
        assert "required_role" in task
    
    print(f"  Decomposed into {len(tasks)} tasks: OK")


def test_job_progress():
    """Test job progress calculation"""
    print("Testing job progress...")
    
    job = Job(id="test", name="Test", description="Test")
    
    # Add tasks
    job.tasks = [
        Task(id="t1", name="Task 1", description="D1", progress_percent=100),
        Task(id="t2", name="Task 2", description="D2", progress_percent=50),
        Task(id="t3", name="Task 3", description="D3", progress_percent=0),
    ]
    
    progress = job.progress_percent
    expected = (100 + 50 + 0) // 3  # 50
    
    assert progress == expected, f"Expected {expected}, got {progress}"
    print(f"  Progress calculation ({progress}%): OK")


def test_role_allocation():
    """Test role allocation"""
    print("Testing role allocation...")
    
    template = SoftwareDevelopmentTemplate()
    roles = template.get_roles()
    
    # Test task with specific role requirement
    task = {
        "id": "test",
        "name": "Test",
        "description": "Test",
        "required_role": "Architect"
    }
    
    allocated = template.allocate_role(task, roles)
    assert allocated is not None
    assert allocated.name == "Architect"
    
    print(f"  Role allocation: OK ({allocated.name})")


def main():
    """Run all tests"""
    print("="*60)
    print("MACP Basic Tests")
    print("="*60 + "\n")
    
    tests = [
        test_job_creation,
        test_task_creation,
        test_template_roles,
        test_task_decomposition,
        test_job_progress,
        test_role_allocation,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
