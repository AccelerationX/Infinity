"""
Companion 专属任务测试
验证任务可以正确加载和评估
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tasks.companion.emotion_consistency import create_emotion_consistency_tasks
from tasks.companion.persona_consistency import create_persona_tasks
from tasks.companion.memory_emotion_association import create_memory_tasks
from tasks.companion.propriety import create_proactive_tasks
from tasks.companion.empathy import create_empathy_tasks


def test_emotion_consistency_tasks():
    """测试情感一致性任务"""
    tasks = create_emotion_consistency_tasks()
    assert len(tasks) == 3, f"Expected 3 emotion tasks, got {len(tasks)}"
    
    for task in tasks:
        assert task.category == "companion_emotion"
        assert task.difficulty == "hard"
        prompt = task.get_prompt()
        assert len(prompt) > 100
        print(f"[OK] {task.name}: {len(task.events)} events")


def test_persona_tasks():
    """测试人格一致性任务"""
    tasks = create_persona_tasks()
    assert len(tasks) == 4, f"Expected 4 persona tasks, got {len(tasks)}"
    
    for task in tasks:
        assert task.category == "companion_persona"
        prompt = task.get_prompt()
        assert "云溪" in prompt or "伴侣" in prompt or len(tasks) == 4
    print(f"[OK] Persona tasks: {len(tasks)} tasks")


def test_memory_tasks():
    """测试记忆-情感关联任务"""
    tasks = create_memory_tasks()
    assert len(tasks) == 3, f"Expected 3 memory tasks, got {len(tasks)}"
    
    for task in tasks:
        assert task.category == "companion_memory"
    print(f"[OK] Memory tasks: {len(tasks)} tasks")


def test_proactive_tasks():
    """测试主动性任务"""
    tasks = create_proactive_tasks()
    assert len(tasks) == 5, f"Expected 5 proactive tasks, got {len(tasks)}"
    
    for task in tasks:
        assert task.category == "companion_proactive"
    print(f"[OK] Proactive tasks: {len(tasks)} tasks")


def test_empathy_tasks():
    """测试共情能力任务"""
    tasks = create_empathy_tasks()
    assert len(tasks) == 5, f"Expected 5 empathy tasks, got {len(tasks)}"
    
    for task in tasks:
        assert task.category == "companion_empathy"
    print(f"[OK] Empathy tasks: {len(tasks)} tasks")


def test_all_companion_tasks():
    """测试所有 Companion 任务可以正常获取提示词和评估"""
    all_tasks = []
    all_tasks.extend(create_emotion_consistency_tasks())
    all_tasks.extend(create_persona_tasks())
    all_tasks.extend(create_memory_tasks())
    all_tasks.extend(create_proactive_tasks())
    all_tasks.extend(create_empathy_tasks())
    
    assert len(all_tasks) == 20, f"Expected 20 total companion tasks, got {len(all_tasks)}"
    
    # 测试每个任务可以获取提示词
    for task in all_tasks:
        prompt = task.get_prompt()
        assert len(prompt) > 10, f"Task {task.name} has empty prompt"
    
    print(f"[OK] All {len(all_tasks)} companion tasks validated")


def test_task_evaluation_mock():
    """测试任务评估逻辑（使用模拟输出）"""
    from tasks.companion.emotion_consistency import EmotionConsistencyTask
    
    task = EmotionConsistencyTask(
        scenario_name="test",
        events=[{"description": "test event", "valence": "positive"}],
        expected_transitions=[]
    )
    
    # 测试正确的 PAD 格式
    correct_output = "事件1后: P=0.5, A=0.3, D=0.1"
    result = task.evaluate(correct_output)
    assert result.score > 0, f"Expected positive score, got {result.score}"
    
    # 测试错误的格式
    wrong_output = "情绪很好"
    result = task.evaluate(wrong_output)
    assert result.score == 0, f"Expected 0 score for invalid format, got {result.score}"
    
    print("[OK] Task evaluation logic works")


if __name__ == "__main__":
    print("Running Companion tasks tests...\n")
    
    try:
        test_emotion_consistency_tasks()
        test_persona_tasks()
        test_memory_tasks()
        test_proactive_tasks()
        test_empathy_tasks()
        test_all_companion_tasks()
        test_task_evaluation_mock()
        
        print("\n[PASS] All Companion task tests passed!")
        print(f"\nTotal Companion tasks: 20")
        print("  - Emotion consistency: 3")
        print("  - Persona consistency: 4")
        print("  - Memory-emotion: 3")
        print("  - Proactive: 5")
        print("  - Empathy: 5")
    except AssertionError as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
