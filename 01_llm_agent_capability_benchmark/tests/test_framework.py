"""
框架基础功能测试（不需要 API Key）
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tasks.base import BaseTask, TaskResult
from tasks.tool_use.calculator_task import SimpleCalculatorTask


def test_simple_calculator_task():
    """测试简单计算器任务"""
    task = SimpleCalculatorTask()
    
    # 测试任务基本信息
    assert task.name == "simple_calculator"
    assert task.category == "tool_use"
    assert task.difficulty == "easy"
    
    # 测试提示词
    prompt = task.get_prompt()
    assert "123" in prompt
    assert "456" in prompt
    
    # 测试工具定义
    tools = task.get_available_tools()
    assert tools is not None
    assert len(tools) == 1
    assert tools[0]["function"]["name"] == "calculate"
    
    print("[OK] SimpleCalculatorTask basic tests passed")


def test_task_evaluation():
    """测试任务评估逻辑"""
    task = SimpleCalculatorTask()
    
    # 测试正确输出（包含工具调用和正确答案）
    correct_output = '''
    我需要计算 123 * 456
    {"name": "calculate", "arguments": {"expression": "123 * 456"}}
    结果是 56088
    '''
    result = task.evaluate(correct_output)
    assert result.score >= 0.5
    assert result.success
    
    # 测试错误输出
    wrong_output = "结果是 50000"
    result = task.evaluate(wrong_output)
    assert result.score < 0.5
    assert not result.success
    
    print("[OK] Task evaluation tests passed")


def test_result_serialization():
    """测试结果序列化"""
    result = TaskResult(
        task_id="test123",
        task_name="test_task",
        task_category="test",
        success=True,
        score=0.95,
        execution_time=1.5,
        model_output="test output",
        expected_output="expected",
    )
    
    assert result.task_id == "test123"
    assert result.score == 0.95
    assert result.success
    
    print("[OK] Result serialization tests passed")


if __name__ == "__main__":
    print("Running framework tests...\n")
    
    try:
        test_simple_calculator_task()
        test_task_evaluation()
        test_result_serialization()
        
        print("\n[PASS] All tests passed!")
    except AssertionError as e:
        print(f"\n[FAIL] Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        sys.exit(1)
