"""
测试 ComputerUseAgent 的端到端主循环（使用 mock backend）
"""
import sys
from unittest.mock import MagicMock

# mock pyautogui
mock_pyautogui = MagicMock()
mock_pyautogui.FAILSAFE = True
mock_pyautogui.PAUSE = 0.1
mock_pyautogui.screenshot.return_value = MagicMock()
mock_pyautogui.screenshot.return_value.convert.return_value.resize.return_value.getdata.return_value = [0] * (320 * 180)
mock_pyautogui.moveTo = MagicMock()
mock_pyautogui.click = MagicMock()
mock_pyautogui.typewrite = MagicMock()
mock_pyautogui.press = MagicMock()
sys.modules["pyautogui"] = mock_pyautogui

from src.agent import ComputerUseAgent


def test_agent_run_with_mock_backend():
    agent = ComputerUseAgent(perception_backend="mock", max_steps=3)
    history = agent.run("点击 OK 按钮")
    assert len(history) >= 1
    assert history[0].plan is not None
    assert history[0].plan.action == "click"


def test_save_and_load_macro():
    agent = ComputerUseAgent(perception_backend="mock", max_steps=2)
    agent.run("点击按钮")
    agent.save_macro("click_demo", trigger="点击任意按钮")

    macros = agent.memory.list_macros()
    assert "click_demo" in macros

    loaded = agent.load_macro("click_demo")
    assert loaded is not None
    assert loaded.name == "click_demo"
    assert len(loaded.steps) >= 1

    agent.memory.delete("click_demo")
    assert "click_demo" not in agent.memory.list_macros()
