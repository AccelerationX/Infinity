"""
端到端演示：打开记事本，让 Agent 输入一段文字，并验证屏幕变化
"""
import os
import subprocess
import sys
import time

# 确保项目根目录在路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import ComputerUseAgent
from src.evaluator import TaskEvaluator


def prepare_notepad():
    """关闭现有记事本，打开一个新的空文档并置顶"""
    print("[Setup] 关闭现有记事本进程...")
    os.system("taskkill /F /IM notepad.exe 2>nul")
    time.sleep(0.5)

    print("[Setup] 启动新记事本...")
    subprocess.Popen(["notepad.exe"])
    time.sleep(1.5)

    # 将记事本窗口置顶，并新建一个空文档、清空内容
    try:
        import pyautogui
        wins = pyautogui.getWindowsWithTitle("Notepad")
        if wins:
            wins[0].activate()
            time.sleep(0.3)
        # 新建文档（避免恢复旧会话）
        pyautogui.hotkey("ctrl", "n")
        time.sleep(0.3)
        # 全选并删除（确保内容为空）
        pyautogui.hotkey("ctrl", "a")
        pyautogui.keyDown("delete")
        pyautogui.keyUp("delete")
        time.sleep(0.2)
    except Exception as e:
        print(f"[Setup] 窗口准备跳过: {e}")


def main():
    prepare_notepad()

    print("\n[Agent] 启动 Computer Use Agent（UIA 后端）...\n")
    agent = ComputerUseAgent(
        perception_backend="uia",
        verifier_backend="pixel_diff",
        max_steps=5,
    )

    history = agent.run("在记事本中输入 'Hello from Computer Use Agent'")

    print("\n" + "=" * 50)
    print("执行结果摘要")
    print("=" * 50)
    for rec in history:
        plan = rec.plan
        print(
            f"Step {rec.step}: {plan.action} | "
            f"verify={rec.verification_passed} | {rec.verification_message}"
        )

    # 保存成功的工作流为宏
    agent.save_macro(
        name="notepad_hello",
        trigger="在记事本中输入 Hello from Computer Use Agent",
    )

    print("\n[Evaluator] 用剪贴板检查记事本内容...")
    # 确保记事本在前台再执行剪贴板检查
    try:
        import pyautogui
        wins = pyautogui.getWindowsWithTitle("Notepad")
        if wins:
            wins[0].activate()
            time.sleep(0.3)
    except Exception as e:
        print(f"[Evaluator] 窗口置顶跳过: {e}")

    evaluator = TaskEvaluator()
    ok, msg = evaluator.check_notepad_contains("Hello from Computer Use Agent")
    print(f"[Evaluator] {'PASS' if ok else 'FAIL'} — {msg}")

    print("\n[Done] 任务结束。请手动关闭记事本。")


if __name__ == "__main__":
    main()
