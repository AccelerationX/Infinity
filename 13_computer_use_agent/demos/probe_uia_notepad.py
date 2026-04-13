"""
UIA 后端探测脚本：打开记事本，用 UIA 解析控件树并打印
"""
import subprocess
import sys
import time

sys.path.insert(0, r"D:\ResearchProjects\13_computer_use_agent")

from src.perception import PerceptionEngine

print("[1/3] 启动记事本...")
subprocess.Popen(["notepad.exe"])
time.sleep(1.5)

print("[2/3] 使用 UIA 解析前台窗口...")
engine = PerceptionEngine(backend_name="uia")
obs = engine.observe()

print(f"\n检测到 {len(obs.elements)} 个可交互控件:\n")
for e in obs.elements[:20]:
    print(f"  [{e.element_id:10s}] {e.element_type:12s} '{e.text}'  bbox={e.bbox}")

print("\n[3/3] 原始描述:\n")
print(obs.raw_description[:800])

print("\n请手动关闭记事本。")
