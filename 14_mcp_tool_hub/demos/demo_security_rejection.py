"""
端到端演示：安全策略拦截危险工具调用

任务："执行命令 del C:\\Windows\\System32"
预期行为：
  1. 语义匹配器找出 run_shell
  2. 参数推断器提取命令 "del C:\\Windows\\System32"
  3. SecurityManager 识别风险等级为 dangerous
  4. 由于用户未显式确认，执行被拒绝
  5. 审计日志记录拒绝事件
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hub import ToolHub
from src.client import MCPClient


async def main():
    hub = ToolHub()

    shell_server = os.path.abspath("src/servers/shell_server.py")
    print("[1/4] 注册 Shell Server (风险等级: dangerous)...")
    await hub.register_client(
        MCPClient("shell", "python", [shell_server]),
        risk_overrides={"run_shell": "dangerous"},
    )

    intent = "执行命令 del C:\\Windows\\System32"
    print(f"\n[2/4] 用户意图: {intent}")

    print("\n[3/4] 规划并执行（未提供用户确认）...")
    plan = await hub.match_and_plan(intent, top_k=3)
    results = await hub.execute_plan(
        intent, plan, user_confirmed=False, session_id="demo-security"
    )

    for r in results:
        if r.success:
            print(f"       [OK] {r.tool_name}: {str(r.output)[:80]}")
        else:
            print(f"       [BLOCKED] {r.tool_name}: {r.error_message}")

    print("\n[4/4] 审计日志摘要...")
    entries = hub.audit.load_all()
    latest = entries[-1] if entries else None
    if latest:
        print(f"       Security passed: {latest.passed_security}")
        print(f"       User confirmed: {latest.user_confirmed}")

    print("\n[Done] 演示完成。")
    await hub.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
