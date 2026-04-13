"""
端到端演示：自然语言 -> 语义匹配 -> DAG 规划 -> 参数推断 -> 安全执行 -> 审计

任务："帮我列出当前目录并计算 23+45"
预期行为：
  1. 语义匹配器找出 list_directory 和 calculate
  2. 参数推断器自动提取路径 "." 和表达式 "23+45"
  3. DAG 规划器按优先级排序（先探索目录，再计算）
  4. Hub 执行工具链并打印结果
  5. 审计日志记录完整链路
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hub import ToolHub
from src.client import MCPClient


async def main():
    hub = ToolHub()

    fs_server = os.path.abspath("src/servers/filesystem_server.py")
    calc_server = os.path.abspath("src/servers/calculator_server.py")

    print("[1/5] 注册 MCP Servers...")
    await hub.register_client(MCPClient("filesystem", "python", [fs_server]))
    await hub.register_client(MCPClient("calculator", "python", [calc_server]))

    registered_tools = [t.name for t in hub.list_tools()]
    print(f"       已注册工具: {registered_tools}")

    intent = "帮我列出当前目录并计算 23+45"
    print(f"\n[2/5] 用户意图: {intent}")

    print("\n[3/5] 语义匹配 + DAG 规划 + 参数推断...")
    plan = await hub.match_and_plan(intent, top_k=5)
    print(f"       计划风险等级: {plan.estimated_risk}")
    for s in plan.steps:
        print(f"       - {s.tool_name}: args={s.arguments} | reason={s.reason}")

    print("\n[4/5] 执行工具链...")
    results = await hub.execute_plan(intent, plan, session_id="demo-001")
    for r in results:
        output_preview = str(r.output)[:80].replace("\n", " ")
        status = "OK" if r.success else "FAIL"
        print(f"       [{status}] {r.tool_name}: {output_preview}")

    print("\n[5/5] 审计日志摘要...")
    entries = hub.audit.load_all()
    latest = entries[-1] if entries else None
    if latest:
        print(f"       Session: {latest.session_id}")
        print(f"       Intent: {latest.intent}")
        print(f"       Steps: {len(latest.executed_steps)}")
        print(f"       Security passed: {latest.passed_security}")

    print("\n[Done] 断开所有 Server 连接...")
    await hub.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
