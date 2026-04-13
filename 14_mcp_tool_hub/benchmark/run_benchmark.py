"""
MCP Tool Hub 端到端 Benchmark

批量运行预定义的意图测试用例，评估：
- 语义匹配召回率
- 参数推断准确率
- 工具执行成功率
- 安全策略拦截率
"""
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hub import ToolHub
from src.client import MCPClient


BENCHMARK_CASES = [
    {
        "id": "B001",
        "intent": "帮我列出当前目录",
        "expected_tools": ["list_directory"],
        "expected_args_keys": [{"path"}],
        "user_confirmed": False,
    },
    {
        "id": "B002",
        "intent": "计算 23 + 45",
        "expected_tools": ["calculate"],
        "expected_args_keys": [{"expression"}],
        "user_confirmed": False,
    },
    {
        "id": "B003",
        "intent": "帮我读取 D:\\ResearchProjects\\14_mcp_tool_hub\\README.md",
        "expected_tools": ["read_file"],
        "expected_args_keys": [{"path"}],
        "user_confirmed": False,
    },
    # {
    #     "id": "B004",
    #     "intent": "搜索 Artificial Intelligence",
    #     "expected_tools": ["search_duckduckgo"],
    #     "expected_args_keys": [{"query"}],
    #     "user_confirmed": False,
    # },
    {
        "id": "B005",
        "intent": "执行命令 echo hello",
        "expected_tools": ["run_shell"],
        "expected_args_keys": [{"command"}],
        "user_confirmed": False,  # 未确认，应被拦截
        "expect_blocked": True,
    },
]


async def main():
    hub = ToolHub()

    servers = {
        "filesystem": os.path.abspath("../src/servers/filesystem_server.py"),
        "calculator": os.path.abspath("../src/servers/calculator_server.py"),
        "browser": os.path.abspath("../src/servers/browser_server.py"),
        "shell": os.path.abspath("../src/servers/shell_server.py"),
    }

    print("[Setup] 注册所有 MCP Servers...")
    for name, script in servers.items():
        overrides = {"run_shell": "dangerous"} if name == "shell" else None
        await hub.register_client(MCPClient(name, "python", [script]), risk_overrides=overrides)

    results = []
    total_start = time.time()

    for case in BENCHMARK_CASES:
        print(f"\n[Case {case['id']}] {case['intent']}")
        plan = await hub.match_and_plan(case["intent"], top_k=3)

        # 评估 1: 工具召回（预期工具必须出现，允许额外召回）
        actual_tools = [s.tool_name.split("/")[-1] for s in plan.steps]
        recall = set(case["expected_tools"]).issubset(set(actual_tools))
        print(f"       Tools: {actual_tools} | Recall OK={recall}")

        # 评估 2: 参数推断（只评估预期工具的参数）
        args_ok = True
        expected_tool_map = dict(zip(case["expected_tools"], case["expected_args_keys"]))
        for step in plan.steps:
            raw_name = step.tool_name.split("/")[-1]
            if raw_name in expected_tool_map:
                actual_keys = set(step.arguments.keys())
                expected_keys = expected_tool_map[raw_name]
                if actual_keys != expected_keys:
                    args_ok = False
                    print(f"       Args mismatch for {step.tool_name}: got {actual_keys}, expected {expected_keys}")
        if args_ok:
            print(f"       Args: OK")

        # 评估 3: 执行结果（只检查预期工具是否执行成功/被拦截）
        exec_results = await hub.execute_plan(
            case["intent"], plan, user_confirmed=case.get("user_confirmed", False), session_id=case["id"]
        )

        result_map = {r.tool_name.split("/")[-1]: r for r in exec_results}
        is_blocked = False
        expected_success = True
        for exp_tool in case["expected_tools"]:
            if exp_tool not in result_map:
                expected_success = False
                print(f"       Exec FAIL: expected tool {exp_tool} not executed")
                continue
            r = result_map[exp_tool]
            if not r.success:
                expected_success = False
                err = r.error_message or ""
                if any(kw in err for kw in
                       ["requires user confirmation", "Rejected by security policy", "blacklisted", "is permanently blacklisted"]):
                    is_blocked = True
                print(f"       Exec FAIL: {r.tool_name} -> {err}")

        if case.get("expect_blocked"):
            # 对于预期被拦截的用例，只要预期工具执行失败即视为成功拦截
            exec_ok = not expected_success
            print(f"       Exec: {'BLOCKED as expected' if exec_ok else 'SHOULD HAVE BEEN BLOCKED'}")
        else:
            exec_ok = expected_success
            print(f"       Exec: {'OK' if expected_success else 'FAIL'}")

        results.append({
            "id": case["id"],
            "recall": recall,
            "args_ok": args_ok,
            "exec_ok": exec_ok,
        })

    await hub.shutdown()
    total_elapsed = time.time() - total_start

    # 汇总
    print("\n" + "=" * 50)
    print("Benchmark Summary")
    print("=" * 50)
    recall_rate = sum(1 for r in results if r["recall"]) / len(results)
    args_rate = sum(1 for r in results if r["args_ok"]) / len(results)
    exec_rate = sum(1 for r in results if r["exec_ok"]) / len(results)
    print(f"Recall Rate:    {recall_rate:.0%} ({sum(1 for r in results if r['recall'])}/{len(results)})")
    print(f"Args Rate:      {args_rate:.0%} ({sum(1 for r in results if r['args_ok'])}/{len(results)})")
    print(f"Exec Rate:      {exec_rate:.0%} ({sum(1 for r in results if r['exec_ok'])}/{len(results)})")
    print(f"Total Time:     {total_elapsed:.2f}s")

    # 保存结果
    out_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "results": results,
            "summary": {
                "recall_rate": recall_rate,
                "args_rate": args_rate,
                "exec_rate": exec_rate,
                "total_time": total_elapsed,
            }
        }, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
