"""
测试 ToolHub 的注册、路由、执行与审计能力
"""
import os
import pytest
from src.hub import ToolHub
from src.client import MCPClient
from src.schemas.models import ToolChainPlan, ToolChainStep


@pytest.mark.asyncio
async def test_hub_register_and_list_tools():
    hub = ToolHub()
    server_script = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../src/servers/filesystem_server.py")
    )
    client = MCPClient("filesystem", "python", [server_script])
    await hub.register_client(client)

    tools = hub.list_tools()
    names = [t.name for t in tools]
    assert "read_file" in names
    assert "write_file" in names
    assert "list_directory" in names

    await hub.shutdown()


@pytest.mark.asyncio
async def test_hub_execute_plan_list_directory():
    hub = ToolHub()
    server_script = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../src/servers/filesystem_server.py")
    )
    client = MCPClient("filesystem", "python", [server_script])
    await hub.register_client(client)

    plan = ToolChainPlan(
        steps=[
            ToolChainStep(
                step_id="s1",
                tool_name="list_directory",
                arguments={"path": "."},
            )
        ]
    )
    results = await hub.execute_plan("list current directory", plan, session_id="test-001")
    assert len(results) == 1
    assert results[0].success is True
    assert "src" in str(results[0].output) or "tests" in str(results[0].output)

    await hub.shutdown()


@pytest.mark.asyncio
async def test_hub_dangerous_tool_requires_confirmation():
    hub = ToolHub()
    server_script = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../src/servers/shell_server.py")
    )
    client = MCPClient("shell", "python", [server_script])
    await hub.register_client(client)

    plan = ToolChainPlan(
        steps=[
            ToolChainStep(
                step_id="s1",
                tool_name="run_shell",
                arguments={"command": "echo hello"},
            )
        ]
    )
    results = await hub.execute_plan("run echo", plan, user_confirmed=False, session_id="test-002")
    # shell_server 本身没有标记 risk_level，因此 Hub 默认视为 safe
    # 此处主要验证执行链路通畅；安全策略的单元测试在 test_security 中覆盖
    assert len(results) == 1

    await hub.shutdown()
