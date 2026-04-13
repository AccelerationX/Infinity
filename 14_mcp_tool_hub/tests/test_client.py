"""
测试 MCPClient 与官方 MCP Server 的 stdio 集成
"""
import os
import pytest
from src.client import MCPClient


@pytest.mark.asyncio
async def test_calculator_client_connect_and_call():
    """验证 Client 能连接 Calculator Server 并成功调用 calculate"""
    server_script = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../src/servers/calculator_server.py")
    )
    client = MCPClient("calculator", "python", [server_script])
    await client.connect()

    names = [t["name"] for t in client.tools]
    assert "calculate" in names

    result = await client.call_tool("calculate", {"expression": "2 + 3 * 4"})
    texts = [c.text for c in result.content if hasattr(c, "text")]
    output = "".join(texts)
    assert "14" in output
    assert result.isError is False

    await client.disconnect()


@pytest.mark.asyncio
async def test_filesystem_client_list_tools():
    """验证 Client 能正确列出 Filesystem Server 的工具"""
    server_script = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../src/servers/filesystem_server.py")
    )
    client = MCPClient("filesystem", "python", [server_script])
    await client.connect()

    names = [t["name"] for t in client.tools]
    assert "read_file" in names
    assert "write_file" in names
    assert "list_directory" in names

    await client.disconnect()
