"""
MCP 标准客户端封装
支持 stdio transport，可扩展为 sse transport
"""
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClient:
    """
    封装官方 MCP ClientSession，提供连接、断开、
    列出工具和调用工具的高层接口。
    """

    def __init__(
        self,
        server_name: str,
        command: str,
        args: list[str],
        cwd: str | None = None,
    ):
        self.server_name = server_name
        self.params = StdioServerParameters(
            command=command, args=args, cwd=cwd
        )
        self.session: ClientSession | None = None
        self._read = None
        self._write = None
        self._stdio_cm = None
        self._tools: list[dict] = []

    async def connect(self):
        """建立与 MCP Server 的 stdio 连接"""
        self._stdio_cm = stdio_client(self.params)
        self._read, self._write = await self._stdio_cm.__aenter__()
        self.session = ClientSession(self._read, self._write)
        await self.session.__aenter__()
        await self.session.initialize()
        tools_result = await self.session.list_tools()
        self._tools = [
            {"server": self.server_name, **t.model_dump()} for t in tools_result.tools
        ]

    async def disconnect(self):
        """安全断开连接，忽略 Windows 上 stdio_client 关闭时的已知 race condition 异常"""
        try:
            if self.session:
                await self.session.__aexit__(None, None, None)
        except (Exception, asyncio.CancelledError):
            pass
        finally:
            self.session = None
        # stdio_client 在 Windows 上关闭时有已知的 race condition，
        # 直接丢弃 generator 比调用 __aexit__ 更稳定
        if self._stdio_cm:
            try:
                await self._stdio_cm.aclose()
            except (Exception, asyncio.CancelledError):
                pass
        self._stdio_cm = None
        self._read = None
        self._write = None

    @property
    def tools(self) -> list[dict]:
        """返回该 Server 提供的所有工具（已序列化为 dict）"""
        return self._tools

    async def call_tool(self, tool_name: str, arguments: dict):
        """调用指定工具"""
        if not self.session:
            raise RuntimeError("Client not connected. Call connect() first.")
        return await self.session.call_tool(tool_name, arguments=arguments)
