"""
Filesystem MCP Server（FastMCP 标准实现）
提供文件读写、目录浏览能力
"""
import os
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("filesystem")


@mcp.tool()
async def read_file(path: str) -> str:
    """读取指定文本文件的全部内容"""
    if not os.path.exists(path):
        return f"Error: File not found: {path}"
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"


@mcp.tool()
async def write_file(path: str, content: str) -> str:
    """向指定路径写入文本内容（风险：会覆盖已有文件）"""
    try:
        # 确保父目录存在
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully written to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


@mcp.tool()
async def list_directory(path: str) -> str:
    """列出指定目录下的所有文件和子目录"""
    if not os.path.isdir(path):
        return f"Error: Not a directory: {path}"
    try:
        entries = os.listdir(path)
        return "\n".join(entries) if entries else "(empty directory)"
    except Exception as e:
        return f"Error listing directory: {e}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
