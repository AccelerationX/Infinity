"""
Shell MCP Server（FastMCP 标准实现）
在受限环境中执行 shell 命令，带有命令黑名单和超时控制。
"""
import os
import subprocess
import shlex
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("shell")

# 危险命令黑名单（大小写不敏感）
_BLACKLIST = {
    "rm", "del", "format", "mkfs", "dd",
    "shutdown", "reboot", "poweroff",
    ">", ">>", "|", ";", "&&", "||",
}


def _is_safe_command(command: str) -> tuple[bool, str]:
    """
    检查命令是否在白名单结构内。
    本实现为简化版：仅允许单个可执行文件及其参数，
    禁止管道、重定向、连续命令等 shell 元字符。
    """
    stripped = command.strip()
    if not stripped:
        return False, "Empty command"

    # 检测明显的危险关键字
    lower_cmd = stripped.lower()
    for danger in _BLACKLIST:
        if lower_cmd.startswith(danger + " ") or lower_cmd == danger:
            return False, f"Command contains blacklisted keyword: {danger}"

    # 检测 shell 元字符（允许在字符串参数内，但此处做极简检查）
    dangerous_chars = set(";|&><`$")
    if any(c in stripped for c in dangerous_chars):
        return False, "Shell metacharacters detected; only simple commands allowed"

    return True, ""


@mcp.tool()
async def run_shell(command: str, cwd: str = ".") -> str:
    """
    执行简单的 shell 命令，标准输出/错误会被捕获并返回。
    命令最长运行 10 秒，超时会被强制终止。
    """
    safe, msg = _is_safe_command(command)
    if not safe:
        return f"Rejected by security policy: {msg}"

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            encoding="utf-8",
            errors="replace",
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        if result.returncode != 0:
            return f"Exit code {result.returncode}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        return stdout or "(command executed successfully with no output)"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 10 seconds"
    except Exception as e:
        return f"Error executing command: {e}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
