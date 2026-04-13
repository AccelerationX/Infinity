"""
Browser MCP Server（FastMCP 标准实现）
提供网页抓取和简单搜索能力
"""
import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("browser")


@mcp.tool()
async def fetch_url(url: str) -> str:
    """获取指定 URL 的原始 HTML 内容（截断至 8000 字符防止过长）"""
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.get(url)
            text = resp.text
            return text[:8000] + ("\n...[truncated]" if len(text) > 8000 else "")
    except Exception as e:
        return f"Error fetching {url}: {e}"


@mcp.tool()
async def search_duckduckgo(query: str) -> str:
    """
    使用 DuckDuckGo HTML 版进行搜索，返回前 5 条结果标题。
    注意：这是简化实现，生产环境建议使用专用搜索 API。
    """
    try:
        url = "https://html.duckduckgo.com/html/"
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, data={"q": query})
            html = resp.text
            titles = []
            # 极简解析：按 result__a 类分割
            for chunk in html.split('<a class="result__a"')[1:6]:
                if ">" in chunk and "<" in chunk.split(">", 1)[1]:
                    title = chunk.split(">", 1)[1].split("<")[0]
                    titles.append(title)
            return "\n".join(titles) if titles else "No results found"
    except Exception as e:
        return f"Error during search: {e}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
