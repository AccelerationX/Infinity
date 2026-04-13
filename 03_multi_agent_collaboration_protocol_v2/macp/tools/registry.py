"""
工具注册表
管理Agent可用的工具
"""
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass


@dataclass
class Tool:
    """工具定义"""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema
    handler: Callable
    
    def execute(self, arguments: Dict, context: Dict) -> Any:
        """执行工具"""
        return self.handler(arguments, context)


class ToolRegistry:
    """工具注册中心"""
    
    _instance = None
    _tools: Dict[str, Tool] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def register(self, tool: Tool):
        """注册工具"""
        self._tools[tool.name] = tool
    
    def get(self, name: str) -> Optional[Tool]:
        """获取工具"""
        return self._tools.get(name)
    
    def list_tools(self) -> List[Tool]:
        """列出所有工具"""
        return list(self._tools.values())
    
    def get_openai_schema(self) -> List[Dict]:
        """获取OpenAI格式的工具定义"""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
            }
            for tool in self._tools.values()
        ]


# 默认工具

def file_read_handler(args: Dict, context: Dict) -> Dict:
    """读取文件"""
    import os
    path = args.get("path", "")
    workspace = context.get("workspace_path", ".")
    full_path = os.path.join(workspace, path)
    
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
        return {"success": True, "content": content, "path": path}
    except Exception as e:
        return {"success": False, "error": str(e), "path": path}


def file_write_handler(args: Dict, context: Dict) -> Dict:
    """写入文件"""
    import os
    path = args.get("path", "")
    content = args.get("content", "")
    workspace = context.get("workspace_path", ".")
    full_path = os.path.join(workspace, path)
    
    try:
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        return {"success": True, "path": path, "bytes_written": len(content)}
    except Exception as e:
        return {"success": False, "error": str(e), "path": path}


def web_search_handler(args: Dict, context: Dict) -> Dict:
    """网页搜索（模拟）"""
    query = args.get("query", "")
    # 实际实现需要调用搜索引擎API
    return {
        "success": True,
        "query": query,
        "results": [
            {"title": "Result 1", "snippet": f"Search result for: {query}"},
            {"title": "Result 2", "snippet": "Another relevant result"},
        ]
    }


def code_analysis_handler(args: Dict, context: Dict) -> Dict:
    """代码分析"""
    code = args.get("code", "")
    # 简单的代码分析
    lines = code.split("\n")
    return {
        "success": True,
        "line_count": len(lines),
        "analysis": f"Code has {len(lines)} lines",
        "suggestions": ["Consider adding comments", "Check for unused imports"]
    }


# 初始化默认工具

def get_default_tools() -> List[Tool]:
    """获取默认工具列表"""
    return [
        Tool(
            name="file_read",
            description="Read content from a file",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to workspace"
                    }
                },
                "required": ["path"]
            },
            handler=file_read_handler
        ),
        Tool(
            name="file_write",
            description="Write content to a file",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to workspace"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write"
                    }
                },
                "required": ["path", "content"]
            },
            handler=file_write_handler
        ),
        Tool(
            name="web_search",
            description="Search the web for information",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"]
            },
            handler=web_search_handler
        ),
        Tool(
            name="code_analysis",
            description="Analyze code and provide suggestions",
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Code to analyze"
                    },
                    "language": {
                        "type": "string",
                        "description": "Programming language"
                    }
                },
                "required": ["code"]
            },
            handler=code_analysis_handler
        ),
    ]
