"""
参数推断模块：根据用户意图和工具的 input_schema 自动填充参数
"""
import json
import re
from .schemas.models import ToolMetadata


class ArgumentResolver:
    """
    将自然语言意图映射到具体工具的参数值。
    支持 LLM 智能填充和基于规则的后备推断。
    """

    def __init__(self, llm_client=None):
        self.llm = llm_client

    def resolve(
        self,
        intent: str,
        tool: ToolMetadata,
    ) -> dict:
        """为指定工具推断最合适的参数"""
        if self.llm is not None:
            return self._llm_resolve(intent, tool)
        return self._fallback_resolve(intent, tool)

    def _llm_resolve(self, intent: str, tool: ToolMetadata) -> dict:
        schema_text = json.dumps(tool.input_schema, ensure_ascii=False, indent=2)
        prompt = f"""You are a parameter extraction assistant.

Given the user's intent and the tool's JSON Schema, output the EXACT arguments needed to call the tool.

Tool name: {tool.name}
Tool description: {tool.description}
JSON Schema for parameters:
{schema_text}

User intent: {intent}

Respond ONLY with a JSON object that matches the schema. No explanation.

JSON:"""
        raw = self.llm(prompt).strip()
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def _fallback_resolve(self, intent: str, tool: ToolMetadata) -> dict:
        """基于规则的参数提取，覆盖常见工具类型"""
        name = tool.name.split("/")[-1]

        # 1. 文件路径提取
        if name in ("read_file", "write_file", "list_directory"):
            # Windows/Linux 绝对路径 或 相对路径
            match = re.search(r"([A-Za-z]:\\[^'\"\s]+|/[\w/._-]+|\./?[\w/._-]+)", intent)
            path = match.group(1) if match else None
            if name == "write_file":
                # 尝试提取引号内的文本作为 content
                text_match = re.search(r"['\"'\"'](.+?)['\"'\"']", intent)
                content = text_match.group(1) if text_match else ""
                if path:
                    return {"path": path, "content": content}
                return {"content": content}
            if name == "list_directory":
                # list_directory 可以安全地默认当前目录
                return {"path": path or "."}
            if path:
                return {"path": path}
            return {}

        # 2. 数学表达式提取
        if name == "calculate":
            # 尝试提取 "计算 ..." 后面的表达式
            match = re.search(r"(?:计算|等于|result of|calculate)\s*[:=]?\s*([0-9\+\-\*\/\(\)\.\s]+)", intent)
            if match:
                return {"expression": match.group(1).strip()}
            # 否则提取任何看起来像数学表达式的部分
            match = re.search(r"([0-9\+\-\*\/\(\)\.\s]{3,})", intent)
            if match:
                return {"expression": match.group(1).strip()}
            return {"expression": "0"}

        # 3. URL 提取
        if name == "fetch_url":
            match = re.search(r"(https?://[^\s]+)", intent)
            url = match.group(1) if match else ""
            return {"url": url}

        # 4. 搜索关键词提取
        if name == "search_duckduckgo":
            # 去掉常见动词，取剩余部分作为 query
            query = re.sub(r"^(搜索|查找|查一下|search for|search)\s*", "", intent, flags=re.I).strip()
            return {"query": query}

        # 5. Shell 命令提取
        if name == "run_shell":
            # 去掉 "执行命令" 等前缀
            cmd = re.sub(r"^(执行命令|运行|run shell|run command|run)\s*[:=]?\s*", "", intent, flags=re.I).strip()
            return {"command": cmd}

        return {}
