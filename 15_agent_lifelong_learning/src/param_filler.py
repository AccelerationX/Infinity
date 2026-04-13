"""
技能参数填充模块：将用户请求中的具体值映射到技能的占位符参数
"""
import re
from .schemas.models import SkillTemplate


class ParamFiller:
    """
    使用基于规则的后备逻辑（以及预留的 LLM 接口），
    从用户自然语言请求中提取参数值，填充到 SkillTemplate 的 action_template 中。
    """

    def __init__(self, llm_client=None):
        self.llm = llm_client

    def fill(self, user_request: str, skill: SkillTemplate) -> dict:
        """
        返回填充后的 action dict（不修改原始 skill.action_template）。
        """
        if self.llm is not None:
            return self._llm_fill(user_request, skill)
        return self._fallback_fill(user_request, skill)

    def _llm_fill(self, user_request: str, skill: SkillTemplate) -> dict:
        import json
        prompt = f"""Given the user request and the skill template, extract the parameter values and return a JSON object.

Skill: {skill.name}
Parameters: {skill.params}
Action Template: {json.dumps(skill.action_template, ensure_ascii=False)}

User Request: {user_request}

Return ONLY a JSON object with the parameter values."""
        raw = self.llm(prompt).strip()
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
        try:
            params = json.loads(raw)
        except Exception:
            params = {}
        return self._inject_params(skill.action_template, params)

    def _fallback_fill(self, user_request: str, skill: SkillTemplate) -> dict:
        """基于规则提取常见参数类型"""
        params = {}
        for p in skill.params:
            val = self._extract_param(user_request, p)
            if val is not None:
                params[p] = val
        return self._inject_params(skill.action_template, params)

    def _extract_param(self, text: str, param_name: str) -> str | None:
        """针对常见参数名做启发式提取"""
        lower_text = text.lower()

        if param_name in ("stock_code", "stock", "code"):
            # 提取 6 位数字股票代码
            match = re.search(r"(\d{6})", text)
            return match.group(1) if match else None

        if param_name in ("city", ):
            # 策略 1: 介词结构 "在 XX 天气" 最可靠
            match = re.search(r"(?:在|去|到|从|查一下|看看|帮)(?:我|人|他|她|它)?(?:查|看)?(?:一下|看)?\s*([\u4e00-\u9fa5]{2,4})(?:明天|今天|昨天|未来|下周|的|天气|气温|空气质量|怎么样|如何)", text)
            if match:
                return match.group(1)
            # 策略 2: 直接 "XX 天气"，限制城市名长度 2-4 字，取最短的匹配
            matches = re.findall(r"([\u4e00-\u9fa5]{2,4})(?:明天|今天|昨天|未来|下周|的|天气|气温|空气质量|怎么样|如何)", text)
            if matches:
                # 返回长度最短的那个，通常就是城市名
                return min(matches, key=len)
            # 策略 3: 英文城市
            match = re.search(r"in\s+([A-Za-z\s]+?)\s+(?:weather|today|tomorrow)", text, re.I)
            if match:
                return match.group(1).strip()
            return None

        if param_name in ("expression", "math", "formula"):
            # 提取数学表达式
            match = re.search(r"(?:计算|等于|算一下|calculate)\s*[:=]?\s*([0-9\+\-\*\/\(\)\.\s]+)", text)
            if match:
                return match.group(1).strip()
            match = re.search(r"([0-9\+\-\*\/\(\)\.\s]{3,})", text)
            if match:
                return match.group(1).strip()
            return None

        if param_name in ("metric", "type", "indicator"):
            # 股票指标类型
            if any(k in lower_text for k in ("k线", "走势", "kline")):
                return "kline"
            if any(k in lower_text for k in ("价格", "股价", "price")):
                return "price"
            if any(k in lower_text for k in ("成交量", "volume")):
                return "volume"
            return "price"

        if param_name in ("date", "time"):
            if "明天" in text or "tomorrow" in lower_text:
                return "tomorrow"
            if "今天" in text or "today" in lower_text:
                return "today"
            if "昨天" in text or "yesterday" in lower_text:
                return "yesterday"
            return None

        if param_name in ("path", "file", "filename"):
            match = re.search(r"([A-Za-z]:\\[^'\"\s]+|/[\w/._-]+|\./?[\w/._-]+)", text)
            return match.group(1) if match else None

        if param_name in ("content", "text"):
            match = re.search(r"['\"'\"'](.+?)['\"'\"']", text)
            return match.group(1) if match else None

        if param_name in ("command", "cmd"):
            match = re.sub(r"^(执行命令|运行|run shell|run command|run)\s*[:=]?\s*", "", text, flags=re.I)
            return match.strip() if match else None

        if param_name in ("query", "keyword", "search"):
            match = re.sub(r"^(搜索|查找|查一下|search for|search|query)\s*", "", text, flags=re.I)
            return match.strip() if match else None

        return None

    def _inject_params(self, template: dict, params: dict) -> dict:
        """递归替换模板中的 {{param}} 占位符"""
        import copy
        result = copy.deepcopy(template)

        def _replace(obj):
            if isinstance(obj, str):
                for key, val in params.items():
                    placeholder = f"{{{{{key}}}}}"
                    if placeholder in obj:
                        obj = obj.replace(placeholder, str(val))
                return obj
            if isinstance(obj, dict):
                return {k: _replace(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_replace(v) for v in obj]
            return obj

        return _replace(result)
