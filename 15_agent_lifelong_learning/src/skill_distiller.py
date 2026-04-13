"""
技能蒸馏模块：用 LLM 将聚类模式抽象为标准化 SkillTemplate
"""
import json
from .schemas.models import SkillTemplate


class SkillDistiller:
    """
    接收 PatternMiner 输出的模式，调用 LLM 生成结构化的技能定义。
    兼容 OpenAI API、Claude、本地 vLLM 等任意实现了 chat/completions 接口的模型。
    """

    def __init__(self, llm_client=None):
        """
        llm_client: 一个可调用对象，接收 prompt 字符串，返回生成的文本字符串。
                   如果为 None，则使用基于规则的后备逻辑（仅用于测试）。
        """
        self.llm = llm_client

    def distill(self, pattern: dict) -> SkillTemplate:
        """将模式蒸馏为 SkillTemplate"""
        if self.llm is None:
            return self._fallback_distill(pattern)
        return self._llm_distill(pattern)

    def _llm_distill(self, pattern: dict) -> SkillTemplate:
        prompt = f"""你是一个技能提取助手。请根据以下用户请求模式，抽象出一个可复用的技能模板。

代表性请求：{pattern['representative_request']}
示例请求：
{chr(10).join(['- ' + r for r in pattern['sample_requests'][:5]])}

示例动作序列：
{json.dumps(pattern['sample_actions'][:2], ensure_ascii=False, indent=2)}

请输出一个 JSON 对象，包含以下字段：
- name: 技能的英文名称（snake_case，不超过 3 个单词）
- description: 技能的中文描述
- params: 从请求中抽象出的参数名列表（字符串数组，如 ["city", "date"]）
- action_template: 一个字典，描述 Agent 应该如何执行这个技能，值中可用 {{param}} 占位符

只输出 JSON，不要任何解释。"""

        raw = self.llm(prompt).strip()
        # 提取 JSON 块
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()

        data = json.loads(raw)
        return SkillTemplate(
            name=data["name"],
            description=data.get("description", ""),
            params=data.get("params", []),
            trigger_patterns=pattern["sample_requests"][:5],
            action_template=data.get("action_template", {}),
        )

    def _fallback_distill(self, pattern: dict) -> SkillTemplate:
        """无 LLM 时的后备逻辑，用于快速验证和测试"""
        req = pattern["representative_request"]
        # 简单规则映射
        if "股价" in req or "K线" in req or "走势" in req:
            return SkillTemplate(
                name="query_stock",
                description="查询指定股票的行情信息",
                params=["stock_code", "metric"],
                trigger_patterns=pattern["sample_requests"][:5],
                action_template={"tool": "stock_query", "params": {"code": "{{stock_code}}", "metric": "{{metric}}"}},
            )
        if "天气" in req or "下雨" in req:
            return SkillTemplate(
                name="query_weather",
                description="查询指定城市的天气",
                params=["city", "date"],
                trigger_patterns=pattern["sample_requests"][:5],
                action_template={"tool": "weather_query", "params": {"city": "{{city}}", "date": "{{date}}"}},
            )
        if "计算" in req or "算一下" in req:
            return SkillTemplate(
                name="calculate_expression",
                description="计算数学表达式",
                params=["expression"],
                trigger_patterns=pattern["sample_requests"][:5],
                action_template={"tool": "calculator", "params": {"expression": "{{expression}}"}},
            )
        return SkillTemplate(
            name="general_skill",
            description="通用技能",
            params=[],
            trigger_patterns=pattern["sample_requests"][:5],
            action_template={"tool": "general", "params": {}},
        )
