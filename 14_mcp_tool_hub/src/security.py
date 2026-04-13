"""
安全模块：风险评估、权限校验与沙箱策略
"""
from .schemas.models import ToolMetadata, RiskLevel


class SecurityManager:
    """
    负责评估工具链的整体风险等级，并对单个工具的调用做权限校验。
    """

    def __init__(self):
        self.dangerous_whitelist: set[str] = set()
        self.dangerous_blacklist: set[str] = set()

    def assess_plan_risk(self, tools: list[ToolMetadata]) -> str:
        """
        根据计划涉及的所有工具，评估整体风险等级。
        规则：只要有一个 dangerous，整体就是 dangerous；
             没有 dangerous 但有 caution，整体是 caution；
             否则为 safe。
        """
        levels = {t.risk_level for t in tools}
        if RiskLevel.DANGEROUS in levels:
            return RiskLevel.DANGEROUS
        if RiskLevel.CAUTION in levels:
            return RiskLevel.CAUTION
        return RiskLevel.SAFE

    def check_tool_permission(
        self, tool: ToolMetadata, user_confirmed: bool
    ) -> tuple[bool, str]:
        """
        检查单个工具是否允许执行。
        返回 (是否允许, 消息)。
        """
        if tool.name in self.dangerous_blacklist:
            return False, f"Tool '{tool.name}' is permanently blacklisted."

        if tool.risk_level == RiskLevel.SAFE:
            return True, ""

        if tool.risk_level == RiskLevel.CAUTION:
            return True, f"Tool '{tool.name}' is cautious; execution logged."

        if tool.risk_level == RiskLevel.DANGEROUS:
            if tool.name in self.dangerous_whitelist:
                return True, f"Tool '{tool.name}' is whitelisted."
            if user_confirmed:
                return True, ""
            return (
                False,
                f"Tool '{tool.name}' (risk={tool.risk_level}) requires explicit user confirmation.",
            )

        return False, f"Unknown risk level for tool '{tool.name}'."
