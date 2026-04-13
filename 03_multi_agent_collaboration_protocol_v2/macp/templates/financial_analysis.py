"""
金融分析模板
适用于股票分析、投资研究等任务
"""
from typing import List, Dict, Any
from .base import DomainTemplate, AgentRole, WorkflowType


class FinancialAnalysisTemplate(DomainTemplate):
    """金融分析领域模板"""
    
    def get_name(self) -> str:
        return "金融分析"
    
    def define_roles(self) -> List[AgentRole]:
        return [
            AgentRole(
                name="数据收集员",
                description="负责收集和整理金融数据",
                skills=["数据收集", "财务报表分析", "数据清洗"],
                responsibilities=["收集股票数据", "整理财务报表", "获取市场数据"],
                output_format="json"
            ),
            AgentRole(
                name="基本面分析师",
                description="负责公司基本面分析",
                skills=["基本面分析", "财务分析", "行业研究", "估值模型"],
                responsibilities=["分析公司财务状况", "评估盈利能力", "行业对比分析"],
                output_format="markdown"
            ),
            AgentRole(
                name="技术面分析师",
                description="负责技术分析",
                skills=["技术分析", "图表分析", "趋势判断", "指标计算"],
                responsibilities=["绘制K线图", "计算技术指标", "识别趋势和形态"],
                output_format="markdown"
            ),
            AgentRole(
                name="风险评估师",
                description="负责风险评估",
                skills=["风险评估", "压力测试", "情景分析", "风险模型"],
                responsibilities=["评估投资风险", "进行压力测试", "识别潜在风险"],
                output_format="markdown"
            ),
            AgentRole(
                name="投资策略师",
                description="负责制定投资策略",
                skills=["投资策略", "资产配置", "组合优化", "宏观分析"],
                responsibilities=["综合分析结果", "制定投资建议", "设计投资组合"],
                output_format="markdown"
            ),
            AgentRole(
                name="报告撰写员",
                description="负责撰写分析报告",
                skills=["报告撰写", "数据可视化", "PPT制作", "结论提炼"],
                responsibilities=["整合分析结果", "制作图表", "撰写投资建议"],
                output_format="markdown"
            )
        ]
    
    def define_workflow(self) -> WorkflowType:
        # 数据收集 → 基本面/技术面/风险评估并行 → 投资策略 → 报告撰写
        return WorkflowType.DAG
    
    def decompose_task(self, user_input: str, context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """分解金融分析任务"""
        # 提取股票代码（简单示例）
        ticker = self._extract_ticker(user_input)
        
        tasks = [
            {
                "id": "collect_data",
                "name": "数据收集",
                "description": f"收集{ticker}的财务数据和市场数据",
                "type": "data_collection",
                "assign_to": "数据收集员",
                "required_skills": ["数据收集"],
                "dependencies": []
            },
            {
                "id": "fundamental",
                "name": "基本面分析",
                "description": f"分析{ticker}的公司基本面",
                "type": "fundamental_analysis",
                "assign_to": "基本面分析师",
                "required_skills": ["基本面分析"],
                "dependencies": ["collect_data"]
            },
            {
                "id": "technical",
                "name": "技术面分析",
                "description": f"分析{ticker}的技术指标和趋势",
                "type": "technical_analysis",
                "assign_to": "技术面分析师",
                "required_skills": ["技术分析"],
                "dependencies": ["collect_data"]
            },
            {
                "id": "risk",
                "name": "风险评估",
                "description": f"评估投资{ticker}的风险",
                "type": "risk_assessment",
                "assign_to": "风险评估师",
                "required_skills": ["风险评估"],
                "dependencies": ["collect_data"]
            },
            {
                "id": "strategy",
                "name": "投资策略",
                "description": f"基于分析结果制定{ticker}的投资策略",
                "type": "investment_strategy",
                "assign_to": "投资策略师",
                "required_skills": ["投资策略"],
                "dependencies": ["fundamental", "technical", "risk"]
            },
            {
                "id": "report",
                "name": "报告撰写",
                "description": "撰写完整的分析报告",
                "type": "report_writing",
                "assign_to": "报告撰写员",
                "required_skills": ["报告撰写"],
                "dependencies": ["strategy"]
            }
        ]
        
        return tasks
    
    def _extract_ticker(self, user_input: str) -> str:
        """从输入中提取股票代码（简单实现）"""
        # 实际应该用更智能的方法
        import re
        # 匹配大写字母（股票代码）
        match = re.search(r'\b([A-Z]{1,5})\b', user_input.upper())
        if match:
            return match.group(1)
        return "股票"
    
    def integrate_outputs(self, tasks: List[Dict[str, Any]], outputs: Dict[str, Any]) -> Dict[str, Any]:
        """整合金融分析产出为报告"""
        data = outputs.get("collect_data", {})
        fundamental = outputs.get("fundamental", {})
        technical = outputs.get("technical", {})
        risk = outputs.get("risk", {})
        strategy = outputs.get("strategy", {})
        report = outputs.get("report", {})
        
        return {
            "tasks_completed": len(tasks),
            "report_title": "投资分析报告",
            "sections": {
                "数据摘要": data.get("summary", ""),
                "基本面分析": fundamental.get("content", ""),
                "技术面分析": technical.get("content", ""),
                "风险评估": risk.get("content", ""),
                "投资建议": strategy.get("content", ""),
                "完整报告": report.get("content", "")
            },
            "summary": self._generate_summary(tasks, outputs),
            "recommendation": strategy.get("recommendation", "HOLD"),
            "risk_level": risk.get("risk_level", "MEDIUM")
        }
