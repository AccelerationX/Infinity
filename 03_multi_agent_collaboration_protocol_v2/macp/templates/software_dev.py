"""
软件研发领域模板
"""
import json
from typing import Dict, Any, List, Optional

from .base import DomainTemplate, AgentRole, WorkflowStep


class SoftwareDevelopmentTemplate(DomainTemplate):
    """
    软件研发领域模板
    
    角色：PM、架构师、TechLead、前端、后端、QA、DevOps
    流程：需求分析 → 架构设计 → 任务规划 → 并行开发 → 集成测试 → 部署
    """
    
    def get_roles(self) -> List[AgentRole]:
        """获取软件研发角色定义"""
        return [
            AgentRole(
                name="ProductManager",
                description="Product Manager - Analyze requirements and create specifications",
                skills=["requirement_analysis", "user_story", "specification"],
                output_format="markdown",
                system_prompt="""You are a Product Manager. Your job is to:
1. Analyze user requirements thoroughly
2. Write clear, actionable specifications
3. Define user stories and acceptance criteria
4. Prioritize features based on value and effort

Output format: Markdown with sections for Requirements, User Stories, and Acceptance Criteria."""
            ),
            AgentRole(
                name="Architect",
                description="System Architect - Design system architecture and APIs",
                skills=["system_design", "api_design", "database_design", "tech_selection"],
                output_format="markdown",
                system_prompt="""You are a System Architect. Your job is to:
1. Design scalable system architecture
2. Define API contracts and data models
3. Select appropriate technologies
4. Ensure security and performance considerations

Output format: Markdown with Architecture Diagram description, API Spec, and Database Schema."""
            ),
            AgentRole(
                name="TechLead",
                description="Tech Lead - Technical planning and task breakdown",
                skills=["task_breakdown", "estimation", "technical_planning", "code_review"],
                output_format="json",
                system_prompt="""You are a Tech Lead. Your job is to:
1. Break down features into technical tasks
2. Estimate effort and identify risks
3. Define coding standards and best practices
4. Review and integrate code from team members

Output format: JSON with tasks, estimates, and dependencies."""
            ),
            AgentRole(
                name="FrontendDev",
                description="Frontend Developer - UI implementation",
                skills=["frontend", "react", "vue", "ui_design", "state_management"],
                output_format="code",
                system_prompt="""You are a Frontend Developer. Your job is to:
1. Implement responsive and accessible UI components
2. Manage application state effectively
3. Integrate with backend APIs
4. Write clean, maintainable code

Output format: Code files with comments explaining implementation decisions."""
            ),
            AgentRole(
                name="BackendDev",
                description="Backend Developer - API and service implementation",
                skills=["backend", "api_implementation", "database", "microservices"],
                output_format="code",
                system_prompt="""You are a Backend Developer. Your job is to:
1. Implement RESTful/GraphQL APIs
2. Design and optimize database queries
3. Handle authentication and authorization
4. Ensure API reliability and performance

Output format: Code files with API documentation."""
            ),
            AgentRole(
                name="QAEngineer",
                description="QA Engineer - Testing and quality assurance",
                skills=["test_design", "automation", "performance_testing", "bug_analysis"],
                output_format="code",
                system_prompt="""You are a QA Engineer. Your job is to:
1. Design comprehensive test cases
2. Write automated tests (unit, integration, e2e)
3. Perform load and performance testing
4. Report bugs with clear reproduction steps

Output format: Test code and test reports."""
            ),
            AgentRole(
                name="DevOps",
                description="DevOps Engineer - Deployment and infrastructure",
                skills=["ci_cd", "docker", "kubernetes", "cloud", "monitoring"],
                output_format="yaml",
                system_prompt="""You are a DevOps Engineer. Your job is to:
1. Set up CI/CD pipelines
2. Create Docker and Kubernetes configurations
3. Manage cloud infrastructure
4. Configure monitoring and alerting

Output format: YAML/JSON configuration files."""
            ),
        ]
    
    def decompose_task(self, 
                       input_text: str,
                       context: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        分解软件研发任务
        
        基于简单启发式规则分解，实际生产环境可以使用LLM进行智能分解
        """
        tasks = []
        
        # 任务1: 需求分析
        tasks.append({
            "id": "analysis",
            "name": "Requirement Analysis",
            "description": f"Analyze requirements for: {input_text}",
            "required_role": "ProductManager",
            "dependencies": [],
        })
        
        # 任务2: 架构设计
        tasks.append({
            "id": "architecture",
            "name": "Architecture Design",
            "description": "Design system architecture and APIs based on requirements",
            "required_role": "Architect",
            "dependencies": ["analysis"],
        })
        
        # 任务3: 技术规划
        tasks.append({
            "id": "planning",
            "name": "Technical Planning",
            "description": "Break down into technical tasks and estimate effort",
            "required_role": "TechLead",
            "dependencies": ["architecture"],
        })
        
        # 任务4-6: 并行开发（前端、后端、QA）
        tasks.append({
            "id": "frontend",
            "name": "Frontend Development",
            "description": "Implement user interface components",
            "required_role": "FrontendDev",
            "dependencies": ["planning"],
        })
        
        tasks.append({
            "id": "backend",
            "name": "Backend Development",
            "description": "Implement APIs and business logic",
            "required_role": "BackendDev",
            "dependencies": ["planning"],
        })
        
        tasks.append({
            "id": "testing",
            "name": "Test Development",
            "description": "Write test cases and automation",
            "required_role": "QAEngineer",
            "dependencies": ["planning"],
        })
        
        # 任务7: DevOps配置
        tasks.append({
            "id": "devops",
            "name": "DevOps Setup",
            "description": "Configure CI/CD and deployment",
            "required_role": "DevOps",
            "dependencies": ["architecture"],
        })
        
        # 任务8: 集成验证
        tasks.append({
            "id": "integration",
            "name": "Integration Verification",
            "description": "Verify integration of all components",
            "required_role": "TechLead",
            "dependencies": ["frontend", "backend", "testing"],
        })
        
        return tasks
    
    def aggregate_outputs(self,
                          outputs: Dict[str, Any],
                          context: Optional[Dict] = None) -> Dict[str, Any]:
        """聚合软件研发输出"""
        
        # 收集代码文件
        code_files = []
        docs = []
        tests = []
        configs = []
        
        for task_id, output in outputs.items():
            if isinstance(output, dict):
                if "code" in str(output).lower():
                    code_files.append({"task": task_id, "content": output})
                elif "test" in str(output).lower():
                    tests.append({"task": task_id, "content": output})
                elif "config" in str(output).lower() or "yaml" in str(output).lower():
                    configs.append({"task": task_id, "content": output})
                else:
                    docs.append({"task": task_id, "content": output})
        
        return {
            "project_type": "software_development",
            "code_files": code_files,
            "documentation": docs,
            "tests": tests,
            "configs": configs,
            "summary": f"Generated {len(code_files)} code files, {len(docs)} docs, {len(tests)} tests, {len(configs)} configs",
        }
