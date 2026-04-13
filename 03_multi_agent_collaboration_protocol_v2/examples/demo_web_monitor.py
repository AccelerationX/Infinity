"""
演示：Web监控仪表盘
展示如何启动Web界面监控多Agent协作
"""
import sys
sys.path.insert(0, "D:\\ResearchProjects\\02_multi_agent_collaboration_protocol_v2")

from macp.core.framework import CollaborationFramework
from macp.templates.software_dev import SoftwareDevelopmentTemplate
from macp.web.dashboard import WebDashboard, create_dashboard_template
from macp.config import configure_models


def main():
    """启动Web监控演示"""
    
    # 配置模型
    print("配置模型API...")
    configure_models()
    
    # 创建模板
    print("初始化软件研发模板...")
    template = SoftwareDevelopmentTemplate()
    
    # 初始化框架
    print("启动MACP框架...")
    framework = CollaborationFramework(
        template=template,
        workspace_path="./demo_workspace",
        max_workers=3
    )
    
    # 创建HTML模板
    create_dashboard_template()
    
    # 启动Web仪表盘
    print("\n启动Web监控仪表盘...")
    dashboard = WebDashboard(framework, port=8080)
    
    # 运行
    dashboard.run(debug=True)


if __name__ == "__main__":
    main()
