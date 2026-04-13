"""
完整演示：多Agent协作 + Web监控

这个示例展示了：
1. 初始化MACP框架
2. 启动Web监控仪表盘
3. 提交协作任务
4. 实时监控Agent状态和进度

访问 http://localhost:8080 查看监控界面
"""
import sys
import time
import threading
sys.path.insert(0, "D:\\ResearchProjects\\02_multi_agent_collaboration_protocol_v2")

from macp import CollaborationFramework
from macp.templates.software_dev import SoftwareDevelopmentTemplate
from macp.web.dashboard import WebDashboard, create_dashboard_template


def demo_with_real_tasks(framework):
    """提交真实任务进行演示"""
    print("\n" + "="*60)
    print("正在提交示例任务...")
    print("="*60)
    
    # 提交几个示例任务
    tasks = [
        "创建一个Python博客网站，包含文章列表、详情页和评论功能",
        "设计一个REST API for 电商订单系统",
    ]
    
    jobs = []
    for i, task in enumerate(tasks, 1):
        print(f"\n[{i}/{len(tasks)}] 提交任务: {task[:50]}...")
        
        def on_progress(data):
            if data["progress"] % 25 == 0:
                print(f"  Progress: {data['progress']}%, Completed: {data['completed']}/{data['total']}")
        
        job = framework.execute(
            input_text=task,
            job_name=f"Demo Job {i}",
            on_progress=on_progress
        )
        jobs.append(job)
        print(f"  Job ID: {job.id}")
    
    print(f"\n已提交 {len(jobs)} 个任务")
    return jobs


def print_status(framework):
    """打印当前状态"""
    print("\n" + "="*60)
    print("当前状态")
    print("="*60)
    
    # Agent状态
    print("\n[Agent状态]")
    agents = framework.get_agent_status()
    for agent in agents:
        status_icon = "●"
        if agent["status"] == "idle":
            status_icon = "🟢"
        elif agent["status"] == "busy":
            status_icon = "🔵"
        elif agent["status"] == "error":
            status_icon = "🔴"
        
        print(f"  {status_icon} {agent['name']} ({agent['role']})")
        if agent.get("current_task"):
            print(f"     正在执行: {agent['current_task']}")
    
    # 整体进度
    print("\n[整体进度]")
    overall = framework.get_overall_progress()
    bar_width = 40
    filled = int(bar_width * overall["overall_progress"] / 100)
    bar = "█" * filled + "░" * (bar_width - filled)
    print(f"  [{bar}] {overall['overall_progress']}%")
    print(f"  活跃: {overall['active_jobs']} | 完成: {overall['completed_jobs']} | 失败: {overall['failed_jobs']} | 总计: {overall['total_jobs']}")
    
    # 告警
    alerts = framework.get_alerts()
    if alerts:
        print("\n[告警信息]")
        for alert in alerts[:5]:
            level_icon = "🔴" if alert["level"] == "error" else "⚠️"
            print(f"  {level_icon} [{alert['level'].upper()}] {alert['title']}")
            print(f"     {alert['message'][:80]}...")


def main():
    """主函数"""
    print("="*60)
    print("MACP - 多Agent协作协议演示")
    print("="*60)
    
    # 初始化框架
    print("\n1. 初始化MACP框架...")
    template = SoftwareDevelopmentTemplate()
    framework = CollaborationFramework(
        template=template,
        workspace_path="./demo_workspace",
        max_workers=3
    )
    print(f"   ✓ 已创建 {len(template.roles)} 个Agent角色")
    
    # 创建HTML模板
    print("\n2. 准备Web界面...")
    create_dashboard_template()
    print("   ✓ HTML模板已创建")
    
    # 启动Web服务器（在后台线程）
    print("\n3. 启动Web监控仪表盘...")
    dashboard = WebDashboard(framework, port=8080)
    
    web_thread = threading.Thread(
        target=dashboard.run,
        kwargs={"debug": False},
        daemon=True
    )
    web_thread.start()
    
    print("   ✓ Web服务器已启动")
    print("   📊 访问 http://localhost:8080 查看监控界面")
    
    # 等待Web服务器启动
    time.sleep(2)
    
    # 提交示例任务
    jobs = demo_with_real_tasks(framework)
    
    # 监控循环
    print("\n" + "="*60)
    print("开始监控（按Ctrl+C停止）")
    print("="*60)
    
    try:
        while True:
            print_status(framework)
            
            # 检查是否所有任务完成
            all_done = all(
                framework.get_job(j.id).status.value in ["completed", "failed", "cancelled"]
                for j in jobs
            )
            
            if all_done:
                print("\n" + "="*60)
                print("所有任务已完成！")
                print("="*60)
                break
            
            time.sleep(3)
            
    except KeyboardInterrupt:
        print("\n\n用户中断")
    
    # 最终状态
    print("\n[最终报告]")
    for job in jobs:
        status = framework.get_job_status(job.id)
        print(f"\n  Job: {status['name']}")
        print(f"  Status: {status['status']}")
        print(f"  Progress: {status['progress_percent']}%")
        print(f"  Duration: {status['duration']:.1f}s" if status['duration'] else "  Duration: N/A")
    
    print("\n" + "="*60)
    print("演示结束。Web服务器仍在运行，访问 http://localhost:8080")
    print("按Ctrl+C完全退出")
    print("="*60)
    
    # 保持运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n再见！")


if __name__ == "__main__":
    main()
