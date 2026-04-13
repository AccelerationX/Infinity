"""
Web Monitor Dashboard
Real-time display of multi-agent collaboration status
"""
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

try:
    from flask import Flask, render_template, jsonify, request
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Warning: Flask not installed. Web dashboard will not be available.")
    print("Install with: pip install flask")

from ..core.framework import CollaborationFramework


class WebDashboard:
    """
    Web Monitor Dashboard
    
    Features:
    1. Real-time display of all agent statuses
    2. Task progress bar
    3. Error alerts and human intervention panel
    4. Job detail view
    """
    
    def __init__(self, framework: CollaborationFramework, port: int = 8080):
        self.framework = framework
        self.port = port
        self.app = None
        
        if FLASK_AVAILABLE:
            import os
            template_dir = os.path.join(os.path.dirname(__file__), "templates")
            self.app = Flask(__name__, template_folder=template_dir)
            self._register_routes()
    
    def _register_routes(self):
        """Register routes"""
        
        @self.app.route("/")
        def index():
            """Main page"""
            return render_template("dashboard.html")
        
        @self.app.route("/api/overview")
        def api_overview():
            """Overall overview data"""
            return jsonify({
                "overall": self.framework.get_overall_progress(),
                "agents": self.framework.get_agent_status(),
                "alerts": self.framework.get_alerts()[:10]  # Recent 10 alerts
            })
        
        @self.app.route("/api/jobs")
        def api_jobs():
            """All jobs list"""
            jobs = self.framework.list_jobs()
            return jsonify([
                {
                    "id": j.id,
                    "name": j.name,
                    "status": j.status.value,
                    "progress": j.progress_percent,
                    "created_at": j.created_at.isoformat() if j.created_at else None
                }
                for j in jobs[:20]  # Recent 20
            ])
        
        @self.app.route("/api/jobs/<job_id>")
        def api_job_detail(job_id: str):
            """Job detail"""
            status = self.framework.get_job_status(job_id)
            if not status:
                return jsonify({"error": "Job not found"}), 404
            return jsonify(status)
        
        @self.app.route("/api/agents")
        def api_agents():
            """All agent statuses"""
            return jsonify(self.framework.get_agent_status())
        
        @self.app.route("/api/alerts")
        def api_alerts():
            """Alert messages"""
            level = request.args.get("level")
            return jsonify(self.framework.get_alerts(level))
        
        @self.app.route("/api/alerts/clear", methods=["POST"])
        def api_clear_alerts():
            """Clear alerts"""
            self.framework.clear_alerts()
            return jsonify({"success": True})
        
        @self.app.route("/api/jobs/<job_id>/cancel", methods=["POST"])
        def api_cancel_job(job_id: str):
            """Cancel job"""
            success = self.framework.cancel_job(job_id)
            return jsonify({"success": success})
        
        @self.app.route("/api/jobs/<job_id>/pause", methods=["POST"])
        def api_pause_job(job_id: str):
            """Pause job"""
            success = self.framework.pause_job(job_id)
            return jsonify({"success": success})
        
        @self.app.route("/api/jobs/<job_id>/resume", methods=["POST"])
        def api_resume_job(job_id: str):
            """Resume job"""
            success = self.framework.resume_job(job_id)
            return jsonify({"success": success})
    
    def run(self, debug: bool = False):
        """Start web server"""
        if not self.app:
            print("Web dashboard not available. Please install Flask.")
            return
        
        print(f"\n" + "="*60)
        print(f"MACP Web Dashboard")
        print(f"="*60)
        print(f"URL: http://localhost:{self.port}")
        print(f"="*60 + "\n")
        
        self.app.run(host="0.0.0.0", port=self.port, debug=debug)


def create_dashboard_template():
    """Ensure HTML template file exists"""
    import os
    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    template_path = os.path.join(template_dir, "dashboard.html")
    
    if os.path.exists(template_path):
        return template_path
    
    # Template should already exist, but create if missing
    os.makedirs(template_dir, exist_ok=True)
    
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>MACP Monitor</title>
    <style>
        body { font-family: sans-serif; background: #f5f7fa; margin: 0; padding: 20px; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .card { background: white; border-radius: 8px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .progress-bar { width: 100%; height: 20px; background: #e8e8e8; border-radius: 10px; overflow: hidden; }
        .progress-fill { height: 100%; background: #667eea; transition: width 0.3s; }
    </style>
</head>
<body>
    <div class="header"><h1>MACP Monitor</h1></div>
    <div class="container">
        <div class="card"><h2>Status</h2><p>Dashboard loading...</p></div>
    </div>
</body>
</html>'''
    
    with open(template_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    return template_path
