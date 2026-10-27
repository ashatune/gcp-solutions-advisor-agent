"""Cloud Run HTTP entrypoint and interactive UI for the GCP Solutions Advisor Agent."""

import asyncio
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from typing import Any, Dict
from urllib.parse import urlparse

from agent.constitution import AGENT_CONSTITUTION
from agent.coordinator import GCPSolutionsCoordinator
from agent.telemetry.logger import get_logger
from eval.run_evals import run_golden_evaluation

logger = get_logger("cloud_run.server")
COORDINATOR = GCPSolutionsCoordinator()

HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>GCP Solutions Advisor Agent (ADK)</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 920px; margin: 2rem auto; padding: 0 1rem; background: #0f172a; color: #f8fafc; }
    h1 { color: #38bdf8; margin-bottom: 0.25rem; }
    .badge { display: inline-block; background: #1e293b; border: 1px solid #38bdf8; color: #38bdf8; padding: 0.2rem 0.6rem; border-radius: 999px; font-size: 0.8rem; margin-right: 0.5rem; }
    .card { background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 1.25rem; margin-top: 1.25rem; }
    textarea, input { width: 100%; box-sizing: border-box; background: #0f172a; color: #f8fafc; border: 1px solid #475569; border-radius: 6px; padding: 0.65rem; margin-top: 0.5rem; font-size: 0.95rem; }
    button { background: #0284c7; color: white; border: none; padding: 0.65rem 1.2rem; border-radius: 6px; font-weight: 600; cursor: pointer; margin-top: 0.75rem; margin-right: 0.5rem; }
    button:hover { background: #0369a1; }
    pre { background: #020617; padding: 1rem; border-radius: 6px; overflow-x: auto; white-space: pre-wrap; border: 1px solid #334155; }
  </style>
</head>
<body>
  <h1>GCP Solutions Advisor Agent</h1>
  <div>
    <span class="badge">Google ADK</span>
    <span class="badge">Gemini 2.5 Flash &amp; Pro Routing</span>
    <span class="badge">OpenTelemetry + PII Scrubbing</span>
    <span class="badge">HITL Gate + Guardrails</span>
  </div>

  <div class="card">
    <h3>Ask a GCP Architecture or Documentation Question</h3>
    <label>Customer Inquiry:</label>
    <textarea id="query" rows="3">Compare Cloud Spanner vs Cloud SQL for PostgreSQL architecture trade-offs for global high availability.</textarea>
    <label style="display:block; margin-top:0.5rem;">Optional Customer Email (Triggers HITL Approval Gate):</label>
    <input id="email" type="text" placeholder="architect@enterprise.com (optional)" />
    <button onclick="sendInquiry()">Run Advisor Agent</button>
    <button onclick="runEvals()" style="background:#059669;">Run 5-Scenario Golden Eval Suite</button>
  </div>

  <div class="card">
    <h3>Agent Telemetry &amp; Grounded Output</h3>
    <pre id="output">Ready. Click "Run Advisor Agent" or "Run 5-Scenario Golden Eval Suite".</pre>
  </div>

  <script>
    async function sendInquiry() {
      const out = document.getElementById('output');
      out.textContent = 'Routing inquiry through GCPSolutionsCoordinator...';
      const payload = {
        session_id: 'web-session-' + Date.now(),
        customer_query: document.getElementById('query').value,
        customer_email: document.getElementById('email').value || null
      };
      const res = await fetch('/advise', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      out.textContent = JSON.stringify(data, null, 2);
    }
    async function runEvals() {
      const out = document.getElementById('output');
      out.textContent = 'Running 5-Scenario Golden Evaluation Suite...';
      const res = await fetch('/eval');
      const data = await res.json();
      out.textContent = JSON.stringify(data, null, 2);
    }
  </script>
</body>
</html>
"""


class AdvisorHTTPHandler(BaseHTTPRequestHandler):
    """HTTP handler serving the Advisor Agent API and interactive UI on Cloud Run."""

    def _send_json(self, status_code: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            body = HTML_DASHBOARD.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/health":
            self._send_json(
                200,
                {
                    "status": "HEALTHY",
                    "service": "gcp-solutions-advisor-agent",
                    "project_id": os.getenv("GCP_PROJECT_ID", "your-gcp-project-id"),
                    "persona": AGENT_CONSTITUTION.persona_title,
                    "models": AGENT_CONSTITUTION.model_routing_policy,
                },
            )
            return

        if parsed.path == "/eval":
            report = asyncio.run(run_golden_evaluation())
            self._send_json(200, report)
            return

        self._send_json(404, {"error": f"Unknown path: {parsed.path}"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/advise":
            self._send_json(404, {"error": f"Unknown POST endpoint: {parsed.path}"})
            return

        content_len = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"
        try:
            data = json.loads(raw_body)
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON request body."})
            return

        customer_query = str(data.get("customer_query") or data.get("query") or "").strip()
        if not customer_query:
            self._send_json(400, {"error": "Field `customer_query` is required."})
            return

        session_id = str(data.get("session_id") or "cloud-run-session")
        customer_email = data.get("customer_email")
        operator_decision = data.get("operator_email_decision")

        result = asyncio.run(
            COORDINATOR.handle_inquiry(
                session_id=session_id,
                customer_query=customer_query,
                customer_email=customer_email,
                operator_email_decision=operator_decision,
            )
        )
        self._send_json(200, asdict(result))

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        logger.info(f"HTTP {self.address_string()} - {format % args}")


def run_server() -> None:
    """Start the Cloud Run HTTP server on `$PORT` (default 8080)."""
    port = int(os.getenv("PORT", "8080"))
    server = ThreadingHTTPServer(("0.0.0.0", port), AdvisorHTTPHandler)
    logger.info(f"Starting GCP Solutions Advisor Agent server on port {port}...")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
