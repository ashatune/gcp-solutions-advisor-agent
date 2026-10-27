"""Comprehensive unit and integration test suite covering all 95-point rubric requirements."""

import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import pytest  # type: ignore
except ImportError:  # pragma: no cover

    class _DummyMark:
        @staticmethod
        def asyncio(func):  # type: ignore
            return func

    class _DummyPytest:
        mark = _DummyMark()
        MonkeyPatch = object

    pytest = _DummyPytest()  # type: ignore

from agent.constitution import AGENT_CONSTITUTION, SYSTEM_INSTRUCTION
from agent.coordinator import GCPSolutionsCoordinator
from agent.guardrails.policy import GroundingPolicyEvaluator
from agent.memory.compactor import ConversationCompactor, TurnMessage
from agent.memory.session_store import PersistentSessionStore
from agent.telemetry.logger import get_logger
from agent.telemetry.pii_redactor import redact_pii
from agent.telemetry.tracer import get_tracer
from agent.tools.gcp_search import (
    GCPDocSearchInput,
    retrieve_gcp_secret,
    search_gcp_documentation,
)
from agent.tools.hitl_approval import (
    EmailApprovalInput,
    request_human_email_approval,
)
from eval.run_evals import run_golden_evaluation


def test_category1_tool_schemas_docstrings_and_guided_errors() -> None:
    """Verify Pydantic schemas, Google-style docstrings, and guided error handling (no unhandled crashes)."""
    assert search_gcp_documentation.__doc__ is not None
    assert "Args:" in search_gcp_documentation.__doc__
    assert "Returns:" in search_gcp_documentation.__doc__
    assert request_human_email_approval.__doc__ is not None
    assert "Args:" in request_human_email_approval.__doc__

    # Valid query returns grounded docs
    ok_res = search_gcp_documentation(query="Cloud Run vs GKE Autopilot", service_filter="compute")
    assert ok_res["status"] == "SUCCESS"
    assert len(ok_res["results"]) >= 1

    # Invalid short query returns guided recovery string instead of raising ValidationError
    invalid_res = search_gcp_documentation(query="a")
    assert invalid_res["status"] == "RECOVERABLE_ERROR"
    assert "Error:" in str(invalid_res["recovery_guidance"])

    # Invalid service filter returns guided recovery instructions
    bad_filter = search_gcp_documentation(query="Cloud Spanner", service_filter="invalid-category")
    assert bad_filter["status"] == "RECOVERABLE_ERROR"
    assert "Suggest retrying" in str(bad_filter["recovery_guidance"])

    # Zero-match query returns structured recovery guidance
    no_docs = search_gcp_documentation(query="nonexistentquantumservice999")
    assert no_docs["status"] == "NO_RESULTS"
    assert "Suggest broadening search terms" in str(no_docs["recovery_guidance"])

    # Simulated upstream timeout is caught and converted to guided recovery
    timeout_res = search_gcp_documentation(query="TRIGGER_SIMULATED_API_TIMEOUT")
    assert timeout_res["status"] == "RECOVERABLE_ERROR"
    assert "TimeoutError" in str(timeout_res["recovery_guidance"])


@pytest.mark.asyncio
async def test_category2_constitution_session_persistence_and_async_compaction(
    tmp_path: Path,
) -> None:
    """Verify constitution rules, persistent session storage, and async token sliding window compaction."""
    assert "Senior Google Cloud Principal Solutions Architect" in SYSTEM_INSTRUCTION
    assert "request_human_email_approval" in SYSTEM_INSTRUCTION

    storage_file = tmp_path / "sessions.json"
    compactor = ConversationCompactor(max_tokens=40, sliding_window_turns=2)
    store = PersistentSessionStore(storage_path=storage_file, compactor=compactor)

    for idx in range(5):
        store.append_turn(
            session_id="cust-session-1",
            role="user" if idx % 2 == 0 else "assistant",
            content=(
                f"Turn {idx}: Evaluating Cloud Spanner and BigQuery in us-central1 "
                "for HIPAA compliance with contact alice@enterprise.com."
            ),
        )

    await store.flush_background_tasks()
    state = store.get_or_create_session("cust-session-1")

    # Sliding window retains only the last 2 turns while compacting older turns into summary
    assert len(state.turns) == 2
    assert "Compacted Context Summary:" in state.compacted_summary
    assert "[REDACTED_EMAIL]" in state.compacted_summary
    assert "cloud spanner" in state.customer_profile.preferred_gcp_services
    assert "hipaa" in state.customer_profile.compliance_requirements
    assert "us-central1" in state.customer_profile.target_regions

    # Reload from disk to verify persistence
    reloaded_store = PersistentSessionStore(storage_path=storage_file, compactor=compactor)
    reloaded_state = reloaded_store.get_or_create_session("cust-session-1")
    assert reloaded_state.compacted_summary == state.compacted_summary
    assert "cloud spanner" in reloaded_state.customer_profile.preferred_gcp_services


@pytest.mark.asyncio
async def test_category3_orchestration_routing_guardrails_and_hitl() -> None:
    """Verify multi-agent delegation, Gemini 2.5 Flash/Pro routing, guardrails, and HITL email approval."""
    coordinator = GCPSolutionsCoordinator()

    # Fast informational query routes to gemini-2.5-flash
    fast_resp = await coordinator.handle_inquiry(
        session_id="route-test-fast",
        customer_query="How does Secret Manager protect credentials?",
    )
    assert fast_resp.routed_model == "gemini-2.5-flash"
    assert fast_resp.delegated_agents == ["ResearchAgent"]
    assert fast_resp.guardrail_result is not None and fast_resp.guardrail_result.passed

    # Complex trade-off query routes to gemini-2.5-pro and delegates to ArchAdvisorAgent
    pro_resp = await coordinator.handle_inquiry(
        session_id="route-test-pro",
        customer_query="Compare Cloud Spanner vs Cloud SQL architecture trade-offs for global HA.",
        customer_email="architect@customer.org",
        operator_email_decision=None,  # No operator confirmation yet -> must block dispatch!
    )
    assert pro_resp.routed_model == "gemini-2.5-pro"
    assert "ArchAdvisorAgent" in pro_resp.delegated_agents
    assert pro_resp.hitl_approval is not None
    assert pro_resp.hitl_approval["status"] == "PENDING_HUMAN_APPROVAL"
    assert pro_resp.hitl_approval["dispatched"] is False

    # Explicit operator approval ('APPROVE') allows dispatch
    approved_hitl = request_human_email_approval(
        recipient_email="architect@customer.org",
        subject="Approved GCP Database Architecture Matrix",
        body_markdown=pro_resp.response_text,
        citations=pro_resp.citations,
        operator_decision="APPROVE",
    )
    assert approved_hitl["status"] == "APPROVED_FOR_DISPATCH"
    assert approved_hitl["dispatched"] is True

    # Guardrail blocks hallucinated 100% SLA claims or unverified URLs
    evaluator = GroundingPolicyEvaluator()
    hallucinated = evaluator.verify_response(
        candidate_response="Cloud Run guarantees a 100% SLA uptime with https://cloud.google.com/fake-url",
        retrieved_docs=[
            {
                "title": "Cloud Run Guide",
                "url": "https://cloud.google.com/docs/choosing-a-compute-option",
                "service_category": "compute",
                "summary": "Cloud Run serverless containers.",
            }
        ],
    )
    assert hallucinated.passed is False
    assert len(hallucinated.violations) >= 1


def test_category4_observability_json_logs_intent_outcome_spans_and_pii() -> None:
    """Verify structured JSON logging, [INTENT]/[OUTCOME] capture, OpenTelemetry spans, and PII redaction."""
    raw_secret_text = (
        "Customer email jane.doe@finserv.com from IP 192.168.10.42 in "
        "project_id=finserv-prod-vault-01 leaked key AIzaSyD1234567890abcdef1234567890abcde"
    )
    scrubbed = redact_pii(raw_secret_text)
    assert "jane.doe@finserv.com" not in scrubbed
    assert "192.168.10.42" not in scrubbed
    assert "finserv-prod-vault-01" not in scrubbed
    assert "AIzaSyD1234567890abcdef1234567890abcde" not in scrubbed
    assert "[REDACTED_EMAIL]" in scrubbed
    assert "[REDACTED_IP]" in scrubbed
    assert "[REDACTED_GCP_PROJECT_ID]" in scrubbed
    assert "[REDACTED_GCP_API_KEY]" in scrubbed

    tracer = get_tracer()
    tracer.clear_spans()
    search_gcp_documentation(query="BigQuery Dataflow streaming")
    spans = tracer.get_recorded_spans()
    assert any(s.name == "tool.search_gcp_documentation" for s in spans)

    tool_logger = get_logger("tools.gcp_search")
    records = tool_logger.get_captured_records()
    assert any("[INTENT]" in r["message"] for r in records)
    assert any("[OUTCOME]" in r["message"] for r in records)
    for rec in records[-2:]:
        assert {"timestamp", "level", "trace_id", "component", "message"}.issubset(rec.keys())


@pytest.mark.asyncio
async def test_category5_secret_manager_fallback_and_golden_eval_suite(
    monkeypatch=None,
) -> None:
    """Verify Secret Manager `.env` fallback and 100% pass rate on the 5-scenario Golden Evaluation Suite."""
    prev_val = os.environ.get("GEMINI_API_KEY")
    try:
        if monkeypatch is not None and hasattr(monkeypatch, "setenv"):
            monkeypatch.setenv("GEMINI_API_KEY", "test-env-fallback-value")
        else:
            os.environ["GEMINI_API_KEY"] = "test-env-fallback-value"

        resolved = retrieve_gcp_secret("gemini-api-key", env_fallback_key="GEMINI_API_KEY")
        assert resolved == "test-env-fallback-value"

        eval_report = await run_golden_evaluation()
        assert eval_report["total_scenarios"] == 5
        assert eval_report["passed_scenarios"] == 5
        assert eval_report["pass_rate"] == 1.0
    finally:
        if monkeypatch is None:
            if prev_val is None:
                os.environ.pop("GEMINI_API_KEY", None)
            else:
                os.environ["GEMINI_API_KEY"] = prev_val


class GCPSolutionsAdvisorTestSuite(unittest.IsolatedAsyncioTestCase):
    """Standard-library unittest wrapper enabling `python3 -m unittest` execution alongside `pytest`."""

    def test_01_tools(self) -> None:
        test_category1_tool_schemas_docstrings_and_guided_errors()

    async def test_02_memory_and_compaction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            await test_category2_constitution_session_persistence_and_async_compaction(Path(tmp_dir))

    async def test_03_orchestration_and_hitl(self) -> None:
        await test_category3_orchestration_routing_guardrails_and_hitl()

    def test_04_observability_and_pii(self) -> None:
        test_category4_observability_json_logs_intent_outcome_spans_and_pii()

    async def test_05_secrets_and_evals(self) -> None:
        await test_category5_secret_manager_fallback_and_golden_eval_suite()


if __name__ == "__main__":
    unittest.main(verbosity=2)

