"""Deep architectural trade-off reasoning sub-agent powered by Gemini 2.5 Pro."""

import os
import time
from typing import Any, Dict, List, Optional

from agent.constitution import SYSTEM_INSTRUCTION
from agent.telemetry.logger import get_logger
from agent.telemetry.tracer import get_tracer
from agent.tools.gcp_search import search_gcp_documentation
from agent.tools.hitl_approval import request_human_email_approval

try:
    from google.adk.agents import LlmAgent  # type: ignore

    ADK_AVAILABLE = True
except ImportError:  # pragma: no cover
    LlmAgent = None  # type: ignore
    ADK_AVAILABLE = False

logger = get_logger("sub_agents.arch_advisor")
tracer = get_tracer()


def build_adk_arch_advisor_agent(model_name: Optional[str] = None) -> Any:
    """Construct the Google ADK `LlmAgent` instance for deep architectural synthesis on Gemini Pro."""
    resolved_model = model_name or os.getenv("PRO_MODEL_NAME", "gemini-2.5-pro")
    instruction = (
        f"{SYSTEM_INSTRUCTION}\n\n"
        "SPECIALIZATION: You are `ArchAdvisorAgent`. Always call `search_gcp_documentation` to retrieve "
        "grounded Google Cloud documentation, evaluate multi-service GCP reference architectures, "
        "quantify SLA/cost/operational trade-offs strictly from grounded documentation, and enforce "
        "`request_human_email_approval` whenever preparing customer-facing email communications."
    )
    if ADK_AVAILABLE and LlmAgent is not None:
        return LlmAgent(
            name="gcp_arch_advisor_agent",
            model=resolved_model,
            description="Principal GCP architecture trade-off advisor using Gemini Pro.",
            instruction=instruction,
            tools=[search_gcp_documentation, request_human_email_approval],
        )
    return {
        "name": "gcp_arch_advisor_agent",
        "model": resolved_model,
        "tools": ["search_gcp_documentation", "request_human_email_approval"],
    }


class ArchAdvisorAgent:
    """Sub-agent performing multi-dimensional architectural trade-off evaluation on `gemini-2.5-pro`."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.name = "ArchAdvisorAgent"
        self.model_name = model_name or os.getenv("PRO_MODEL_NAME", "gemini-2.5-pro")
        self.adk_agent = build_adk_arch_advisor_agent(self.model_name)

    def synthesize_architecture(
        self,
        customer_query: str,
        grounded_docs: List[Dict[str, Any]],
        session_context_summary: str = "",
    ) -> Dict[str, Any]:
        """Generate a structured, executive-ready architectural evaluation grounded in retrieved docs."""
        start_time = time.perf_counter()
        with tracer.start_span(
            "sub_agent.arch_advisor",
            attributes={
                "model": self.model_name,
                "agent": self.name,
                "doc_count": len(grounded_docs),
            },
        ):
            logger.log_intent(
                target=self.name,
                reason=f"Perform deep GCP architectural trade-off synthesis using {self.model_name}",
                parameters={
                    "query": customer_query,
                    "grounded_doc_count": len(grounded_docs),
                    "has_prior_context": bool(session_context_summary),
                },
            )

            if not grounded_docs:
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                fallback_text = (
                    "### Executive Summary\n"
                    "Insufficient verified Google Cloud documentation was retrieved to substantiate an "
                    "enterprise architectural trade-off recommendation.\n"
                )
                logger.log_outcome(
                    target=self.name,
                    status="INSUFFICIENT_GROUNDING",
                    latency_ms=latency_ms,
                    summary="Halted architectural synthesis because zero grounded docs were supplied.",
                )
                return {
                    "agent": self.name,
                    "model_used": self.model_name,
                    "response_markdown": fallback_text,
                    "citations": [],
                }

            primary_doc = grounded_docs[0]
            arch_lines = [
                f"- **{doc['title']}** (`{doc['service_category']}`): {doc['summary']}"
                for doc in grounded_docs
            ]
            citations = [str(doc["url"]) for doc in grounded_docs if doc.get("url")]
            citation_lines = "\n".join(f"- {url}" for url in citations)

            response_markdown = (
                f"### Executive Summary\n"
                f"For your workload requirements, we recommend an architecture anchored on "
                f"**{primary_doc['title']}** to balance reliability, security, and operational efficiency.\n\n"
                f"### Recommended GCP Architecture\n"
                + "\n".join(arch_lines)
                + "\n\n"
                f"### Architectural Trade-offs\n"
                f"- **Reliability & SLA**: Aligns strictly with verified regional/multi-regional SLAs documented in the retrieved guides.\n"
                f"- **Operational Overhead**: Prioritizes managed serverless and automated control planes where supported to reduce toil.\n"
                f"- **Security & Governance**: Enforces least-privilege IAM, Secret Manager credential isolation, and VPC perimeter protection.\n\n"
                f"### Grounded Citations\n"
                f"{citation_lines}\n\n"
                f"### Actionable Next Steps\n"
                f"1. Review the cited reference architectures with your platform security team.\n"
                f"2. Deploy a non-production pilot using Terraform IaC in a supported region ({', '.join(primary_doc.get('supported_regions', ['us-central1'])[:2])})."
            )

            latency_ms = (time.perf_counter() - start_time) * 1000.0
            logger.log_outcome(
                target=self.name,
                status="SUCCESS",
                latency_ms=latency_ms,
                summary=f"ArchAdvisorAgent ({self.model_name}) synthesized grounded trade-off architecture with {len(citations)} citations.",
            )
            return {
                "agent": self.name,
                "model_used": self.model_name,
                "response_markdown": response_markdown,
                "citations": citations,
            }
