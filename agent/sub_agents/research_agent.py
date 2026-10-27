"""Grounded GCP documentation research sub-agent powered by Gemini 2.5 Flash."""

import os
import time
from typing import Any, Dict, List, Optional

from agent.constitution import SYSTEM_INSTRUCTION
from agent.telemetry.logger import get_logger
from agent.telemetry.tracer import get_tracer
from agent.tools.gcp_search import search_gcp_documentation

try:
    from google.adk.agents import LlmAgent  # type: ignore

    ADK_AVAILABLE = True
except ImportError:  # pragma: no cover
    LlmAgent = None  # type: ignore
    ADK_AVAILABLE = False

logger = get_logger("sub_agents.research_agent")
tracer = get_tracer()


def build_adk_research_agent(model_name: Optional[str] = None) -> Any:
    """Construct the Google ADK `LlmAgent` instance for fast grounded GCP documentation retrieval."""
    resolved_model = model_name or os.getenv("FAST_MODEL_NAME", "gemini-2.5-flash")
    instruction = (
        f"{SYSTEM_INSTRUCTION}\n\n"
        "SPECIALIZATION: You are `ResearchAgent`. Always call `search_gcp_documentation` to gather "
        "verified Google Cloud service facts, regional availability, and SLA figures quickly and accurately."
    )
    if ADK_AVAILABLE and LlmAgent is not None:
        return LlmAgent(
            name="gcp_research_agent",
            model=resolved_model,
            description="Fast grounded GCP documentation retrieval specialist using Gemini 2.5 Flash.",
            instruction=instruction,
            tools=[search_gcp_documentation],
        )
    return {
        "name": "gcp_research_agent",
        "model": resolved_model,
        "tools": ["search_gcp_documentation"],
    }


class ResearchAgent:
    """Sub-agent responsible for fast grounded retrieval and factual synthesis on `gemini-2.5-flash`."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.name = "ResearchAgent"
        self.model_name = model_name or os.getenv("FAST_MODEL_NAME", "gemini-2.5-flash")
        self.adk_agent = build_adk_research_agent(self.model_name)

    def retrieve_grounding_context(
        self,
        customer_query: str,
        service_filter: Optional[str] = None,
        region_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute grounded GCP documentation search with self-correcting fallback if filters are too narrow."""
        start_time = time.perf_counter()
        with tracer.start_span(
            "sub_agent.research_agent",
            attributes={"model": self.model_name, "agent": self.name},
        ):
            logger.log_intent(
                target=self.name,
                reason=f"Execute fast grounded documentation lookup using {self.model_name}",
                parameters={
                    "query": customer_query,
                    "service_filter": service_filter,
                    "region_filter": region_filter,
                },
            )

            search_response = search_gcp_documentation(
                query=customer_query,
                service_filter=service_filter,
                region_filter=region_filter,
                max_results=3,
            )

            # Self-healing retry guided by structured tool recovery message
            if search_response.get("status") != "SUCCESS" and (service_filter or region_filter):
                search_response = search_gcp_documentation(
                    query=customer_query,
                    service_filter=None,
                    region_filter=None,
                    max_results=3,
                )

            results: List[Dict[str, Any]] = list(search_response.get("results", []))  # type: ignore[arg-type]
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            logger.log_outcome(
                target=self.name,
                status=str(search_response.get("status", "UNKNOWN")),
                latency_ms=latency_ms,
                summary=f"ResearchAgent ({self.model_name}) retrieved {len(results)} grounded documents.",
            )
            return {
                "agent": self.name,
                "model_used": self.model_name,
                "status": search_response.get("status"),
                "documents": results,
                "recovery_guidance": search_response.get("recovery_guidance"),
            }

    def format_fast_informational_response(
        self,
        customer_query: str,
        documents: List[Dict[str, Any]],
    ) -> str:
        """Synthesize a concise, grounded response for straightforward informational queries."""
        if not documents:
            return (
                "### Executive Summary\n"
                "I could not verify official Google Cloud documentation matching your exact query constraints.\n\n"
                "### Actionable Next Steps\n"
                "- Broaden the GCP service keywords or remove specific regional constraints."
            )

        primary = documents[0]
        bullet_facts = "\n".join(
            f"- **{doc['title']}**: {doc['summary']} (Regions: {', '.join(doc.get('supported_regions', [])[:4])})"
            for doc in documents
        )
        citations = "\n".join(f"- {doc['url']}" for doc in documents)

        return (
            f"### Executive Summary\n"
            f"Based on official Google Cloud documentation regarding **{primary['title']}**, "
            f"here is the verified technical guidance for your inquiry.\n\n"
            f"### Grounded GCP Findings\n"
            f"{bullet_facts}\n\n"
            f"### Grounded Citations\n"
            f"{citations}\n\n"
            f"### Actionable Next Steps\n"
            f"- Validate regional quota availability in the Google Cloud Console before provisioning."
        )
