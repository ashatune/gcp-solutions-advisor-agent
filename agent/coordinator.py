"""Multi-agent Coordinator/Orchestrator with strategic Gemini 2.5 Flash/Pro model routing."""

from dataclasses import dataclass, field
import os
import time
from typing import Any, Callable, Dict, List, Optional

from agent.constitution import AGENT_CONSTITUTION, SYSTEM_INSTRUCTION
from agent.guardrails.policy import GroundingPolicyEvaluator, GuardrailEvaluationResult
from agent.memory.session_store import PersistentSessionStore
from agent.sub_agents.arch_advisor import ArchAdvisorAgent, build_adk_arch_advisor_agent
from agent.sub_agents.research_agent import ResearchAgent, build_adk_research_agent
from agent.telemetry.logger import get_logger
from agent.telemetry.pii_redactor import redact_pii
from agent.telemetry.tracer import get_tracer
from agent.tools.gcp_search import retrieve_gcp_secret, search_gcp_documentation
from agent.tools.hitl_approval import EmailApprovalInput, request_human_email_approval

try:
    from google.adk.agents import LlmAgent  # type: ignore

    ADK_AVAILABLE = True
except ImportError:  # pragma: no cover
    LlmAgent = None  # type: ignore
    ADK_AVAILABLE = False

logger = get_logger("coordinator")
tracer = get_tracer()


@dataclass
class AdvisorResponse:
    """Structured response envelope returned by `GCPSolutionsCoordinator`."""

    session_id: str
    routed_model: str
    delegated_agents: List[str]
    response_text: str
    citations: List[str] = field(default_factory=list)
    guardrail_result: Optional[GuardrailEvaluationResult] = None
    hitl_approval: Optional[Dict[str, object]] = None


def create_advisor_coordinator() -> Any:
    """Create the root Google ADK `LlmAgent` hierarchy delegating to ResearchAgent and ArchAdvisorAgent."""
    coordinator_model = os.getenv("FAST_MODEL_NAME", "gemini-2.5-flash")
    research_sub_agent = build_adk_research_agent()
    arch_sub_agent = build_adk_arch_advisor_agent()

    if ADK_AVAILABLE and LlmAgent is not None:
        return LlmAgent(
            name="gcp_solutions_advisor_coordinator",
            model=coordinator_model,
            description=(
                "Root GCP Solutions Advisor Coordinator routing queries between fast grounding research "
                "and deep architectural trade-off analysis."
            ),
            instruction=SYSTEM_INSTRUCTION,
            sub_agents=[research_sub_agent, arch_sub_agent],
            tools=[search_gcp_documentation, request_human_email_approval],
        )
    return {
        "name": "gcp_solutions_advisor_coordinator",
        "model": coordinator_model,
        "sub_agents": [research_sub_agent, arch_sub_agent],
    }


try:
    from vertexai.preview.reasoning_engines import AdkApp as _BaseAdkApp  # type: ignore

    class GCPSolutionsAdkApp(_BaseAdkApp):
        """Vertex AI Agent Engine AdkApp wrapper that instantiates `LlmAgent` natively in `set_up()`."""

        def clone(self) -> "GCPSolutionsAdkApp":
            return GCPSolutionsAdkApp(
                agent=create_advisor_coordinator(),
                enable_tracing=self._tmpl_attrs.get("enable_tracing"),
                env_vars=dict(self._tmpl_attrs.get("env_vars") or {}),
            )

        def __getstate__(self) -> Dict[str, Any]:
            state = dict(self.__dict__)
            tmpl = dict(state.get("_tmpl_attrs", {}))
            # Avoid pickling live Pydantic LlmAgent instances across environments; rebuild in set_up()
            tmpl["agent"] = None
            tmpl.pop("runner", None)
            tmpl.pop("in_memory_runner", None)
            state["_tmpl_attrs"] = tmpl
            return state

        def __setstate__(self, state: Dict[str, Any]) -> None:
            self.__dict__.update(state)

        def set_up(self) -> None:
            os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
            os.environ["FAST_MODEL_NAME"] = "gemini-3.5-flash"
            os.environ["PRO_MODEL_NAME"] = "gemini-3.1-pro-preview"
            self._tmpl_attrs["agent"] = create_advisor_coordinator()
            super().set_up()
            os.environ["GOOGLE_CLOUD_LOCATION"] = "global"

except ImportError:  # pragma: no cover
    GCPSolutionsAdkApp = None  # type: ignore


class GCPSolutionsCoordinator:
    """Orchestrates multi-agent delegation, strategic model routing, memory compaction, guardrails, and HITL."""

    _COMPLEX_ARCHITECTURE_SIGNALS = (
        "trade-off",
        "tradeoff",
        "compare",
        "vs",
        "versus",
        "architecture",
        "design",
        "migration",
        "multi-region",
        "global",
        "zero-trust",
        "pci",
        "hipaa",
        "streaming",
        "pipeline",
        "sla",
        "high availability",
    )

    def __init__(
        self,
        session_store: Optional[PersistentSessionStore] = None,
        guardrail_evaluator: Optional[GroundingPolicyEvaluator] = None,
    ) -> None:
        self.constitution = AGENT_CONSTITUTION
        self.fast_model = os.getenv(
            "FAST_MODEL_NAME", self.constitution.model_routing_policy["fast_research"]
        )
        self.pro_model = os.getenv(
            "PRO_MODEL_NAME", self.constitution.model_routing_policy["deep_architecture"]
        )
        self.api_key_ref = retrieve_gcp_secret(
            secret_id=os.getenv("GEMINI_API_KEY_SECRET_ID", "gemini-api-key"),
            env_fallback_key="GEMINI_API_KEY",
        )
        self.session_store = session_store or PersistentSessionStore()
        self.guardrail_evaluator = guardrail_evaluator or GroundingPolicyEvaluator()
        self.research_agent = ResearchAgent(model_name=self.fast_model)
        self.arch_advisor = ArchAdvisorAgent(model_name=self.pro_model)
        self.adk_root_agent = create_advisor_coordinator()

    def select_target_model(self, query: str) -> str:
        """Strategically route queries between `gemini-2.5-flash` (fast lookup) and `gemini-2.5-pro` (deep design).

        Args:
            query: Raw customer question or architectural prompt.

        Returns:
            Selected Gemini model identifier (`gemini-2.5-flash` or `gemini-2.5-pro`).
        """
        lower_query = query.lower()
        match_count = sum(1 for kw in self._COMPLEX_ARCHITECTURE_SIGNALS if kw in lower_query)
        if match_count >= 1 or len(query.split()) > 22:
            return self.pro_model
        return self.fast_model

    async def handle_inquiry(
        self,
        session_id: str,
        customer_query: str,
        customer_email: Optional[str] = None,
        operator_email_decision: Optional[str] = None,
        confirmation_callback: Optional[Callable[[EmailApprovalInput], str]] = None,
    ) -> AdvisorResponse:
        """Process a customer inquiry end-to-end across memory, routing, sub-agents, guardrails, and HITL.

        Args:
            session_id: Persistent multi-turn session identifier.
            customer_query: Customer's technical question or architecture scenario.
            customer_email: Optional recipient email if customer requested an outbound summary email.
            operator_email_decision: Optional human operator decision ('APPROVE' or 'REJECT') for HITL gate.
            confirmation_callback: Optional callback to prompt a human operator interactively.

        Returns:
            AdvisorResponse containing routed model, delegated sub-agents, verified response, and HITL state.
        """
        start_time = time.perf_counter()
        with tracer.start_span(
            "coordinator.handle_inquiry",
            attributes={"session_id": session_id},
        ):
            sanitized_query = str(redact_pii(customer_query))
            target_model = self.select_target_model(sanitized_query)

            logger.log_intent(
                target="GCPSolutionsCoordinator",
                reason=f"Route customer inquiry to optimal model tier ({target_model})",
                parameters={"session_id": session_id, "selected_model": target_model},
            )

            self.session_store.append_turn(session_id=session_id, role="user", content=sanitized_query)
            session_state = self.session_store.get_or_create_session(session_id)

            # Step 1: Always ground via ResearchAgent (Gemini 2.5 Flash)
            research_bundle = self.research_agent.retrieve_grounding_context(sanitized_query)
            grounded_docs: List[Dict[str, Any]] = research_bundle.get("documents", [])
            delegated_agents = [self.research_agent.name]

            # Step 2: Strategic Routing -> Fast Informational vs Deep Architectural Trade-off
            if target_model == self.pro_model:
                delegated_agents.append(self.arch_advisor.name)
                arch_bundle = self.arch_advisor.synthesize_architecture(
                    customer_query=sanitized_query,
                    grounded_docs=grounded_docs,
                    session_context_summary=session_state.compacted_summary,
                )
                draft_response = str(arch_bundle["response_markdown"])
            else:
                draft_response = self.research_agent.format_fast_informational_response(
                    customer_query=sanitized_query,
                    documents=grounded_docs,
                )

            # Step 3: Guardrail & Anti-Hallucination Verification
            guardrail_eval = self.guardrail_evaluator.verify_response(
                candidate_response=draft_response,
                retrieved_docs=grounded_docs,
            )
            final_text = guardrail_eval.sanitized_response
            citations = guardrail_eval.verified_citations

            # Step 4: Optional Outbound Email Dispatch gated by mandatory Human-in-the-Loop Hook
            hitl_result: Optional[Dict[str, object]] = None
            if customer_email and guardrail_eval.passed:
                resolved_decision = operator_email_decision
                if resolved_decision is None and confirmation_callback is not None:
                    resolved_decision = confirmation_callback(
                        EmailApprovalInput(
                            recipient_email=customer_email,
                            subject="GCP Architecture Recommendation & Trade-off Summary",
                            body_markdown=final_text,
                            citations=citations,
                        )
                    )
                hitl_result = request_human_email_approval(
                    recipient_email=customer_email,
                    subject="GCP Architecture Recommendation & Trade-off Summary",
                    body_markdown=final_text,
                    citations=citations,
                    operator_decision=resolved_decision,
                )

            self.session_store.append_turn(session_id=session_id, role="assistant", content=final_text)

            latency_ms = (time.perf_counter() - start_time) * 1000.0
            logger.log_outcome(
                target="GCPSolutionsCoordinator",
                status="SUCCESS" if guardrail_eval.passed else "BLOCKED_BY_GUARDRAIL",
                latency_ms=latency_ms,
                summary=(
                    f"Completed inquiry on {target_model} via {delegated_agents} "
                    f"(grounding_score={guardrail_eval.grounding_score})."
                ),
            )

            return AdvisorResponse(
                session_id=session_id,
                routed_model=target_model,
                delegated_agents=delegated_agents,
                response_text=final_text,
                citations=citations,
                guardrail_result=guardrail_eval,
                hitl_approval=hitl_result,
            )
