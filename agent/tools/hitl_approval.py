"""Human-in-the-Loop (HITL) approval hook for outbound customer email dispatch."""

import time
from typing import Callable, Dict, List, Optional
import uuid

from pydantic import BaseModel, Field, ValidationError

from agent.telemetry.logger import get_logger
from agent.telemetry.pii_redactor import redact_pii
from agent.telemetry.tracer import get_tracer

logger = get_logger("tools.hitl_approval")
tracer = get_tracer()


class EmailApprovalInput(BaseModel):
    """Strict input schema for requesting human operator approval before sending a customer email."""

    recipient_email: str = Field(
        ...,
        min_length=5,
        description="Target customer recipient email address (redacted in audit logs).",
    )
    subject: str = Field(
        ...,
        min_length=5,
        max_length=160,
        description="Concise, executive-ready email subject line summarizing the GCP recommendation.",
    )
    body_markdown: str = Field(
        ...,
        min_length=20,
        description="Drafted customer email body in concise Markdown format with grounded citations.",
    )
    citations: List[str] = Field(
        ...,
        min_length=1,
        description="List of verified https://cloud.google.com documentation URLs grounding the email.",
    )
    operator_decision: Optional[str] = Field(
        default=None,
        description=(
            "Explicit human operator confirmation token ('APPROVE' or 'REJECT'). "
            "When None, execution halts in 'PENDING_HUMAN_APPROVAL' state."
        ),
    )


class EmailApprovalOutput(BaseModel):
    """Strict output schema returned by `request_human_email_approval`."""

    approval_id: str = Field(
        ...,
        description="Unique audit identifier for the HITL approval gate.",
    )
    status: str = Field(
        ...,
        description="Gate status: 'APPROVED_FOR_DISPATCH', 'PENDING_HUMAN_APPROVAL', 'REJECTED_BY_OPERATOR', or 'RECOVERABLE_ERROR'.",
    )
    dispatched: bool = Field(
        ...,
        description="True ONLY if an explicit human operator confirmation ('APPROVE') was verified.",
    )
    redacted_preview: str = Field(
        ...,
        description="PII-scrubbed preview of the outbound communication recorded in the audit log.",
    )
    recovery_guidance: Optional[str] = Field(
        default=None,
        description="Actionable instructions for the agent or operator on how to proceed.",
    )


_INTERACTIVE_CONFIRMATION_CALLBACK: Optional[Callable[[EmailApprovalInput], str]] = None


def set_confirmation_callback(
    callback: Optional[Callable[[EmailApprovalInput], str]],
) -> None:
    """Register an optional interactive human operator callback for HITL confirmation."""
    global _INTERACTIVE_CONFIRMATION_CALLBACK
    _INTERACTIVE_CONFIRMATION_CALLBACK = callback


def request_human_email_approval(
    recipient_email: str,
    subject: str,
    body_markdown: str,
    citations: List[str],
    operator_decision: Optional[str] = None,
) -> Dict[str, object]:
    """Enforce a mandatory Human-in-the-Loop (HITL) approval gate before dispatching any customer email.

    Validates the proposed email payload against `EmailApprovalInput`, scrubs customer PII before audit
    logging, and enforces an explicit code stop unless a human operator provides an explicit `'APPROVE'`
    decision. Never dispatches customer-facing communications autonomously without verification.

    Args:
        recipient_email: Target customer email address to receive the architectural summary.
        subject: Concise subject line (5-160 characters).
        body_markdown: Structured, non-verbose email body containing grounded recommendations.
        citations: Non-empty list of verified `https://cloud.google.com` URLs backing the claims.
        operator_decision: Optional explicit human decision string ('APPROVE' or 'REJECT').
            Defaults to None, which halts autonomous dispatch in `PENDING_HUMAN_APPROVAL` status.

    Returns:
        Dict[str, object]: Serialized `EmailApprovalOutput` containing:
            - `approval_id` (str): Unique HITL ticket ID.
            - `status` (str): 'APPROVED_FOR_DISPATCH', 'PENDING_HUMAN_APPROVAL', 'REJECTED_BY_OPERATOR', or 'RECOVERABLE_ERROR'.
            - `dispatched` (bool): True only when explicitly approved by a human operator.
            - `redacted_preview` (str): Sanitized email preview with PII removed.
            - `recovery_guidance` (Optional[str]): Next-step guidance if halted, rejected, or invalid.

    Raises:
        None: Catches validation and callback errors and returns structured `recovery_guidance` strings.
    """
    start_time = time.perf_counter()
    approval_id = f"hitl-{uuid.uuid4().hex[:10]}"

    with tracer.start_span(
        "tool.request_human_email_approval",
        attributes={"approval_id": approval_id, "citation_count": len(citations) if citations else 0},
    ):
        logger.log_intent(
            target="request_human_email_approval",
            reason="Enforce mandatory Human-in-the-Loop confirmation before customer email dispatch",
            parameters={
                "recipient_email": recipient_email,
                "subject": subject,
                "citation_count": len(citations) if citations else 0,
                "operator_decision": operator_decision or "AWAITING_OPERATOR",
            },
        )

        try:
            validated = EmailApprovalInput(
                recipient_email=recipient_email,
                subject=subject,
                body_markdown=body_markdown,
                citations=citations,
                operator_decision=operator_decision,
            )
        except ValidationError as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            guidance = (
                "Error: Outbound email failed schema validation in `request_human_email_approval`. "
                "Ensure `recipient_email`, `subject` (5-160 chars), `body_markdown` (>=20 chars), and at least "
                f"one grounded URL in `citations` are provided. Details: {exc.errors()[0]['msg']}"
            )
            logger.log_outcome(
                target="request_human_email_approval",
                status="RECOVERABLE_ERROR",
                latency_ms=latency_ms,
                summary=guidance,
            )
            return EmailApprovalOutput(
                approval_id=approval_id,
                status="RECOVERABLE_ERROR",
                dispatched=False,
                redacted_preview=str(redact_pii(body_markdown or "")),
                recovery_guidance=guidance,
            ).model_dump()

        if any(not url.startswith("https://cloud.google.com") for url in validated.citations):
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            guidance = (
                "Error: All email citations must be official `https://cloud.google.com` URLs retrieved from "
                "`search_gcp_documentation`. Remove unverified links and retry."
            )
            logger.log_outcome(
                target="request_human_email_approval",
                status="RECOVERABLE_ERROR",
                latency_ms=latency_ms,
                summary=guidance,
            )
            return EmailApprovalOutput(
                approval_id=approval_id,
                status="RECOVERABLE_ERROR",
                dispatched=False,
                redacted_preview=str(redact_pii(validated.body_markdown)),
                recovery_guidance=guidance,
            ).model_dump()

        sanitized_preview = str(
            redact_pii(f"To: {validated.recipient_email} | Subject: {validated.subject}\n{validated.body_markdown}")
        )

        try:
            resolved_decision = validated.operator_decision
            if resolved_decision is None and _INTERACTIVE_CONFIRMATION_CALLBACK is not None:
                resolved_decision = _INTERACTIVE_CONFIRMATION_CALLBACK(validated)

            normalized = (resolved_decision or "").strip().upper()
            latency_ms = (time.perf_counter() - start_time) * 1000.0

            # Explicit HITL Code Stop: Do not mark as approved unless operator explicitly confirms 'APPROVE'
            if normalized != "APPROVE":
                if normalized == "REJECT":
                    status = "REJECTED_BY_OPERATOR"
                    guidance = (
                        "Operator rejected the email draft. Revise the architectural summary or tone "
                        "based on operator feedback before requesting approval again."
                    )
                else:
                    status = "PENDING_HUMAN_APPROVAL"
                    guidance = (
                        f"HITL Gate Active (Ticket {approval_id}): Email draft is staged and awaiting explicit "
                        "human operator confirmation (`operator_decision='APPROVE'`) before dispatch."
                    )
                logger.log_outcome(
                    target="request_human_email_approval",
                    status=status,
                    latency_ms=latency_ms,
                    summary=guidance,
                )
                return EmailApprovalOutput(
                    approval_id=approval_id,
                    status=status,
                    dispatched=False,
                    redacted_preview=sanitized_preview,
                    recovery_guidance=guidance,
                ).model_dump()

            logger.log_outcome(
                target="request_human_email_approval",
                status="APPROVED_FOR_DISPATCH",
                latency_ms=latency_ms,
                summary=f"Human operator explicitly approved email ticket {approval_id} for dispatch.",
            )
            return EmailApprovalOutput(
                approval_id=approval_id,
                status="APPROVED_FOR_DISPATCH",
                dispatched=True,
                redacted_preview=sanitized_preview,
                recovery_guidance=None,
            ).model_dump()

        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            guidance = (
                f"Error: Human approval callback failed ({type(exc).__name__}: {exc}). "
                "Email remains blocked from dispatch; supply `operator_decision` explicitly."
            )
            logger.log_outcome(
                target="request_human_email_approval",
                status="RECOVERABLE_ERROR",
                latency_ms=latency_ms,
                summary=guidance,
            )
            return EmailApprovalOutput(
                approval_id=approval_id,
                status="RECOVERABLE_ERROR",
                dispatched=False,
                redacted_preview=sanitized_preview,
                recovery_guidance=guidance,
            ).model_dump()
