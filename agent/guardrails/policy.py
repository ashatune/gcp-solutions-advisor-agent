"""Grounding verification and anti-hallucination guardrail policy plugin."""

from dataclasses import dataclass, field
import re
from typing import Dict, List

from agent.telemetry.logger import get_logger
from agent.telemetry.pii_redactor import redact_pii

logger = get_logger("guardrails.policy")


@dataclass
class GuardrailEvaluationResult:
    """Structured result of evaluating a candidate agent response against grounding and safety policies."""

    passed: bool
    grounding_score: float
    verified_citations: List[str] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)
    sanitized_response: str = ""


class GroundingPolicyEvaluator:
    """Validates candidate responses against retrieved GCP documentation sources and blocks unsupported claims."""

    _PROHIBITED_CLAIM_PATTERNS = [
        (
            re.compile(r"\b100(?:\.0+)?%\s*(?:sla|uptime|availability)\b", re.IGNORECASE),
            "Unsupported 100% SLA guarantee claim (no GCP service offers a 100% uptime SLA).",
        ),
        (
            re.compile(r"\bunlimited\s+free\s+(?:tier|compute|gpu|spanner)\b", re.IGNORECASE),
            "Unsupported 'unlimited free' resource claim not backed by official GCP pricing docs.",
        ),
        (
            re.compile(r"\bzero[- ]latency\s+global\s+replication\b", re.IGNORECASE),
            "Physically impossible 'zero-latency global replication' claim.",
        ),
    ]

    def __init__(self, minimum_grounding_score: float = 0.65) -> None:
        self.minimum_grounding_score = minimum_grounding_score

    def verify_response(
        self,
        candidate_response: str,
        retrieved_docs: List[Dict[str, object]],
    ) -> GuardrailEvaluationResult:
        """Verify that `candidate_response` is grounded in `retrieved_docs` and free of prohibited hallucinations.

        Args:
            candidate_response: Draft response produced by the advisor agent.
            retrieved_docs: List of grounded `GCPDocumentChunk` dictionaries returned by `search_gcp_documentation`.

        Returns:
            GuardrailEvaluationResult indicating pass/fail status, grounding score, citations, and safe output.
        """
        violations: List[str] = []
        clean_response = str(redact_pii(candidate_response))

        for pattern, reason in self._PROHIBITED_CLAIM_PATTERNS:
            if pattern.search(clean_response):
                violations.append(reason)

        if not retrieved_docs:
            violations.append(
                "No grounded GCP documentation sources were provided to verify architectural claims."
            )
            logger.warning(
                "Guardrail blocked ungrounded response due to missing documentation sources.",
                metadata={"violations": violations},
            )
            blocked_msg = (
                "I cannot verify this architectural recommendation against official Google Cloud documentation "
                "at this time. Please refine the query so grounded reference sources can be retrieved."
            )
            return GuardrailEvaluationResult(
                passed=False,
                grounding_score=0.0,
                verified_citations=[],
                violations=violations,
                sanitized_response=blocked_msg,
            )

        allowed_urls = {str(doc.get("url", "")) for doc in retrieved_docs if doc.get("url")}
        cited_urls = re.findall(r"https://cloud\.google\.com[^\s\)\]>\"']*", clean_response)
        verified_citations = [url for url in cited_urls if url in allowed_urls]
        unverified_citations = [url for url in cited_urls if url not in allowed_urls]

        if unverified_citations:
            violations.append(
                f"Response cites URLs not present in retrieved grounding docs: {unverified_citations}"
            )

        if not verified_citations:
            violations.append(
                "Response must include at least one verified `https://cloud.google.com` citation from retrieved sources."
            )

        source_corpus = " ".join(
            f"{doc.get('title', '')} {doc.get('summary', '')} {doc.get('service_category', '')}"
            for doc in retrieved_docs
        ).lower()
        source_tokens = {
            tok.strip(".,():;")
            for tok in source_corpus.split()
            if len(tok.strip(".,():;")) >= 4
        }
        response_tokens = {
            tok.strip(".,():;*")
            for tok in clean_response.lower().split()
            if len(tok.strip(".,():;*")) >= 4
        }

        overlap = len(source_tokens.intersection(response_tokens))
        overlap_ratio = min(1.0, overlap / max(6, min(len(source_tokens), 20)))
        citation_bonus = 0.25 if verified_citations else 0.0
        grounding_score = round(min(1.0, overlap_ratio + citation_bonus), 2)

        if grounding_score < self.minimum_grounding_score:
            violations.append(
                f"Grounding score {grounding_score:.2f} is below policy threshold {self.minimum_grounding_score:.2f}."
            )

        passed = len(violations) == 0
        if not passed:
            logger.warning(
                "Guardrail policy blocked candidate response.",
                metadata={"grounding_score": grounding_score, "violations": violations},
            )
            safe_fallback = (
                "Guardrail Policy Block: The generated response contained claims that could not be verified "
                f"against retrieved Google Cloud documentation ({'; '.join(violations)})."
            )
            return GuardrailEvaluationResult(
                passed=False,
                grounding_score=grounding_score,
                verified_citations=verified_citations,
                violations=violations,
                sanitized_response=safe_fallback,
            )

        return GuardrailEvaluationResult(
            passed=True,
            grounding_score=grounding_score,
            verified_citations=verified_citations,
            violations=[],
            sanitized_response=clean_response,
        )
