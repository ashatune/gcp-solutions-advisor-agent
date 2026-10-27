"""Grounded GCP documentation search tool and Secret Manager integration with guided error recovery."""

import os
import time
from typing import Dict, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from agent.telemetry.logger import get_logger
from agent.telemetry.tracer import get_tracer

load_dotenv()

try:
    from google.cloud import secretmanager  # type: ignore

    SECRET_MANAGER_AVAILABLE = True
except ImportError:  # pragma: no cover
    SECRET_MANAGER_AVAILABLE = False

logger = get_logger("tools.gcp_search")
tracer = get_tracer()


class GCPDocumentChunk(BaseModel):
    """Schema representing a single verified Google Cloud documentation excerpt."""

    title: str = Field(
        ...,
        description="Official title of the Google Cloud documentation page or architecture guide.",
    )
    url: str = Field(
        ...,
        description="Canonical https://cloud.google.com documentation URL used for citation grounding.",
    )
    service_category: str = Field(
        ...,
        description="Primary GCP service family (e.g., 'compute', 'databases', 'networking', 'security', 'ai-ml').",
    )
    summary: str = Field(
        ...,
        description="Grounded technical excerpt detailing capabilities, SLAs, quotas, or architectural trade-offs.",
    )
    supported_regions: List[str] = Field(
        default_factory=list,
        description="List of verified GCP regions where the feature or architecture pattern is supported.",
    )


class GCPDocSearchInput(BaseModel):
    """Strict input schema for querying official Google Cloud documentation."""

    query: str = Field(
        ...,
        min_length=3,
        max_length=300,
        description="Technical search query describing the GCP services, workload pattern, or architectural question.",
    )
    service_filter: Optional[str] = Field(
        default=None,
        description=(
            "Optional GCP service category filter. Valid values: 'compute', 'databases', "
            "'networking', 'security', 'serverless', 'analytics', 'ai-ml'."
        ),
    )
    region_filter: Optional[str] = Field(
        default=None,
        description="Optional GCP region constraint (e.g., 'us-central1', 'europe-west1', 'multi-region').",
    )
    max_results: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of grounded documentation chunks to return (between 1 and 10).",
    )


class GCPDocSearchOutput(BaseModel):
    """Strict output schema returned by `search_gcp_documentation`."""

    status: str = Field(
        ...,
        description="Execution status: 'SUCCESS', 'NO_RESULTS', or 'RECOVERABLE_ERROR'.",
    )
    query_executed: str = Field(
        ...,
        description="Normalized search query that was executed against the GCP knowledge index.",
    )
    results: List[GCPDocumentChunk] = Field(
        default_factory=list,
        description="List of verified GCP documentation chunks matching the query criteria.",
    )
    recovery_guidance: Optional[str] = Field(
        default=None,
        description="Actionable guidance for the LLM when a query returns no results or encounters validation issues.",
    )


# Curated, grounded GCP Reference Knowledge Base for deterministic offline & CI evaluation
_CURATED_GCP_DOCS: List[GCPDocumentChunk] = [
    GCPDocumentChunk(
        title="Cloud Run vs Google Kubernetes Engine (GKE) Architecture Guide",
        url="https://cloud.google.com/docs/choosing-a-compute-option",
        service_category="compute",
        summary=(
            "Cloud Run provides fully managed serverless container execution with scale-to-zero, automatic HTTPS "
            "load balancing, and per-100ms billing (99.95% multi-zonal SLA). GKE Autopilot and Standard suit complex "
            "microservices needing custom networking meshes, StatefulSets, GPU node pools, or sidecar daemonsets."
        ),
        supported_regions=["us-central1", "us-east1", "europe-west1", "asia-east1", "multi-region"],
    ),
    GCPDocumentChunk(
        title="Cloud Spanner vs Cloud SQL for PostgreSQL: Relational Database Trade-offs",
        url="https://cloud.google.com/Architecture-Framework/databases",
        service_category="databases",
        summary=(
            "Cloud Spanner delivers globally distributed horizontal write scaling with external consistency and "
            "up to 99.999% multi-region SLA, ideal for mission-critical financial ledgers and global inventory. "
            "Cloud SQL for PostgreSQL offers lower entry cost and full PostgreSQL extension compatibility for "
            "regional OLTP workloads up to 64 TB with 99.99% High Availability SLA."
        ),
        supported_regions=["us-central1", "us-east1", "europe-west1", "nam-eur-asia1", "multi-region"],
    ),
    GCPDocumentChunk(
        title="BigQuery Data Warehouse & Dataflow Streaming Analytics Reference Architecture",
        url="https://cloud.google.com/architecture/big-data-analytics/streaming",
        service_category="analytics",
        summary=(
            "Pub/Sub ingests global event streams at scale with at-least-once delivery, feeding Apache Beam "
            "pipelines on Dataflow for windowing, deduplication, and PII redaction before writing via BigQuery "
            "Storage Write API for sub-second real-time operational analytics."
        ),
        supported_regions=["us-central1", "us-east1", "europe-west1", "US", "EU", "multi-region"],
    ),
    GCPDocumentChunk(
        title="VPC Service Controls, Secret Manager, and Zero-Trust Security Best Practices",
        url="https://cloud.google.com/security/best-practices",
        service_category="security",
        summary=(
            "Protect sensitive APIs against data exfiltration using VPC Service Controls security perimeters, "
            "Cloud KMS customer-managed encryption keys (CMEK), and Secret Manager with least-privilege workload "
            "identity federation instead of service account key files."
        ),
        supported_regions=["us-central1", "us-east1", "europe-west1", "global", "multi-region"],
    ),
    GCPDocumentChunk(
        title="Vertex AI Agent Builder, Gemini 2.5 Routing, and Grounded Enterprise RAG",
        url="https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/overview",
        service_category="ai-ml",
        summary=(
            "Deploy enterprise multi-agent systems using Google Agent Development Kit (ADK) and Vertex AI Agent Engine. "
            "Use Gemini 2.5 Flash for low-latency document grounding and classification, and Gemini 2.5 Pro for "
            "complex multi-constraint architectural synthesis and trade-off matrices."
        ),
        supported_regions=["us-central1", "us-east4", "europe-west1", "asia-northeast1"],
    ),
]


def retrieve_gcp_secret(
    secret_id: str,
    project_id: Optional[str] = None,
    env_fallback_key: Optional[str] = None,
) -> str:
    """Retrieve a sensitive credential from Google Cloud Secret Manager with local `.env` fallback.

    Queries Google Cloud Secret Manager `versions/latest` for the specified `secret_id` when running
    in a configured GCP environment. Gracefully falls back to environment variables loaded from `.env`
    during local development or automated CI evaluation so no secrets are ever hardcoded.

    Args:
        secret_id: Identifier of the secret in Google Cloud Secret Manager (e.g., 'gemini-api-key').
        project_id: Optional GCP project ID. Defaults to `GCP_PROJECT_ID` environment variable.
        env_fallback_key: Optional environment variable name to check if Secret Manager is unreachable.

    Returns:
        Resolved secret value string, or a safe placeholder indicator if neither source is configured.

    Raises:
        ValueError: Only if `secret_id` is empty or whitespace.
    """
    if not secret_id or not secret_id.strip():
        raise ValueError("secret_id must be a non-empty string.")

    resolved_project = project_id or os.getenv("GCP_PROJECT_ID", "")
    if (
        SECRET_MANAGER_AVAILABLE
        and resolved_project
        and resolved_project != "your-gcp-project-id"
        and os.getenv("ENABLE_LIVE_SECRET_MANAGER", "false").lower() == "true"
    ):
        try:
            client = secretmanager.SecretManagerServiceClient()
            resource_name = f"projects/{resolved_project}/secrets/{secret_id}/versions/latest"
            response = client.access_secret_version(request={"name": resource_name})
            return response.payload.data.decode("UTF-8")
        except Exception as exc:
            logger.warning(
                f"Secret Manager lookup failed for '{secret_id}', falling back to environment variable: {exc}"
            )

    fallback_env = env_fallback_key or secret_id.upper().replace("-", "_")
    return os.getenv(fallback_env, f"unconfigured-secret:{secret_id}")


def search_gcp_documentation(
    query: str,
    service_filter: Optional[str] = None,
    region_filter: Optional[str] = None,
    max_results: int = 3,
) -> Dict[str, object]:
    """Search official Google Cloud documentation for grounded architectural facts, SLAs, and trade-offs.

    Validates inputs against `GCPDocSearchInput`, executes a relevance search across verified Google Cloud
    architecture references, emits OpenTelemetry spans and `[INTENT]`/`[OUTCOME]` audit logs, and returns
    a serialized `GCPDocSearchOutput` dictionary. Never raises unhandled runtime exceptions to the LLM;
    instead returns structured `recovery_guidance` strings so the agent can self-correct its search query.

    Args:
        query: Technical search query describing the GCP services, workload pattern, or architectural question.
        service_filter: Optional GCP service category filter ('compute', 'databases', 'networking',
            'security', 'serverless', 'analytics', 'ai-ml').
        region_filter: Optional GCP region constraint (e.g., 'us-central1', 'europe-west1', 'multi-region').
        max_results: Maximum number of grounded documentation chunks to return (1 to 10, default 3).

    Returns:
        Dict[str, object]: Serialized `GCPDocSearchOutput` containing:
            - `status` (str): 'SUCCESS', 'NO_RESULTS', or 'RECOVERABLE_ERROR'.
            - `query_executed` (str): Normalized query string.
            - `results` (List[Dict]): Grounded `GCPDocumentChunk` dictionaries with `title`, `url`, `summary`.
            - `recovery_guidance` (Optional[str]): Guided corrective instructions if the query failed or returned 0 docs.

    Raises:
        None: All `ValidationError` and runtime exceptions are caught and converted into guided recovery strings.
    """
    start_time = time.perf_counter()
    with tracer.start_span(
        "tool.search_gcp_documentation",
        attributes={
            "query": query,
            "service_filter": service_filter or "none",
            "region_filter": region_filter or "none",
        },
    ):
        logger.log_intent(
            target="search_gcp_documentation",
            reason="Retrieve grounded official GCP documentation before formulating architectural guidance",
            parameters={
                "query": query,
                "service_filter": service_filter,
                "region_filter": region_filter,
                "max_results": max_results,
            },
        )

        try:
            validated_input = GCPDocSearchInput(
                query=query,
                service_filter=service_filter,
                region_filter=region_filter,
                max_results=max_results,
            )
        except ValidationError as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            guidance = (
                "Error: Invalid search parameters provided to `search_gcp_documentation`. "
                "Ensure `query` is between 3 and 300 characters and `max_results` is between 1 and 10. "
                f"Validation details: {exc.errors()[0]['msg']}"
            )
            logger.log_outcome(
                target="search_gcp_documentation",
                status="RECOVERABLE_ERROR",
                latency_ms=latency_ms,
                summary=guidance,
            )
            return GCPDocSearchOutput(
                status="RECOVERABLE_ERROR",
                query_executed=str(query),
                results=[],
                recovery_guidance=guidance,
            ).model_dump()

        try:
            valid_categories = {
                "compute",
                "databases",
                "networking",
                "security",
                "serverless",
                "analytics",
                "ai-ml",
            }
            if (
                validated_input.service_filter
                and validated_input.service_filter.lower() not in valid_categories
            ):
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                guidance = (
                    f"Error: Unsupported service_filter '{validated_input.service_filter}'. "
                    f"Suggest retrying with `service_filter=None` or one of: {sorted(valid_categories)}."
                )
                logger.log_outcome(
                    target="search_gcp_documentation",
                    status="RECOVERABLE_ERROR",
                    latency_ms=latency_ms,
                    summary=guidance,
                )
                return GCPDocSearchOutput(
                    status="RECOVERABLE_ERROR",
                    query_executed=validated_input.query,
                    results=[],
                    recovery_guidance=guidance,
                ).model_dump()

            if "TRIGGER_SIMULATED_API_TIMEOUT" in validated_input.query:
                raise TimeoutError("Upstream Vertex AI Search index timed out after 5000ms.")

            query_tokens = {
                token.lower().strip(".,?!")
                for token in validated_input.query.split()
                if len(token.strip(".,?!")) >= 2
            }

            scored_chunks: List[tuple[int, GCPDocumentChunk]] = []
            for doc in _CURATED_GCP_DOCS:
                if validated_input.service_filter:
                    filt = validated_input.service_filter.lower()
                    if filt == "serverless" and doc.service_category != "compute":
                        continue
                    elif filt != "serverless" and doc.service_category != filt:
                        continue

                if validated_input.region_filter:
                    if validated_input.region_filter not in doc.supported_regions:
                        continue

                haystack = f"{doc.title} {doc.service_category} {doc.summary}".lower()
                score = sum(2 if tok in doc.title.lower() else 1 for tok in query_tokens if tok in haystack)
                if score > 0:
                    scored_chunks.append((score, doc))

            scored_chunks.sort(key=lambda item: item[0], reverse=True)
            matched_docs = [item[1] for item in scored_chunks[: validated_input.max_results]]

            latency_ms = (time.perf_counter() - start_time) * 1000.0
            if not matched_docs:
                guidance = (
                    "Error: Query returned no docs. Suggest broadening search terms, removing "
                    "`region_filter` or `service_filter` constraints, or querying core GCP service names "
                    "(e.g., 'Cloud Run GKE', 'Cloud Spanner Cloud SQL', 'BigQuery Dataflow', 'VPC Service Controls')."
                )
                logger.log_outcome(
                    target="search_gcp_documentation",
                    status="NO_RESULTS",
                    latency_ms=latency_ms,
                    summary=guidance,
                )
                return GCPDocSearchOutput(
                    status="NO_RESULTS",
                    query_executed=validated_input.query,
                    results=[],
                    recovery_guidance=guidance,
                ).model_dump()

            logger.log_outcome(
                target="search_gcp_documentation",
                status="SUCCESS",
                latency_ms=latency_ms,
                summary=f"Retrieved {len(matched_docs)} grounded GCP documentation chunks.",
            )
            return GCPDocSearchOutput(
                status="SUCCESS",
                query_executed=validated_input.query,
                results=matched_docs,
                recovery_guidance=None,
            ).model_dump()

        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            guidance = (
                f"Error: Documentation search encountered a transient error ({type(exc).__name__}: {exc}). "
                "Suggest retrying with a simpler keyword query and omitting optional filters."
            )
            logger.log_outcome(
                target="search_gcp_documentation",
                status="RECOVERABLE_ERROR",
                latency_ms=latency_ms,
                summary=guidance,
            )
            return GCPDocSearchOutput(
                status="RECOVERABLE_ERROR",
                query_executed=validated_input.query,
                results=[],
                recovery_guidance=guidance,
            ).model_dump()
