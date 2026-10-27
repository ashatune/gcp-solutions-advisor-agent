"""OpenTelemetry distributed tracing wrapper covering entry queries, sub-agent delegations, and tool calls."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import time
from typing import Any, Dict, Iterator, List, Optional
import uuid

try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

    OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover
    OTEL_AVAILABLE = False

_CURRENT_TRACE_ID: ContextVar[str] = ContextVar("current_trace_id", default="")


@dataclass
class SpanRecord:
    """Lightweight representation of a recorded distributed trace span."""

    name: str
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    start_time_ns: int
    end_time_ns: int = 0
    attributes: Dict[str, Any] = field(default_factory=dict)
    status: str = "OK"

    @property
    def duration_ms(self) -> float:
        """Compute span duration in milliseconds."""
        if self.end_time_ns <= self.start_time_ns:
            return 0.0
        return (self.end_time_ns - self.start_time_ns) / 1_000_000.0


if OTEL_AVAILABLE:

    class _InMemorySpanExporter(SpanExporter):
        """Captures OpenTelemetry spans in memory without requiring an external collector."""

        def __init__(self) -> None:
            self.spans: List[ Any] = []

        def export(self, spans: Any) -> "SpanExportResult":
            self.spans.extend(spans)
            return SpanExportResult.SUCCESS

        def shutdown(self) -> None:
            return None


class TelemetryTracer:
    """Manages OpenTelemetry spans across coordinator queries, sub-agent delegations, and tool invocations."""

    def __init__(self, service_name: str = "gcp-solutions-advisor-agent") -> None:
        self.service_name = service_name
        self._recorded_spans: List[SpanRecord] = []
        self._active_span_stack: List[str] = []

        if OTEL_AVAILABLE:
            resource = Resource.create({"service.name": service_name})
            self._provider = TracerProvider(resource=resource)
            self._exporter = _InMemorySpanExporter()
            self._provider.add_span_processor(SimpleSpanProcessor(self._exporter))
            self._otel_tracer = self._provider.get_tracer(service_name)
        else:
            self._otel_tracer = None

    @contextmanager
    def start_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
    ) -> Iterator[SpanRecord]:
        """Start a distributed trace span and record timing and attributes.

        Args:
            name: Span name (e.g., 'coordinator.handle_inquiry', 'sub_agent.research', 'tool.search_gcp_documentation').
            attributes: Metadata attributes to attach to the span.
            trace_id: Optional explicit trace identifier; generates a new one if root span.

        Yields:
            SpanRecord capturing the active span context.
        """
        active_trace_id = trace_id or _CURRENT_TRACE_ID.get() or uuid.uuid4().hex
        token = _CURRENT_TRACE_ID.set(active_trace_id)
        parent_span_id = self._active_span_stack[-1] if self._active_span_stack else None
        span_id = uuid.uuid4().hex[:16]

        record = SpanRecord(
            name=name,
            trace_id=active_trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            start_time_ns=time.time_ns(),
            attributes=dict(attributes or {}),
        )
        self._active_span_stack.append(span_id)

        if self._otel_tracer is not None:
            with self._otel_tracer.start_as_current_span(name) as otel_span:
                for key, value in record.attributes.items():
                    otel_span.set_attribute(key, str(value))
                try:
                    yield record
                except Exception as exc:
                    record.status = f"ERROR: {type(exc).__name__}"
                    otel_span.record_exception(exc)
                    raise
                finally:
                    record.end_time_ns = time.time_ns()
                    self._active_span_stack.pop()
                    self._recorded_spans.append(record)
                    if not self._active_span_stack:
                        _CURRENT_TRACE_ID.reset(token)
        else:
            try:
                yield record
            except Exception as exc:
                record.status = f"ERROR: {type(exc).__name__}"
                raise
            finally:
                record.end_time_ns = time.time_ns()
                self._active_span_stack.pop()
                self._recorded_spans.append(record)
                if not self._active_span_stack:
                    _CURRENT_TRACE_ID.reset(token)

    def get_recorded_spans(self) -> List[SpanRecord]:
        """Return all recorded spans for observability verification."""
        return list(self._recorded_spans)

    def clear_spans(self) -> None:
        """Clear recorded spans (used between test runs)."""
        self._recorded_spans.clear()


_DEFAULT_TRACER = TelemetryTracer()


def get_tracer() -> TelemetryTracer:
    """Return the shared TelemetryTracer instance."""
    return _DEFAULT_TRACER


def get_current_trace_id() -> str:
    """Return the active trace ID or generate a fallback trace ID."""
    current = _CURRENT_TRACE_ID.get()
    return current if current else "no-active-trace"
