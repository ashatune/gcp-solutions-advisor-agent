"""Structured JSON logger with explicit [INTENT] vs [OUTCOME] capture and automatic PII scrubbing."""

from datetime import datetime, timezone
import json
import logging
import sys
from typing import Any, Dict, List, Optional

from agent.telemetry.pii_redactor import PIIRedactor
from agent.telemetry.tracer import get_current_trace_id


class JSONLogFormatter(logging.Formatter):
    """Formats Python log records as single-line JSON with timestamp, level, trace_id, component, and message."""

    def __init__(self, redactor: Optional[PIIRedactor] = None) -> None:
        super().__init__()
        self._redactor = redactor or PIIRedactor()

    def format(self, record: logging.LogRecord) -> str:
        trace_id = getattr(record, "trace_id", None) or get_current_trace_id()
        component = getattr(record, "component", record.name)
        raw_message = record.getMessage()
        sanitized_message = self._redactor.scrub_text(raw_message)

        log_object: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "trace_id": trace_id,
            "component": component,
            "message": sanitized_message,
        }

        extra_metadata = getattr(record, "metadata", None)
        if extra_metadata:
            log_object["metadata"] = self._redactor.scrub_payload(extra_metadata)

        return json.dumps(log_object, ensure_ascii=False)


class StructuredLogger:
    """Enterprise AgentOps logger capturing [INTENT] and [OUTCOME] events with PII redaction."""

    def __init__(self, component: str = "gcp_solutions_advisor") -> None:
        self.component = component
        self._redactor = PIIRedactor()
        self._records: List[Dict[str, Any]] = []
        self._logger = logging.getLogger(f"advisor.{component}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False

        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(JSONLogFormatter(redactor=self._redactor))
            self._logger.addHandler(handler)

    def _emit(
        self,
        level: int,
        message: str,
        trace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        resolved_trace_id = trace_id or get_current_trace_id()
        clean_message = self._redactor.scrub_text(message)
        clean_meta = self._redactor.scrub_payload(metadata or {})

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": logging.getLevelName(level),
            "trace_id": resolved_trace_id,
            "component": self.component,
            "message": clean_message,
            "metadata": clean_meta,
        }
        self._records.append(entry)
        self._logger.log(
            level,
            clean_message,
            extra={
                "trace_id": resolved_trace_id,
                "component": self.component,
                "metadata": clean_meta,
            },
        )
        return entry

    def info(
        self,
        message: str,
        trace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Log an informational message with automatic PII scrubbing."""
        return self._emit(logging.INFO, message, trace_id=trace_id, metadata=metadata)

    def warning(
        self,
        message: str,
        trace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Log a warning message with automatic PII scrubbing."""
        return self._emit(logging.WARNING, message, trace_id=trace_id, metadata=metadata)

    def error(
        self,
        message: str,
        trace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Log an error message with automatic PII scrubbing."""
        return self._emit(logging.ERROR, message, trace_id=trace_id, metadata=metadata)

    def log_intent(
        self,
        target: str,
        reason: str,
        parameters: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record an explicit [INTENT] log entry prior to executing a tool or sub-agent call."""
        message = f"[INTENT] Invoking '{target}' | Reason: {reason}"
        return self._emit(
            logging.INFO,
            message,
            trace_id=trace_id,
            metadata={"phase": "INTENT", "target": target, "parameters": parameters or {}},
        )

    def log_outcome(
        self,
        target: str,
        status: str,
        latency_ms: float,
        summary: str,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record an explicit [OUTCOME] log entry following tool or sub-agent execution."""
        message = (
            f"[OUTCOME] Completed '{target}' | Status: {status} | "
            f"Latency: {latency_ms:.2f}ms | Summary: {summary}"
        )
        return self._emit(
            logging.INFO,
            message,
            trace_id=trace_id,
            metadata={
                "phase": "OUTCOME",
                "target": target,
                "status": status,
                "latency_ms": round(latency_ms, 2),
                "summary": summary,
            },
        )

    def get_captured_records(self) -> List[Dict[str, Any]]:
        """Return in-memory structured log entries for inspection in tests and evaluations."""
        return list(self._records)


_LOGGERS: Dict[str, StructuredLogger] = {}


def get_logger(component: str = "gcp_solutions_advisor") -> StructuredLogger:
    """Get or create a singleton StructuredLogger for the given component."""
    if component not in _LOGGERS:
        _LOGGERS[component] = StructuredLogger(component=component)
    return _LOGGERS[component]
