"""Persistent session backend with async customer profile extraction and PII scrubbing."""

import asyncio
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory.compactor import ConversationCompactor, TurnMessage
from agent.telemetry.logger import get_logger
from agent.telemetry.pii_redactor import redact_pii

logger = get_logger("memory.session_store")


@dataclass
class CustomerProfile:
    """Extracted architectural context for a multi-turn enterprise customer engagement."""

    primary_workloads: List[str] = field(default_factory=list)
    preferred_gcp_services: List[str] = field(default_factory=list)
    compliance_requirements: List[str] = field(default_factory=list)
    target_regions: List[str] = field(default_factory=list)


@dataclass
class SessionState:
    """Serializable multi-turn session state persisted across restarts."""

    session_id: str
    turns: List[TurnMessage] = field(default_factory=list)
    compacted_summary: str = ""
    customer_profile: CustomerProfile = field(default_factory=CustomerProfile)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PersistentSessionStore:
    """Disk-backed JSON persistent session store with non-blocking async memory consolidation."""

    _SERVICE_KEYWORDS = (
        "cloud run",
        "gke",
        "cloud spanner",
        "cloud sql",
        "bigquery",
        "dataflow",
        "pub/sub",
        "vertex ai",
        "secret manager",
        "vpc service controls",
    )
    _COMPLIANCE_KEYWORDS = ("hipaa", "pci-dss", "soc2", "gdpr", "fedramp", "cmek", "zero-trust")
    _REGION_KEYWORDS = ("us-central1", "us-east1", "europe-west1", "asia-east1", "multi-region")

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        compactor: Optional[ConversationCompactor] = None,
    ) -> None:
        self.storage_path = storage_path
        self.compactor = compactor or ConversationCompactor()
        self._sessions: Dict[str, SessionState] = {}
        self._background_tasks: set[asyncio.Task[Any]] = set()
        if self.storage_path and self.storage_path.exists():
            self._load_from_disk()

    def _load_from_disk(self) -> None:
        """Load persisted sessions from JSON storage on initialization."""
        if not self.storage_path or not self.storage_path.exists():
            return
        try:
            raw_data = json.loads(self.storage_path.read_text(encoding="utf-8"))
            for session_id, state_dict in raw_data.items():
                turns = [TurnMessage(**t) for t in state_dict.get("turns", [])]
                profile = CustomerProfile(**state_dict.get("customer_profile", {}))
                self._sessions[session_id] = SessionState(
                    session_id=session_id,
                    turns=turns,
                    compacted_summary=state_dict.get("compacted_summary", ""),
                    customer_profile=profile,
                    metadata=state_dict.get("metadata", {}),
                )
        except Exception as exc:
            logger.warning(f"Failed to load session store from disk: {exc}")

    def _persist_to_disk(self) -> None:
        """Write current sanitized session states to disk if `storage_path` is configured."""
        if not self.storage_path:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            sid: redact_pii(asdict(state)) for sid, state in self._sessions.items()
        }
        self.storage_path.write_text(
            json.dumps(serializable, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def get_or_create_session(self, session_id: str) -> SessionState:
        """Retrieve an existing `SessionState` or initialize a new one."""
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionState(session_id=session_id)
        return self._sessions[session_id]

    def _extract_profile_signals(self, text: str, profile: CustomerProfile) -> None:
        """Extract GCP service interests, compliance requirements, and regions from sanitized text."""
        lower = text.lower()
        for svc in self._SERVICE_KEYWORDS:
            if svc in lower and svc not in profile.preferred_gcp_services:
                profile.preferred_gcp_services.append(svc)
        for comp in self._COMPLIANCE_KEYWORDS:
            if comp in lower and comp not in profile.compliance_requirements:
                profile.compliance_requirements.append(comp)
        for reg in self._REGION_KEYWORDS:
            if reg in lower and reg not in profile.target_regions:
                profile.target_regions.append(reg)

    async def _async_consolidate_session(self, session_id: str, latest_content: str) -> None:
        """Non-blocking background coroutine for profile extraction, compaction, and disk persistence."""
        await asyncio.sleep(0)
        state = self.get_or_create_session(session_id)
        self._extract_profile_signals(latest_content, state.customer_profile)

        summary_holder = {"summary": state.compacted_summary}
        retained_turns, compaction_task = self.compactor.schedule_compaction(
            session_id=session_id,
            turns=state.turns,
            summary_holder=summary_holder,
        )
        state.turns = retained_turns
        if compaction_task is not None:
            await compaction_task
        state.compacted_summary = summary_holder["summary"]
        self._persist_to_disk()

    def append_turn(self, session_id: str, role: str, content: str) -> TurnMessage:
        """Append a PII-redacted turn and trigger non-blocking async consolidation when an event loop is active."""
        state = self.get_or_create_session(session_id)
        clean_content = str(redact_pii(content))
        turn = TurnMessage(role=role, content=clean_content)
        state.turns.append(turn)

        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(self._async_consolidate_session(session_id, clean_content))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
        except RuntimeError:
            # Synchronous execution path fallback
            self._extract_profile_signals(clean_content, state.customer_profile)
            summary_holder = {"summary": state.compacted_summary}
            retained_turns, _ = self.compactor.schedule_compaction(
                session_id=session_id,
                turns=state.turns,
                summary_holder=summary_holder,
            )
            state.turns = retained_turns
            state.compacted_summary = summary_holder["summary"]
            self._persist_to_disk()

        return turn

    async def flush_background_tasks(self) -> None:
        """Wait for all scheduled async profile extraction and compaction tasks to finish."""
        if self._background_tasks:
            await asyncio.gather(*list(self._background_tasks), return_exceptions=True)
        await self.compactor.wait_for_background_tasks()
        self._persist_to_disk()
