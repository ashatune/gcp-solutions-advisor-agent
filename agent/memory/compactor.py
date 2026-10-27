"""Token-based sliding window history compactor with non-blocking asyncio background summarization."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from agent.telemetry.logger import get_logger
from agent.telemetry.pii_redactor import redact_pii

logger = get_logger("memory.compactor")


@dataclass
class TurnMessage:
    """Represents a single conversational turn with estimated token count."""

    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    token_count: int = 0

    def __post_init__(self) -> None:
        self.content = str(redact_pii(self.content))
        if self.token_count <= 0:
            self.token_count = max(1, len(self.content.split()) * 4 // 3)


class ConversationCompactor:
    """Enforces a token budget and sliding window by compacting older turns asynchronously in the background."""

    def __init__(
        self,
        max_tokens: int = 800,
        sliding_window_turns: int = 4,
    ) -> None:
        self.max_tokens = max_tokens
        self.sliding_window_turns = sliding_window_turns
        self._background_tasks: set[asyncio.Task[str]] = set()

    @staticmethod
    def estimate_tokens(turns: List[TurnMessage]) -> int:
        """Compute total estimated tokens across a list of conversation turns."""
        return sum(turn.token_count for turn in turns)

    def needs_compaction(self, turns: List[TurnMessage]) -> bool:
        """Determine whether the turn history exceeds either the token budget or sliding window threshold."""
        if len(turns) <= self.sliding_window_turns:
            return False
        return (
            self.estimate_tokens(turns) > self.max_tokens
            or len(turns) > self.sliding_window_turns * 2
        )

    def summarize_turns_sync(
        self,
        evicted_turns: List[TurnMessage],
        existing_summary: str = "",
    ) -> str:
        """Synthesize a concise, PII-scrubbed architectural context summary from evicted turns."""
        highlights: List[str] = []
        if existing_summary:
            highlights.append(existing_summary.strip())

        for turn in evicted_turns:
            snippet = " ".join(turn.content.strip().split())
            if len(snippet) > 140:
                snippet = snippet[:137] + "..."
            highlights.append(f"[{turn.role.upper()}]: {snippet}")

        combined = " | ".join(highlights)
        return str(redact_pii(f"Compacted Context Summary: {combined}"))

    async def _async_compact_worker(
        self,
        session_id: str,
        evicted_turns: List[TurnMessage],
        existing_summary: str,
        target_container: Dict[str, str],
    ) -> str:
        """Background coroutine that summarizes older turns without blocking the main user response path."""
        await asyncio.sleep(0)
        summary = self.summarize_turns_sync(evicted_turns, existing_summary=existing_summary)
        target_container["summary"] = summary
        logger.info(
            f"Completed async history compaction for session '{session_id}' "
            f"(evicted {len(evicted_turns)} turns)."
        )
        return summary

    def schedule_compaction(
        self,
        session_id: str,
        turns: List[TurnMessage],
        summary_holder: Dict[str, str],
    ) -> Tuple[List[TurnMessage], Optional[asyncio.Task[str]]]:
        """Apply sliding window truncation and schedule non-blocking background summarization if needed.

        Args:
            session_id: Unique session identifier.
            turns: Full list of conversation turns.
            summary_holder: Mutable dict holding `'summary'` updated asynchronously by the background task.

        Returns:
            Tuple of `(retained_recent_turns, optional_background_asyncio_task)`.
        """
        if not self.needs_compaction(turns):
            return list(turns), None

        evicted_turns = turns[: -self.sliding_window_turns]
        retained_turns = list(turns[-self.sliding_window_turns :])
        existing_summary = summary_holder.get("summary", "")

        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(
                self._async_compact_worker(
                    session_id=session_id,
                    evicted_turns=evicted_turns,
                    existing_summary=existing_summary,
                    target_container=summary_holder,
                )
            )
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
            return retained_turns, task
        except RuntimeError:
            # No running event loop; perform immediate synchronous fallback while keeping API consistent
            summary_holder["summary"] = self.summarize_turns_sync(
                evicted_turns, existing_summary=existing_summary
            )
            return retained_turns, None

    async def wait_for_background_tasks(self) -> None:
        """Await all pending background compaction tasks (useful in tests and graceful shutdowns)."""
        if self._background_tasks:
            await asyncio.gather(*list(self._background_tasks), return_exceptions=True)
