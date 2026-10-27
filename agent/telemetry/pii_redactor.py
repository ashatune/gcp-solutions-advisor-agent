"""Active customer PII and secret scrubbing pipeline before logging or memory persistence."""

import re
from typing import Any, Dict, List, Pattern, Tuple, Union


class PIIRedactor:
    """Scrubs customer PII, GCP project IDs, IPv4/IPv6 addresses, emails, and API secrets."""

    def __init__(self) -> None:
        self._patterns: List[Tuple[Pattern[str], str]] = [
            # Google API Keys (AIza...)
            (re.compile(r"\bAIza[0-9A-Za-z\-_]{20,50}\b"), "[REDACTED_GCP_API_KEY]"),
            # OAuth / Bearer tokens / Private key headers
            (
                re.compile(r"ya29\.[0-9A-Za-z\-_]+"),
                "[REDACTED_OAUTH_TOKEN]",
            ),
            (
                re.compile(
                    r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?[A-Za-z0-9\-_\.]{12,}['\"]?"
                ),
                r"\1=[REDACTED_SECRET]",
            ),
            # Email addresses
            (
                re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
                "[REDACTED_EMAIL]",
            ),
            # IPv4 addresses
            (
                re.compile(
                    r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"
                ),
                "[REDACTED_IP]",
            ),
            # GCP Project IDs in explicit context (e.g., project_id=my-prod-proj-123, projects/my-prod-123)
            (
                re.compile(
                    r"(?i)(?:projects/|project[_-]?id\s*[:=]\s*['\"]?|project\s+['\"])([a-z][a-z0-9\-]{4,28}[a-z0-9])['\"]?"
                ),
                "[REDACTED_GCP_PROJECT_ID]",
            ),
            # US Phone numbers
            (
                re.compile(
                    r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
                ),
                "[REDACTED_PHONE]",
            ),
        ]

    def scrub_text(self, text: str) -> str:
        """Redact all recognized PII and secret patterns from a string.

        Args:
            text: Raw input string potentially containing sensitive customer data.

        Returns:
            Sanitized string with sensitive tokens replaced by redaction placeholders.
        """
        if not isinstance(text, str) or not text:
            return text
        sanitized = text
        for pattern, replacement in self._patterns:
            sanitized = pattern.sub(replacement, sanitized)
        return sanitized

    def scrub_payload(self, payload: Any) -> Any:
        """Recursively scrub dictionaries, lists, and primitive values.

        Args:
            payload: Arbitrary structure (dict, list, str, int, etc.) to sanitize.

        Returns:
            Deeply sanitized structure safe for logs and persistent storage.
        """
        if isinstance(payload, str):
            return self.scrub_text(payload)
        if isinstance(payload, dict):
            sanitized_dict: Dict[str, Any] = {}
            for key, value in payload.items():
                lower_key = str(key).lower()
                if any(
                    sensitive in lower_key
                    for sensitive in ("api_key", "secret", "password", "auth_token", "private_key")
                ):
                    sanitized_dict[key] = "[REDACTED_SECRET]"
                elif lower_key in ("project_id", "gcp_project_id"):
                    sanitized_dict[key] = "[REDACTED_GCP_PROJECT_ID]"
                else:
                    sanitized_dict[key] = self.scrub_payload(value)
            return sanitized_dict
        if isinstance(payload, list):
            return [self.scrub_payload(item) for item in payload]
        if isinstance(payload, tuple):
            return tuple(self.scrub_payload(item) for item in payload)
        return payload


_DEFAULT_REDACTOR = PIIRedactor()


def redact_pii(data: Union[str, Dict[str, Any], List[Any], Any]) -> Any:
    """Convenience function to scrub strings or structured payloads of PII and secrets."""
    return _DEFAULT_REDACTOR.scrub_payload(data)
