"""LLM provider protocol and exception types."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


class LLMError(Exception):
    """Base class for LLM errors. Always carries a `raw_response` if available."""


class LLMAuthError(LLMError):
    """401 / 403 from the LLM endpoint."""


class LLMRateLimitError(LLMError):
    """429 from the LLM endpoint."""


class LLMParseError(LLMError):
    """The LLM returned a response that didn't match the expected schema."""


@runtime_checkable
class LLMClient(Protocol):
    """A minimal LLM client. Both OllamaClient and OpenAICompatibleClient implement this."""

    def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.2,
        json_mode: bool = True,
        max_tokens: int = 4096,
    ) -> str: ...

    def health(self) -> bool: ...
