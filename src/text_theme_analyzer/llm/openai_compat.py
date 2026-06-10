"""OpenAI-compatible chat completions client (Ollama Pro + swap targets)."""

from __future__ import annotations

import json
from typing import Any

import httpx

from text_theme_analyzer.llm.base import LLMAuthError, LLMError, LLMRateLimitError


def _post_chat(
    *,
    base_url: str,
    api_key: str,
    model: str,
    system: str,
    user: str,
    temperature: float,
    json_mode: bool,
    max_tokens: int,
    timeout_s: float,
) -> str:
    base = base_url.rstrip("/")
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        r = httpx.post(
            f"{base}/v1/chat/completions",
            json=body,
            headers=headers,
            timeout=timeout_s,
        )
    except httpx.HTTPError as e:
        raise LLMError(f"HTTP error: {e}") from e
    if r.status_code in (401, 403):
        raise LLMAuthError(f"Auth failed ({r.status_code}): {r.text[:200]}")
    if r.status_code == 429:
        raise LLMRateLimitError(f"Rate limited: {r.text[:200]}")
    if r.status_code >= 400:
        raise LLMError(f"LLM call failed ({r.status_code}): {r.text[:300]}")
    try:
        data = r.json()
    except json.JSONDecodeError as e:
        raise LLMError(f"Non-JSON response from LLM: {r.text[:200]}") from e
    try:
        return str(data["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as e:
        raise LLMError(f"Unexpected response shape: {data}") from e


class OpenAICompatibleClient:
    """Any OpenAI-shape /v1/chat/completions endpoint: OpenAI, Groq, Together, …"""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_s: float = 120.0,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s

    def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.2,
        json_mode: bool = True,
        max_tokens: int = 4096,
    ) -> str:
        return _post_chat(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            system=system,
            user=user,
            temperature=temperature,
            json_mode=json_mode,
            max_tokens=max_tokens,
            timeout_s=self.timeout_s,
        )

    def health(self) -> bool:
        try:
            r = httpx.get(
                f"{self.base_url.rstrip('/')}/v1/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10.0,
            )
            return r.status_code == 200
        except httpx.HTTPError:
            return False
