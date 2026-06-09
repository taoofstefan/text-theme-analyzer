"""Build an LLMClient from a Config + env vars."""

from __future__ import annotations

import os

from text_theme_analyzer.config import Config, Provider
from text_theme_analyzer.llm.base import LLMClient, LLMError
from text_theme_analyzer.llm.ollama import OllamaClient
from text_theme_analyzer.llm.openai_compat import OpenAICompatibleClient


def build_client(config: Config) -> LLMClient:
    if config.provider == Provider.OLLAMA:
        api_key = os.environ.get(config.ollama.api_key_env, "")
        if not api_key:
            raise LLMError(
                f"No Ollama API key set. Set {config.ollama.api_key_env} in your env or .env file."
            )
        return OllamaClient(
            base_url=config.ollama.base_url,
            api_key=api_key,
            model=config.model,
            timeout_s=config.ollama.timeout_s,
        )
    if config.provider == Provider.OPENAI_COMPAT:
        base_url = os.environ.get(config.openai_compat.base_url_env, "")
        api_key = os.environ.get(config.openai_compat.api_key_env, "")
        model = os.environ.get(config.openai_compat.model_env, config.model)
        if not base_url or not api_key:
            raise LLMError(
                f"For --provider openai_compat you need both "
                f"{config.openai_compat.base_url_env} and "
                f"{config.openai_compat.api_key_env} set in env."
            )
        return OpenAICompatibleClient(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_s=120.0,
        )
    raise LLMError(f"Unknown provider: {config.provider}")
