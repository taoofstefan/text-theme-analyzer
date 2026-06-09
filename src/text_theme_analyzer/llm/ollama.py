"""Ollama (Pro) LLM client. Same wire shape as OpenAI-compatible."""

from text_theme_analyzer.llm.openai_compat import OpenAICompatibleClient


class OllamaClient(OpenAICompatibleClient):
    """Ollama Pro / cloud. Identical to OpenAICompatibleClient — the /v1/chat/completions
    endpoint is exposed with the same body shape, so we just inherit.

    Default base URL points at Ollama Pro cloud. Override with TEXTHEME_OLLAMA_BASE_URL
    for self-hosted Ollama.
    """

    DEFAULT_BASE_URL = "https://api.ollama.com"
