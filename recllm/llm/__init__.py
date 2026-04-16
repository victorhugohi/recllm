"""LLM backend abstraction layer."""

from recllm.llm.base import LLMClient
from recllm.llm.ollama import OllamaClient

__all__ = ["LLMClient", "OllamaClient"]


# Lazy imports for optional backends
def __getattr__(name: str):
    if name == "OpenAIClient":
        from recllm.llm.openai_client import OpenAIClient
        return OpenAIClient
    if name == "LlamaCppClient":
        from recllm.llm.llamacpp import LlamaCppClient
        return LlamaCppClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
