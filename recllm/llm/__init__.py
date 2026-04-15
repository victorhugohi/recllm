"""LLM backend abstraction layer."""

from recllm.llm.base import LLMClient
from recllm.llm.ollama import OllamaClient

__all__ = ["LLMClient", "OllamaClient"]
