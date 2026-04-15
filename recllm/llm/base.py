"""Abstract base class for LLM backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class LLMClient(ABC):
    """Abstract base class for LLM backend clients.

    All LLM backends (Ollama, llama.cpp, OpenAI, Anthropic) implement
    this interface, enabling backend-agnostic recommendation pipelines.
    """

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a single prompt.

        Args:
            prompt: Input prompt text.
            **kwargs: Backend-specific generation parameters
                (temperature, max_tokens, etc.).

        Returns:
            Generated text response.
        """
        ...

    def generate_batch(
        self, prompts: list[str], **kwargs
    ) -> list[str]:
        """Generate text for multiple prompts.

        Default implementation calls generate() sequentially.
        Backends may override for true batching or concurrent execution.

        Args:
            prompts: List of input prompts.
            **kwargs: Backend-specific generation parameters.

        Returns:
            List of generated text responses.
        """
        return [self.generate(p, **kwargs) for p in prompts]

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Generate embedding vectors for text inputs.

        Args:
            texts: List of text strings to embed.

        Returns:
            Array of shape (len(texts), embedding_dim).
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name!r})"
