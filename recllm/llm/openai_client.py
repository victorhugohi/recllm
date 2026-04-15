"""OpenAI-compatible LLM client.

Works with OpenAI API, Azure OpenAI, and any OpenAI-compatible endpoint
(vLLM, Together AI, Groq, etc.).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from recllm.llm.base import LLMClient


class OpenAIClient(LLMClient):
    """LLM client for OpenAI and compatible APIs.

    Requires the `openai` package: pip install recllm[cloud]

    Args:
        model: Model name (e.g., "gpt-4o-mini", "gpt-3.5-turbo").
        api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
        base_url: API base URL. Override for Azure or compatible endpoints.
        embedding_model: Model for embeddings (default: "text-embedding-3-small").
        max_tokens: Maximum tokens for generation.
        temperature: Sampling temperature.

    Example:
        >>> llm = OpenAIClient(model="gpt-4o-mini")
        >>> response = llm.generate("Describe this movie: Inception")
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        embedding_model: str = "text-embedding-3-small",
        max_tokens: int = 512,
        temperature: float = 0.7,
    ):
        try:
            import openai
        except ImportError as err:
            raise ImportError(
                "OpenAI client requires the 'openai' package. "
                "Install with: pip install recllm[cloud]"
            ) from err

        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url

        self._client = openai.OpenAI(**kwargs)
        self._model = model
        self._embedding_model = embedding_model
        self.max_tokens = max_tokens
        self.temperature = temperature

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using the OpenAI chat completions API."""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
        )
        return response.choices[0].message.content or ""

    def generate_batch(
        self,
        prompts: list[str],
        max_workers: int = 4,
        **kwargs,
    ) -> list[str]:
        """Generate text for multiple prompts concurrently."""
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(self.generate, p, **kwargs) for p in prompts]
            return [f.result() for f in futures]

    def embed(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings using the OpenAI embeddings API."""
        response = self._client.embeddings.create(
            model=self._embedding_model,
            input=texts,
        )
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings, dtype=np.float32)
