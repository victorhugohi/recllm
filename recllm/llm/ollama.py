"""Ollama LLM backend client (ADR-002: primary local inference)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

from recllm.llm.base import LLMClient


class OllamaClient(LLMClient):
    """LLM client for Ollama local inference.

    Communicates with the Ollama HTTP API for text generation and
    embeddings. Ollama must be installed and running separately.

    Args:
        model: Ollama model name (e.g., "mistral:7b", "llama3.1:8b").
        base_url: Ollama API base URL.
        timeout: Request timeout in seconds.
        num_ctx: Context window size in tokens.

    Example:
        >>> llm = OllamaClient(model="mistral:7b")
        >>> response = llm.generate("Describe this movie: The Matrix")
        >>> embeddings = llm.embed(["action movie", "romantic comedy"])
    """

    def __init__(
        self,
        model: str = "mistral:7b",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
        num_ctx: int = 4096,
    ):
        self._model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.num_ctx = num_ctx

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Ollama API.

        Args:
            prompt: Input prompt.
            **kwargs: Additional options passed to Ollama
                (temperature, top_p, etc.).

        Returns:
            Generated text.

        Raises:
            ConnectionError: If Ollama is not running.
            requests.HTTPError: If the API returns an error.
        """
        options = {"num_ctx": self.num_ctx}
        options.update(kwargs.pop("options", {}))

        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        payload.update(kwargs)

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()["response"]
        except requests.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Is Ollama running? Start it with: ollama serve"
            )

    def generate_batch(
        self,
        prompts: list[str],
        max_workers: int = 4,
        **kwargs,
    ) -> list[str]:
        """Generate text for multiple prompts using concurrent requests.

        Args:
            prompts: List of input prompts.
            max_workers: Number of concurrent threads.
            **kwargs: Passed to generate().

        Returns:
            List of generated texts.
        """
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(self.generate, p, **kwargs) for p in prompts]
            return [f.result() for f in futures]

    def embed(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings using Ollama API.

        Args:
            texts: List of texts to embed.

        Returns:
            Array of shape (len(texts), embedding_dim).
        """
        embeddings = []
        for text in texts:
            try:
                response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self._model, "prompt": text},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                embeddings.append(response.json()["embedding"])
            except requests.ConnectionError:
                raise ConnectionError(
                    f"Cannot connect to Ollama at {self.base_url}. "
                    "Is Ollama running? Start it with: ollama serve"
                )
        return np.array(embeddings, dtype=np.float32)

    def is_available(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5,
            )
            if response.status_code != 200:
                return False
            models = [m["name"] for m in response.json().get("models", [])]
            return self._model in models or self._model.split(":")[0] in [
                m.split(":")[0] for m in models
            ]
        except (requests.ConnectionError, requests.Timeout):
            return False

    def unload(self) -> None:
        """Unload the model from GPU memory (ADR-005 support)."""
        try:
            requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self._model, "keep_alive": 0},
                timeout=10,
            )
        except requests.ConnectionError:
            pass
