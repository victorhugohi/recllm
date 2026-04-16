"""llama.cpp LLM backend client for direct GGUF inference.

Provides local LLM inference without needing an external server like Ollama.
Loads GGUF model files directly via llama-cpp-python bindings.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from recllm.llm.base import LLMClient


class LlamaCppClient(LLMClient):
    """LLM client using llama-cpp-python for direct GGUF model inference.

    Loads GGUF quantized models directly, without requiring Ollama or
    any external server. Useful for maximum control over inference
    parameters and for environments where Ollama cannot be installed.

    Requires the `llama-cpp-python` package: pip install recllm[llamacpp]

    Args:
        model_path: Path to a GGUF model file.
        n_ctx: Context window size in tokens.
        n_gpu_layers: Number of layers to offload to GPU (-1 = all).
        n_threads: Number of CPU threads (None = auto-detect).
        verbose: Whether to print llama.cpp logs.

    Example:
        >>> llm = LlamaCppClient("models/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
        >>> response = llm.generate("Describe this movie: The Matrix")
    """

    def __init__(
        self,
        model_path: str | Path,
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        n_threads: int | None = None,
        verbose: bool = False,
    ):
        try:
            from llama_cpp import Llama
        except ImportError as err:
            raise ImportError(
                "LlamaCppClient requires the 'llama-cpp-python' package. "
                "Install with: pip install recllm[llamacpp]"
            ) from err

        self._model_path = str(model_path)
        self.n_ctx = n_ctx

        kwargs: dict = {
            "model_path": self._model_path,
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "verbose": verbose,
            "embedding": True,
        }
        if n_threads is not None:
            kwargs["n_threads"] = n_threads

        self._llm = Llama(**kwargs)

    @property
    def model_name(self) -> str:
        return Path(self._model_path).stem

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Input prompt text.
            **kwargs: Generation parameters:
                max_tokens (int): Maximum tokens to generate (default: 512).
                temperature (float): Sampling temperature (default: 0.7).
                top_p (float): Top-p sampling (default: 0.9).
                stop (list[str]): Stop sequences.

        Returns:
            Generated text response.
        """
        output = self._llm(
            prompt,
            max_tokens=kwargs.get("max_tokens", 512),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
            stop=kwargs.get("stop"),
            echo=False,
        )
        return output["choices"][0]["text"].strip()

    def embed(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for text inputs.

        Args:
            texts: List of text strings to embed.

        Returns:
            Array of shape (len(texts), embedding_dim).
        """
        embeddings = []
        for text in texts:
            emb = self._llm.embed(text)
            embeddings.append(emb)
        return np.array(embeddings, dtype=np.float32)

    def unload(self) -> None:
        """Free the model from memory."""
        if hasattr(self, "_llm"):
            del self._llm
