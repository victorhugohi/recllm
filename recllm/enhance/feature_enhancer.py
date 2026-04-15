"""LLM-as-Feature-Enhancer pattern (RLMRec, KAR).

Uses an LLM to generate enriched textual features for users and items,
then converts them to dense embeddings that augment collaborative signals.
Implements content-addressed disk caching (ADR-004) to avoid redundant calls.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from recllm.data.base import InteractionData
from recllm.llm.base import LLMClient

logger = logging.getLogger(__name__)


def _content_hash(text: str) -> str:
    """Compute content-addressed hash for caching."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class FeatureEnhancer:
    """LLM-based feature enhancement for recommendation data.

    Generates enriched textual descriptions for users and items using
    prompt templates, then produces dense embedding vectors via the
    LLM's embedding endpoint. Results are cached to disk (ADR-004)
    to avoid repeated LLM calls across experiments.

    Args:
        llm: LLM client instance (e.g., OllamaClient).
        cache_dir: Directory for disk-based feature cache.
        user_prompt_template: Template for generating user profiles.
            Must contain {user_id} and optionally {interactions}.
        item_prompt_template: Template for generating item descriptions.
            Must contain {item_id} and optionally {features}.
        batch_size: Number of prompts to send per LLM batch call.

    Example:
        >>> from recllm.llm.ollama import OllamaClient
        >>> llm = OllamaClient(model="mistral:7b")
        >>> enhancer = FeatureEnhancer(llm)
        >>> enhanced = enhancer.enhance_items(data, feature_col="title")
        >>> # enhanced.item_features now includes LLM embeddings
    """

    DEFAULT_USER_TEMPLATE = (
        "Based on this user's interaction history, generate a brief profile "
        "describing their preferences and interests.\n\n"
        "User ID: {user_id}\n"
        "Items interacted with: {interactions}\n\n"
        "User profile:"
    )

    DEFAULT_ITEM_TEMPLATE = (
        "Generate a rich, descriptive summary of this item for a "
        "recommendation system.\n\n"
        "Item ID: {item_id}\n"
        "Item information: {features}\n\n"
        "Description:"
    )

    def __init__(
        self,
        llm: LLMClient,
        cache_dir: str | Path = "llm_cache",
        user_prompt_template: str | None = None,
        item_prompt_template: str | None = None,
        batch_size: int = 8,
    ):
        self.llm = llm
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.user_prompt_template = user_prompt_template or self.DEFAULT_USER_TEMPLATE
        self.item_prompt_template = item_prompt_template or self.DEFAULT_ITEM_TEMPLATE
        self.batch_size = batch_size

        # Sub-directories for different cache types
        self._text_cache_dir = self.cache_dir / "texts"
        self._embed_cache_dir = self.cache_dir / "embeddings"
        self._text_cache_dir.mkdir(exist_ok=True)
        self._embed_cache_dir.mkdir(exist_ok=True)

    def _get_cached_text(self, key: str) -> str | None:
        """Retrieve cached text generation result."""
        path = self._text_cache_dir / f"{key}.json"
        if path.exists():
            data = json.loads(path.read_text())
            return data.get("text")
        return None

    def _set_cached_text(self, key: str, prompt: str, text: str) -> None:
        """Cache a text generation result."""
        path = self._text_cache_dir / f"{key}.json"
        path.write_text(json.dumps({
            "prompt_hash": _content_hash(prompt),
            "text": text,
            "model": self.llm.model_name,
        }))

    def _get_cached_embedding(self, key: str) -> np.ndarray | None:
        """Retrieve cached embedding."""
        path = self._embed_cache_dir / f"{key}.npy"
        if path.exists():
            return np.load(path)
        return None

    def _set_cached_embedding(self, key: str, embedding: np.ndarray) -> None:
        """Cache an embedding vector."""
        path = self._embed_cache_dir / f"{key}.npy"
        np.save(path, embedding)

    def enhance_items(
        self,
        data: InteractionData,
        feature_col: str | None = None,
        embed: bool = True,
    ) -> EnhancedFeatures:
        """Generate LLM-enhanced features for items.

        For each unique item, generates a textual description using the
        item prompt template, optionally converts to embeddings.

        Args:
            data: Interaction data (uses item_features if available).
            feature_col: Column name in item_features to use as item info.
                If None, uses item_id only.
            embed: Whether to also generate embedding vectors.

        Returns:
            EnhancedFeatures with texts and optionally embeddings.
        """
        item_ids = data.item_ids
        texts: dict[int, str] = {}
        prompts_to_generate: list[tuple[int, str]] = []

        for item_id in item_ids:
            cache_key = f"item_{item_id}_{self.llm.model_name}"

            cached = self._get_cached_text(cache_key)
            if cached is not None:
                texts[int(item_id)] = cached
                continue

            # Build feature string
            features = str(item_id)
            if data.item_features is not None and feature_col:
                row = data.item_features.filter(
                    data.item_features["item_id"] == item_id
                )
                if len(row) > 0 and feature_col in row.columns:
                    features = str(row[feature_col][0])

            prompt = self.item_prompt_template.format(
                item_id=item_id, features=features
            )
            prompts_to_generate.append((int(item_id), prompt))

        # Generate in batches
        if prompts_to_generate:
            logger.info(
                "Generating LLM descriptions for %d items (%d cached)",
                len(prompts_to_generate),
                len(texts),
            )
            for batch_start in range(0, len(prompts_to_generate), self.batch_size):
                batch = prompts_to_generate[batch_start : batch_start + self.batch_size]
                batch_prompts = [p for _, p in batch]
                responses = self.llm.generate_batch(batch_prompts)

                for (iid, prompt), response in zip(batch, responses, strict=False):
                    texts[iid] = response
                    cache_key = f"item_{iid}_{self.llm.model_name}"
                    self._set_cached_text(cache_key, prompt, response)

        # Generate embeddings if requested
        embeddings = None
        if embed:
            embeddings = self._get_embeddings(
                {iid: text for iid, text in texts.items()},
                prefix="item",
            )

        return EnhancedFeatures(
            entity_type="item",
            texts=texts,
            embeddings=embeddings,
            model_name=self.llm.model_name,
        )

    def enhance_users(
        self,
        data: InteractionData,
        max_history: int = 20,
        embed: bool = True,
    ) -> EnhancedFeatures:
        """Generate LLM-enhanced features for users.

        For each unique user, builds a prompt from their interaction
        history and generates a textual user profile.

        Args:
            data: Interaction data.
            max_history: Maximum number of interactions to include in prompt.
            embed: Whether to also generate embedding vectors.

        Returns:
            EnhancedFeatures with texts and optionally embeddings.
        """
        user_ids = data.user_ids
        arrays = data.to_numpy()

        # Build user histories
        user_items: dict[int, list[int]] = {}
        for u, i in zip(arrays["user_id"], arrays["item_id"], strict=False):
            user_items.setdefault(int(u), []).append(int(i))

        texts: dict[int, str] = {}
        prompts_to_generate: list[tuple[int, str]] = []

        for user_id in user_ids:
            cache_key = f"user_{user_id}_{self.llm.model_name}"

            cached = self._get_cached_text(cache_key)
            if cached is not None:
                texts[int(user_id)] = cached
                continue

            history = user_items.get(int(user_id), [])[:max_history]
            prompt = self.user_prompt_template.format(
                user_id=user_id, interactions=", ".join(str(i) for i in history)
            )
            prompts_to_generate.append((int(user_id), prompt))

        if prompts_to_generate:
            logger.info(
                "Generating LLM profiles for %d users (%d cached)",
                len(prompts_to_generate),
                len(texts),
            )
            for batch_start in range(0, len(prompts_to_generate), self.batch_size):
                batch = prompts_to_generate[batch_start : batch_start + self.batch_size]
                batch_prompts = [p for _, p in batch]
                responses = self.llm.generate_batch(batch_prompts)

                for (uid, prompt), response in zip(batch, responses, strict=False):
                    texts[uid] = response
                    cache_key = f"user_{uid}_{self.llm.model_name}"
                    self._set_cached_text(cache_key, prompt, response)

        embeddings = None
        if embed:
            embeddings = self._get_embeddings(
                {uid: text for uid, text in texts.items()},
                prefix="user",
            )

        return EnhancedFeatures(
            entity_type="user",
            texts=texts,
            embeddings=embeddings,
            model_name=self.llm.model_name,
        )

    def _get_embeddings(
        self,
        texts: dict[int, str],
        prefix: str,
    ) -> dict[int, np.ndarray]:
        """Generate or retrieve cached embeddings for text dict."""
        embeddings: dict[int, np.ndarray] = {}
        to_embed: list[tuple[int, str]] = []

        for entity_id, text in texts.items():
            cache_key = f"{prefix}_{entity_id}_{self.llm.model_name}_emb"
            cached = self._get_cached_embedding(cache_key)
            if cached is not None:
                embeddings[entity_id] = cached
            else:
                to_embed.append((entity_id, text))

        if to_embed:
            logger.info(
                "Generating embeddings for %d %ss (%d cached)",
                len(to_embed),
                prefix,
                len(embeddings),
            )
            batch_texts = [text for _, text in to_embed]
            batch_embeddings = self.llm.embed(batch_texts)

            for (entity_id, _), emb in zip(to_embed, batch_embeddings, strict=False):
                embeddings[entity_id] = emb
                cache_key = f"{prefix}_{entity_id}_{self.llm.model_name}_emb"
                self._set_cached_embedding(cache_key, emb)

        return embeddings


class EnhancedFeatures:
    """Container for LLM-generated features.

    Holds both textual descriptions and dense embeddings for a set
    of entities (users or items).

    Attributes:
        entity_type: "user" or "item".
        texts: Mapping from entity ID to generated text.
        embeddings: Mapping from entity ID to embedding vector.
        model_name: LLM model used for generation.
    """

    def __init__(
        self,
        entity_type: str,
        texts: dict[int, str],
        embeddings: dict[int, np.ndarray] | None = None,
        model_name: str = "",
    ):
        self.entity_type = entity_type
        self.texts = texts
        self.embeddings = embeddings
        self.model_name = model_name

    @property
    def n_entities(self) -> int:
        return len(self.texts)

    @property
    def embedding_dim(self) -> int | None:
        if self.embeddings:
            first = next(iter(self.embeddings.values()))
            return len(first)
        return None

    def to_numpy(self, entity_ids: list[int] | None = None) -> np.ndarray:
        """Convert embeddings to a 2D array.

        Args:
            entity_ids: Ordered list of IDs. If None, uses sorted keys.

        Returns:
            Array of shape (n_entities, embedding_dim).
        """
        if self.embeddings is None:
            raise ValueError("No embeddings available. Call enhance with embed=True.")
        if entity_ids is None:
            entity_ids = sorted(self.embeddings.keys())
        return np.stack([self.embeddings[eid] for eid in entity_ids])

    def __repr__(self) -> str:
        emb_info = f", embedding_dim={self.embedding_dim}" if self.embeddings else ""
        return (
            f"EnhancedFeatures(type={self.entity_type!r}, "
            f"n={self.n_entities}{emb_info}, model={self.model_name!r})"
        )
