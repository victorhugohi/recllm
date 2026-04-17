"""LLM-as-Explainer pattern.

Generates natural language explanations for recommendations,
improving transparency and user trust. Uses the LLM to articulate
why a particular item was recommended based on user history and
item characteristics.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from recllm.llm.base import LLMClient

logger = logging.getLogger(__name__)


class LLMExplainer:
    """LLM-based recommendation explainer.

    Generates human-readable explanations for why items were
    recommended to a user. Supports multiple explanation styles
    and caches results for efficiency.

    Args:
        llm: LLM client instance.
        style: Explanation style — "conversational", "analytical", or "brief".
        cache_dir: Directory for caching explanations.

    Example:
        >>> explainer = LLMExplainer(llm, style="conversational")
        >>> explanation = explainer.explain(
        ...     user_history=["The Matrix", "Inception", "Interstellar"],
        ...     recommended_item="Tenet",
        ...     score=0.92,
        ... )
        >>> print(explanation)
        "Based on your interest in mind-bending sci-fi films like
        The Matrix and Inception, we think you'll enjoy Tenet..."
    """

    CONVERSATIONAL_TEMPLATE = (
        "You are a recommendation system explaining why an item was suggested.\n\n"
        "The user has enjoyed: {history}\n"
        "We recommended: {item}\n"
        "Relevance score: {score:.0%}\n\n"
        "Write a friendly 2-3 sentence explanation of why this item "
        "matches the user's tastes. Be specific about connections."
    )

    ANALYTICAL_TEMPLATE = (
        "Analyze the recommendation of '{item}' for a user who "
        "previously engaged with: {history}\n\n"
        "Respond in exactly three short sentences, one per point, "
        "no headers or bullets:\n"
        "1. Key preference pattern (one sentence).\n"
        "2. How the item matches that pattern (one sentence).\n"
        "3. Confidence with justification (one sentence).\n\n"
        "Relevance score: {score:.0%}. "
        "Keep the full response under 120 words."
    )

    BRIEF_TEMPLATE = (
        "User likes: {history}\n"
        "Recommended: {item} (score: {score:.0%})\n\n"
        "Explain in one sentence why this recommendation makes sense:"
    )

    def __init__(
        self,
        llm: LLMClient,
        style: str = "conversational",
        cache_dir: str | Path | None = None,
    ):
        if style not in ("conversational", "analytical", "brief"):
            raise ValueError(f"Unknown style: {style}")

        self.llm = llm
        self.style = style

        if style == "conversational":
            self.template = self.CONVERSATIONAL_TEMPLATE
        elif style == "analytical":
            self.template = self.ANALYTICAL_TEMPLATE
        else:
            self.template = self.BRIEF_TEMPLATE

        self._cache: dict[str, str] = {}
        self._cache_dir = Path(cache_dir) if cache_dir else None
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    def explain(
        self,
        user_history: list[str],
        recommended_item: str,
        score: float = 0.0,
        custom_context: str | None = None,
    ) -> str:
        """Generate an explanation for a single recommendation.

        Args:
            user_history: List of item descriptions the user liked.
            recommended_item: Description of the recommended item.
            score: Relevance score (0-1).
            custom_context: Additional context to include in prompt.

        Returns:
            Natural language explanation string.
        """
        cache_key = self._make_cache_key(user_history, recommended_item, score)

        if cache_key in self._cache:
            return self._cache[cache_key]

        if self._cache_dir:
            cached = self._load_from_disk(cache_key)
            if cached:
                self._cache[cache_key] = cached
                return cached

        history_str = ", ".join(user_history[-10:])  # Limit for prompt length
        prompt = self.template.format(
            history=history_str,
            item=recommended_item,
            score=score,
        )
        if custom_context:
            prompt += f"\n\nAdditional context: {custom_context}"

        explanation = self.llm.generate(prompt)

        self._cache[cache_key] = explanation
        if self._cache_dir:
            self._save_to_disk(cache_key, explanation)

        return explanation

    def explain_batch(
        self,
        user_history: list[str],
        recommended_items: list[str],
        scores: list[float] | None = None,
    ) -> list[str]:
        """Generate explanations for multiple recommendations.

        Args:
            user_history: User's interaction history.
            recommended_items: List of recommended item descriptions.
            scores: Optional list of relevance scores.

        Returns:
            List of explanation strings.
        """
        if scores is None:
            scores = [0.0] * len(recommended_items)

        # Check cache first
        results: list[str | None] = [None] * len(recommended_items)
        to_generate: list[tuple[int, str]] = []

        history_str = ", ".join(user_history[-10:])

        for idx, (item, score) in enumerate(zip(recommended_items, scores, strict=False)):
            cache_key = self._make_cache_key(user_history, item, score)
            cached = self._cache.get(cache_key)
            if cached is None and self._cache_dir:
                cached = self._load_from_disk(cache_key)
            if cached:
                results[idx] = cached
                self._cache[cache_key] = cached
            else:
                prompt = self.template.format(
                    history=history_str, item=item, score=score
                )
                to_generate.append((idx, prompt))

        if to_generate:
            prompts = [p for _, p in to_generate]
            responses = self.llm.generate_batch(prompts)

            for (idx, _prompt), response in zip(to_generate, responses, strict=False):
                item = recommended_items[idx]
                score = scores[idx]
                cache_key = self._make_cache_key(user_history, item, score)

                results[idx] = response
                self._cache[cache_key] = response
                if self._cache_dir:
                    self._save_to_disk(cache_key, response)

        return [r or "" for r in results]

    def _make_cache_key(
        self, history: list[str], item: str, score: float
    ) -> str:
        content = json.dumps({
            "history": sorted(history),
            "item": item,
            "score": round(score, 2),
            "style": self.style,
            "model": self.llm.model_name,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _load_from_disk(self, key: str) -> str | None:
        if not self._cache_dir:
            return None
        path = self._cache_dir / f"expl_{key}.json"
        if path.exists():
            data = json.loads(path.read_text())
            return data.get("explanation")
        return None

    def _save_to_disk(self, key: str, explanation: str) -> None:
        if not self._cache_dir:
            return
        path = self._cache_dir / f"expl_{key}.json"
        path.write_text(json.dumps({
            "explanation": explanation,
            "model": self.llm.model_name,
            "style": self.style,
        }))

    def __repr__(self) -> str:
        return (
            f"LLMExplainer(style={self.style!r}, "
            f"model={self.llm.model_name!r})"
        )
