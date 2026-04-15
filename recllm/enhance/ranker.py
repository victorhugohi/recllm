"""LLM-as-Ranker pattern (TALLRec, Hou et al. 2024).

Uses an LLM to re-rank a candidate list of items for a given user,
leveraging natural language understanding of item descriptions and
user preferences. Supports pointwise, pairwise, and listwise ranking.
"""

from __future__ import annotations

import logging
import re

import numpy as np

from recllm.llm.base import LLMClient

logger = logging.getLogger(__name__)


class LLMRanker:
    """LLM-based re-ranker for recommendation candidates.

    Takes a candidate set (e.g., from a base model's top-100) and
    uses an LLM to re-rank them based on natural language reasoning
    about user preferences and item descriptions.

    Supports three ranking modes following the literature:
    - **pointwise**: Score each item independently (simplest, most parallel).
    - **pairwise**: Compare pairs and aggregate (more accurate, O(n^2)).
    - **listwise**: Present full list to LLM for ordering (Hou et al. 2024).

    Args:
        llm: LLM client instance.
        mode: Ranking mode — "pointwise", "pairwise", or "listwise".
        prompt_template: Custom prompt template. If None, uses mode default.
        max_candidates: Maximum candidates to re-rank per user.

    Example:
        >>> ranker = LLMRanker(llm, mode="listwise")
        >>> reranked = ranker.rerank(
        ...     user_history=["The Matrix", "Inception"],
        ...     candidates=["Tenet", "Titanic", "Interstellar", "The Notebook"],
        ... )
    """

    POINTWISE_TEMPLATE = (
        "On a scale of 1-10, how likely would a user who enjoyed these items "
        "enjoy the candidate item?\n\n"
        "User's liked items: {history}\n"
        "Candidate item: {candidate}\n\n"
        "Score (just the number):"
    )

    LISTWISE_TEMPLATE = (
        "A user has enjoyed these items: {history}\n\n"
        "Please rank the following candidate items from most to least "
        "relevant for this user. Return ONLY the numbers in order, "
        "separated by commas (e.g., '3,1,4,2').\n\n"
        "Candidates:\n{candidates_list}\n\n"
        "Ranking:"
    )

    PAIRWISE_TEMPLATE = (
        "A user has enjoyed: {history}\n\n"
        "Which item would this user prefer?\n"
        "A: {item_a}\n"
        "B: {item_b}\n\n"
        "Answer with just 'A' or 'B':"
    )

    def __init__(
        self,
        llm: LLMClient,
        mode: str = "listwise",
        prompt_template: str | None = None,
        max_candidates: int = 20,
    ):
        if mode not in ("pointwise", "pairwise", "listwise"):
            raise ValueError(f"Unknown mode: {mode}. Use pointwise, pairwise, or listwise.")
        self.llm = llm
        self.mode = mode
        self.max_candidates = max_candidates

        if prompt_template:
            self.prompt_template = prompt_template
        elif mode == "pointwise":
            self.prompt_template = self.POINTWISE_TEMPLATE
        elif mode == "pairwise":
            self.prompt_template = self.PAIRWISE_TEMPLATE
        else:
            self.prompt_template = self.LISTWISE_TEMPLATE

    def rerank(
        self,
        user_history: list[str],
        candidates: list[str],
        candidate_ids: list[int] | None = None,
    ) -> list[tuple[int | str, float]]:
        """Re-rank candidate items using the LLM.

        Args:
            user_history: List of item descriptions the user has interacted with.
            candidates: List of candidate item descriptions to rank.
            candidate_ids: Optional item IDs corresponding to candidates.
                If None, uses 0-based indices.

        Returns:
            List of (item_id_or_index, score) tuples, sorted by score descending.
        """
        candidates = candidates[: self.max_candidates]
        if candidate_ids:
            candidate_ids = candidate_ids[: self.max_candidates]
        else:
            candidate_ids = list(range(len(candidates)))

        history_str = ", ".join(user_history)

        if self.mode == "pointwise":
            return self._rerank_pointwise(history_str, candidates, candidate_ids)
        elif self.mode == "pairwise":
            return self._rerank_pairwise(history_str, candidates, candidate_ids)
        else:
            return self._rerank_listwise(history_str, candidates, candidate_ids)

    def _rerank_pointwise(
        self, history: str, candidates: list[str], ids: list[int]
    ) -> list[tuple[int, float]]:
        """Score each candidate independently."""
        prompts = [
            self.prompt_template.format(history=history, candidate=c)
            for c in candidates
        ]
        responses = self.llm.generate_batch(prompts)

        scored = []
        for cid, response in zip(ids, responses, strict=False):
            score = self._parse_score(response)
            scored.append((cid, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _rerank_listwise(
        self, history: str, candidates: list[str], ids: list[int]
    ) -> list[tuple[int, float]]:
        """Present all candidates and ask LLM to rank them."""
        candidates_list = "\n".join(
            f"{i + 1}. {c}" for i, c in enumerate(candidates)
        )
        prompt = self.prompt_template.format(
            history=history, candidates_list=candidates_list
        )
        response = self.llm.generate(prompt)

        # Parse ranking from response
        ranking = self._parse_ranking(response, len(candidates))

        # Convert to scored list (higher position = higher score)
        n = len(candidates)
        scored = []
        for rank_pos, candidate_idx in enumerate(ranking):
            if 0 <= candidate_idx < len(ids):
                score = (n - rank_pos) / n  # 1.0 for first, decreasing
                scored.append((ids[candidate_idx], score))

        # Add any candidates not in ranking with score 0
        ranked_ids = {ids[ci] for ci in ranking if 0 <= ci < len(ids)}
        for cid in ids:
            if cid not in ranked_ids:
                scored.append((cid, 0.0))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _rerank_pairwise(
        self, history: str, candidates: list[str], ids: list[int]
    ) -> list[tuple[int, float]]:
        """Compare pairs and count wins."""
        n = len(candidates)
        wins = np.zeros(n, dtype=np.float64)

        # Generate all pair prompts
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((i, j))

        prompts = [
            self.prompt_template.format(
                history=history, item_a=candidates[i], item_b=candidates[j]
            )
            for i, j in pairs
        ]

        responses = self.llm.generate_batch(prompts)

        for (i, j), response in zip(pairs, responses, strict=False):
            response = response.strip().upper()
            if "A" in response and "B" not in response:
                wins[i] += 1
            elif "B" in response:
                wins[j] += 1
            else:
                wins[i] += 0.5
                wins[j] += 0.5

        # Normalize wins to [0, 1]
        max_wins = max(wins.max(), 1.0)
        scored = [(ids[i], float(wins[i] / max_wins)) for i in range(n)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    @staticmethod
    def _parse_score(response: str) -> float:
        """Extract numerical score from LLM response."""
        numbers = re.findall(r"(\d+(?:\.\d+)?)", response)
        if numbers:
            score = float(numbers[0])
            return min(score, 10.0) / 10.0  # Normalize to [0, 1]
        return 0.5  # Default if parsing fails

    @staticmethod
    def _parse_ranking(response: str, n_candidates: int) -> list[int]:
        """Parse a ranking string like '3,1,4,2' into 0-indexed list."""
        numbers = re.findall(r"(\d+)", response)
        ranking = []
        seen = set()
        for num_str in numbers:
            idx = int(num_str) - 1  # Convert 1-indexed to 0-indexed
            if 0 <= idx < n_candidates and idx not in seen:
                ranking.append(idx)
                seen.add(idx)
        return ranking

    def __repr__(self) -> str:
        return (
            f"LLMRanker(mode={self.mode!r}, "
            f"max_candidates={self.max_candidates}, "
            f"model={self.llm.model_name!r})"
        )
