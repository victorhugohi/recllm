"""End-to-end recommendation pipeline.

Orchestrates the full flow: load data -> preprocess -> (optional) LLM enhance
-> train model -> evaluate, with reproducibility and logging.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from recllm.data.base import InteractionData
from recllm.models.base import BaseModel
from recllm.utils.reproducibility import set_seed

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Container for pipeline execution results.

    Attributes:
        metrics: Evaluation metrics dict.
        model: Trained model instance.
        train_data: Training split used.
        test_data: Test split used.
        enhanced_features: LLM-enhanced features if enhancement was used.
        config: Pipeline configuration snapshot.
        timing: Dict of stage durations in seconds.
    """

    metrics: dict[str, float]
    model: BaseModel
    train_data: InteractionData
    test_data: InteractionData
    enhanced_features: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = ["Pipeline Results", "=" * 40]
        lines.append(f"Model: {self.model.__class__.__name__}")
        lines.append(f"Train size: {self.train_data.n_interactions:,}")
        lines.append(f"Test size: {self.test_data.n_interactions:,}")
        lines.append("")
        lines.append("Metrics:")
        for name, value in sorted(self.metrics.items()):
            lines.append(f"  {name}: {value:.4f}")
        if self.timing:
            lines.append("")
            lines.append("Timing:")
            for stage, duration in self.timing.items():
                lines.append(f"  {stage}: {duration:.2f}s")
        return "\n".join(lines)


class RecommendationPipeline:
    """End-to-end recommendation experiment pipeline.

    Provides a structured way to run reproducible recommendation
    experiments with optional LLM feature enhancement.

    Args:
        model: Recommendation model instance.
        seed: Random seed for reproducibility.
        eval_metrics: Metrics to compute. Default: ["ndcg@10", "hr@10"].

    Example:
        >>> from recllm.models.popularity import PopularityBaseline
        >>> from recllm.data.movielens import MovieLensLoader
        >>>
        >>> data = MovieLensLoader("100k").load().preprocess()
        >>> model = PopularityBaseline()
        >>> pipeline = RecommendationPipeline(model)
        >>> result = pipeline.run(data)
        >>> print(result.summary())
    """

    def __init__(
        self,
        model: BaseModel,
        seed: int = 42,
        eval_metrics: list[str] | None = None,
    ):
        self.model = model
        self.seed = seed
        self.eval_metrics = eval_metrics or ["ndcg@10", "hr@10"]

    def run(
        self,
        data: InteractionData,
        epochs: int = 20,
        enhancer: Any | None = None,
        enhance_items: bool = False,
        enhance_users: bool = False,
        feature_col: str | None = None,
        split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    ) -> PipelineResult:
        """Execute the full pipeline.

        Args:
            data: Input interaction data (will be split if not already).
            epochs: Training epochs.
            enhancer: Optional FeatureEnhancer instance for LLM enhancement.
            enhance_items: Whether to generate LLM item features.
            enhance_users: Whether to generate LLM user features.
            feature_col: Item feature column for enhancement prompts.
            split_ratios: Train/val/test ratios if data needs splitting.

        Returns:
            PipelineResult with metrics, model, and timing.
        """
        timing: dict[str, float] = {}
        enhanced_features: dict[str, Any] = {}
        config = {
            "model": self.model.__class__.__name__,
            "seed": self.seed,
            "epochs": epochs,
            "eval_metrics": self.eval_metrics,
            "enhance_items": enhance_items,
            "enhance_users": enhance_users,
        }

        # Step 1: Seed
        set_seed(self.seed)
        logger.info("Pipeline started with seed=%d", self.seed)

        # Step 2: Split data if needed
        t0 = time.time()
        if data.train is not None and data.test is not None:
            train_data = data.train
            test_data = data.test
            val_data = data.val
        else:
            from recllm.data.splitting import random_split

            split_result = random_split(
                data,
                test_ratio=split_ratios[2],
                val_ratio=split_ratios[1],
                seed=self.seed,
            )
            train_data = split_result.train
            val_data = split_result.val
            test_data = split_result.test
        timing["split"] = time.time() - t0
        logger.info(
            "Data split: train=%d, val=%s, test=%d",
            train_data.n_interactions,
            val_data.n_interactions if val_data else "N/A",
            test_data.n_interactions,
        )

        # Step 3: Optional LLM enhancement
        if enhancer is not None:
            if enhance_items:
                t0 = time.time()
                item_features = enhancer.enhance_items(
                    train_data, feature_col=feature_col
                )
                enhanced_features["items"] = item_features
                timing["enhance_items"] = time.time() - t0
                logger.info(
                    "Enhanced %d items (dim=%s)",
                    item_features.n_entities,
                    item_features.embedding_dim,
                )

            if enhance_users:
                t0 = time.time()
                user_features = enhancer.enhance_users(train_data)
                enhanced_features["users"] = user_features
                timing["enhance_users"] = time.time() - t0
                logger.info(
                    "Enhanced %d users (dim=%s)",
                    user_features.n_entities,
                    user_features.embedding_dim,
                )

        # Step 4: Train
        t0 = time.time()
        self.model.fit(train_data, epochs=epochs, val_data=val_data)
        timing["train"] = time.time() - t0
        logger.info("Training completed in %.2fs", timing["train"])

        # Step 5: Evaluate
        t0 = time.time()
        metrics = self.model.evaluate(test_data, metrics=self.eval_metrics)
        timing["evaluate"] = time.time() - t0
        logger.info("Evaluation: %s", metrics)

        return PipelineResult(
            metrics=metrics,
            model=self.model,
            train_data=train_data,
            test_data=test_data,
            enhanced_features=enhanced_features,
            config=config,
            timing=timing,
        )

    def compare(
        self,
        models: dict[str, BaseModel],
        data: InteractionData,
        epochs: int = 20,
        **kwargs,
    ) -> dict[str, PipelineResult]:
        """Run pipeline for multiple models and compare results.

        Args:
            models: Dict mapping model names to model instances.
            data: Input data (split once, shared across models).
            epochs: Training epochs for each model.
            **kwargs: Additional args passed to run().

        Returns:
            Dict mapping model names to PipelineResult.
        """
        results: dict[str, PipelineResult] = {}

        # Split once for fair comparison
        set_seed(self.seed)
        from recllm.data.splitting import random_split

        split_ratios = kwargs.pop("split_ratios", (0.8, 0.1, 0.1))
        split_result = random_split(
            data,
            test_ratio=split_ratios[2],
            val_ratio=split_ratios[1],
            seed=self.seed,
        )
        data.train = split_result.train
        data.val = split_result.val
        data.test = split_result.test

        for name, model in models.items():
            logger.info("Running model: %s", name)
            self.model = model
            results[name] = self.run(data, epochs=epochs, **kwargs)
            logger.info(
                "%s -> %s",
                name,
                {k: f"{v:.4f}" for k, v in results[name].metrics.items()},
            )

        return results
