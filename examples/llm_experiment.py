"""End-to-end LLM enhancement experiment on MovieLens-100K.

Demonstrates all three LLM-RecSys integration patterns with a real
Ollama model, comparing baseline vs. LLM-enhanced recommendations.

This is the first real validation of RecLLM's full pipeline.

Usage:
    python examples/llm_experiment.py
    python examples/llm_experiment.py --model qwen3:8b --epochs 10
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from recllm.data.movielens import MovieLens
from recllm.data.splitting import random_split
from recllm.enhance.explainer import LLMExplainer
from recllm.enhance.feature_enhancer import FeatureEnhancer
from recllm.enhance.ranker import LLMRanker
from recllm.eval.metrics import compute_metrics
from recllm.llm.ollama import OllamaClient
from recllm.utils.reproducibility import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_experiment(
    ollama_model: str = "qwen3:8b",
    epochs: int = 10,
    seed: int = 42,
    n_enhance_items: int = 50,
    n_rerank_users: int = 20,
    output_dir: str = "results",
):
    """Run the full LLM experiment.

    Args:
        ollama_model: Ollama model name.
        epochs: Training epochs for base models.
        seed: Random seed.
        n_enhance_items: Number of items to enhance (subset for speed).
        n_rerank_users: Number of users for re-ranking demo.
        output_dir: Output directory for results.
    """
    set_seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Step 0: Verify Ollama ---
    # think=False skips chain-of-thought for reasoning models (qwen3,
    # deepseek-r1) so the experiment finishes in minutes, not hours.
    llm = OllamaClient(
        model=ollama_model, num_ctx=2048, timeout=300, think=False
    )
    if not llm.is_available():
        logger.error(
            "Ollama model %s not available. "
            "Start Ollama and pull the model first.",
            ollama_model,
        )
        return
    logger.info("Ollama model %s is available.", ollama_model)

    # --- Step 1: Load data ---
    logger.info("Loading MovieLens-100K...")
    loader = MovieLens("100k").load()
    data = loader.preprocess(
        min_user_interactions=5, min_item_interactions=5
    )

    # Check item features
    has_features = data.data.item_features is not None
    logger.info(
        "Dataset: %d users, %d items, %d interactions. Item features: %s",
        data.data.n_users, data.data.n_items,
        data.data.n_interactions, has_features,
    )

    split = random_split(
        data.data, test_ratio=0.1, val_ratio=0.1, seed=seed
    )
    train = split.train
    val = split.val
    test = split.test

    results = {
        "ollama_model": ollama_model,
        "seed": seed,
        "epochs": epochs,
        "data": {
            "n_users": data.data.n_users,
            "n_items": data.data.n_items,
            "n_interactions": data.data.n_interactions,
        },
    }

    # --- Step 2: Train baseline models ---
    logger.info("Training baseline models...")
    from recllm.models.bpr import BPR
    from recllm.models.popularity import PopularityBaseline

    baselines = {
        "Popularity": PopularityBaseline(),
        "BPR": BPR(embedding_dim=64, learning_rate=0.01, device="cpu"),
    }

    metrics_list = ["ndcg@10", "hr@10", "mrr"]
    baseline_results = {}

    for name, model in baselines.items():
        t0 = time.time()
        model.fit(train, epochs=epochs, val_data=val)
        train_time = time.time() - t0

        metrics = compute_metrics(model, test, metrics_list)
        baseline_results[name] = {
            "metrics": metrics,
            "train_time": round(train_time, 2),
        }
        logger.info(
            "  %s: NDCG@10=%.4f, HR@10=%.4f, MRR=%.4f (%.1fs)",
            name, metrics["ndcg@10"], metrics["hr@10"],
            metrics["mrr"], train_time,
        )

    results["baselines"] = {
        k: v["metrics"] for k, v in baseline_results.items()
    }

    # --- Step 3: LLM Feature Enhancement ---
    logger.info(
        "=== Pattern 1: LLM Feature Enhancement (text-only, %d items) ===",
        n_enhance_items,
    )
    enhancer = FeatureEnhancer(
        llm, cache_dir="llm_cache", batch_size=1,
    )

    # Enhance a subset of items (text descriptions only, no embeddings)
    # Use item features (titles) if available
    feature_col = "title" if has_features else None
    t0 = time.time()

    # Create a smaller subset for the demo
    import polars as pl
    top_items = (
        train.interactions.group_by("item_id")
        .len()
        .sort("len", descending=True)
        .head(n_enhance_items)["item_id"]
        .to_list()
    )

    # Build a mini InteractionData with just these items
    mini_df = train.interactions.filter(
        pl.col("item_id").is_in(top_items)
    )
    from recllm.data.base import InteractionData
    mini_data = InteractionData(
        interactions=mini_df,
        item_features=data.data.item_features,
        metadata={"name": "mini-subset"},
    )

    enhanced_items = enhancer.enhance_items(
        mini_data, feature_col=feature_col, embed=False,
    )
    enhance_time = time.time() - t0

    logger.info(
        "Enhanced %d items in %.1fs",
        enhanced_items.n_entities, enhance_time,
    )

    # Show sample enhanced texts
    sample_ids = list(enhanced_items.texts.keys())[:3]
    sample_texts = {}
    for iid in sample_ids:
        text = enhanced_items.texts[iid]
        sample_texts[str(iid)] = text[:200] + "..." if len(text) > 200 else text
        logger.info("  Item %d: %s", iid, text[:100])

    results["feature_enhancement"] = {
        "n_items_enhanced": enhanced_items.n_entities,
        "time_s": round(enhance_time, 2),
        "sample_texts": sample_texts,
    }

    # --- Step 4: LLM Re-Ranking ---
    logger.info(
        "=== Pattern 2: LLM Re-Ranking (%d users, listwise) ===",
        n_rerank_users,
    )

    # Get BPR's top candidates for a few users, then re-rank with LLM
    bpr_model = baselines["BPR"]
    ranker = LLMRanker(llm, mode="listwise", max_candidates=10)

    # Build item title lookup
    title_map = {}
    if has_features and "title" in data.data.item_features.columns:
        for row in data.data.item_features.iter_rows(named=True):
            title_map[row["item_id"]] = row["title"]

    rerank_results = []
    test_users = list(
        test.interactions["user_id"].unique().sort().to_list()
    )[:n_rerank_users]

    t0 = time.time()
    for user_id in test_users:
        # Get BPR recommendations
        seen = set()
        if hasattr(bpr_model, "_user_interactions"):
            seen = bpr_model._user_interactions.get(user_id, set())

        try:
            bpr_recs = bpr_model.recommend(
                user_id, n=10, exclude_seen=True, seen_items=seen,
            )
        except (RuntimeError, KeyError):
            continue

        # Build user history from titles
        user_items = train.interactions.filter(
            pl.col("user_id") == user_id
        )["item_id"].to_list()[:10]
        user_history = [
            title_map.get(int(iid), f"Item {iid}")
            for iid in user_items
        ]

        # Build candidate list from BPR recs
        candidates = [
            title_map.get(int(iid), f"Item {iid}")
            for iid, _ in bpr_recs
        ]
        candidate_ids = [int(iid) for iid, _ in bpr_recs]

        if not candidates:
            continue

        # Re-rank with LLM
        reranked = ranker.rerank(
            user_history=user_history,
            candidates=candidates,
            candidate_ids=candidate_ids,
        )

        rerank_results.append({
            "user_id": int(user_id),
            "bpr_order": candidate_ids,
            "llm_order": [int(iid) for iid, _ in reranked],
            "user_history": user_history[:5],
        })

        logger.info(
            "  User %d: BPR top-3 = %s, LLM top-3 = %s",
            user_id,
            [title_map.get(iid, str(iid)) for iid in candidate_ids[:3]],
            [
                title_map.get(int(iid), str(iid))
                for iid, _ in reranked[:3]
            ],
        )

    rerank_time = time.time() - t0
    logger.info(
        "Re-ranked %d users in %.1fs",
        len(rerank_results), rerank_time,
    )

    results["reranking"] = {
        "n_users": len(rerank_results),
        "time_s": round(rerank_time, 2),
        "mode": "listwise",
        "samples": rerank_results[:5],
    }

    # --- Step 5: LLM Explanation ---
    logger.info("=== Pattern 3: LLM Explanation (3 styles) ===")

    explanation_results = {}

    # Pick a user and their top recommendation
    if rerank_results:
        sample_user = rerank_results[0]
        user_history = sample_user["user_history"]
        top_item_id = sample_user["llm_order"][0]
        top_item_name = title_map.get(
            top_item_id, f"Item {top_item_id}"
        )

        for style in ["conversational", "analytical", "brief"]:
            explainer_s = LLMExplainer(llm, style=style)
            t0 = time.time()
            explanation = explainer_s.explain(
                user_history=user_history,
                recommended_item=top_item_name,
                score=0.92,
            )
            exp_time = time.time() - t0

            explanation_results[style] = {
                "text": explanation,
                "time_s": round(exp_time, 2),
            }
            logger.info(
                "  [%s] (%0.1fs): %s",
                style, exp_time, explanation[:150],
            )

    results["explanations"] = {
        "item": title_map.get(top_item_id, str(top_item_id))
        if rerank_results else "N/A",
        "styles": explanation_results,
    }

    # --- Step 6: Save results ---
    results_path = output_path / "llm_experiment.json"
    results_path.write_text(json.dumps(results, indent=2, default=str))
    logger.info("Full results saved to %s", results_path)

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("LLM EXPERIMENT RESULTS")
    print("=" * 60)
    print(f"Ollama model: {ollama_model}")
    print(f"Dataset: MovieLens-100K ({data.data.n_interactions:,} interactions)")
    print()

    print("Baseline Metrics:")
    for name, res in baseline_results.items():
        m = res["metrics"]
        print(
            f"  {name}: NDCG@10={m['ndcg@10']:.4f}, "
            f"HR@10={m['hr@10']:.4f}, MRR={m['mrr']:.4f}"
        )
    print()

    print(f"Feature Enhancement: {enhanced_items.n_entities} items "
          f"in {enhance_time:.1f}s")
    print(f"Re-Ranking: {len(rerank_results)} users "
          f"in {rerank_time:.1f}s (listwise)")
    print(f"Explanations: {len(explanation_results)} styles generated")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RecLLM LLM Enhancement Experiment"
    )
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-items", type=int, default=50)
    parser.add_argument("--n-users", type=int, default=20)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    run_experiment(
        ollama_model=args.model,
        epochs=args.epochs,
        seed=args.seed,
        n_enhance_items=args.n_items,
        n_rerank_users=args.n_users,
        output_dir=args.output_dir,
    )
