"""End-to-end MovieLens benchmark: compare all RecLLM models.

This script demonstrates the full RecLLM pipeline:
1. Load and preprocess MovieLens-100K
2. Train all available models (Popularity, BPR, NCF, SASRec, LightGCN, DeepFM)
3. Evaluate with standard metrics (NDCG@10, HR@10, MRR)
4. Run statistical significance tests between models
5. Export results as CSV for thesis tables

Usage:
    python examples/movielens_benchmark.py
    python examples/movielens_benchmark.py --dataset ml-100k --epochs 30
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from recllm.data.movielens import MovieLens
from recllm.data.splitting import random_split
from recllm.eval.metrics import compute_metrics
from recllm.eval.significance import (
    bootstrap_ci,
    compute_per_user_metrics,
    paired_t_test,
    wilcoxon_test,
)
from recllm.utils.reproducibility import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_models(device: str = "cpu") -> dict:
    """Build all available models with default hyperparameters."""
    from recllm.models.bpr import BPR
    from recllm.models.deepfm import DeepFM
    from recllm.models.lightgcn import LightGCN
    from recllm.models.ncf import NCF
    from recllm.models.popularity import PopularityBaseline
    from recllm.models.sasrec import SASRec

    return {
        "Popularity": PopularityBaseline(),
        "BPR": BPR(embedding_dim=64, learning_rate=0.01, device=device),
        "NCF": NCF(gmf_dim=32, mlp_dims=[64, 32, 16], loss="bce", device=device),
        "DeepFM": DeepFM(embed_dim=32, mlp_dims=[128, 64, 32], loss="bce", device=device),
        "SASRec": SASRec(embedding_dim=64, n_heads=2, n_layers=2, max_seq_len=50, device=device),
        "LightGCN": LightGCN(embedding_dim=64, n_layers=3, device=device),
    }


def run_benchmark(
    dataset: str = "ml-100k",
    epochs: int = 20,
    seed: int = 42,
    device: str = "cpu",
    output_dir: str = "results",
) -> dict:
    """Run the full benchmark.

    Args:
        dataset: MovieLens variant ("ml-100k" or "ml-1m").
        epochs: Training epochs for each model.
        seed: Random seed for reproducibility.
        device: PyTorch device.
        output_dir: Directory for result files.

    Returns:
        Dict with metrics, timing, and significance results.
    """
    set_seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Load and preprocess data ---
    logger.info("Loading %s dataset...", dataset)
    variant = dataset.replace("ml-", "")
    data = MovieLens(variant).load().preprocess(min_user=5, min_item=5)
    logger.info(
        "Dataset: %d users, %d items, %d interactions (density: %.4f%%)",
        data.data.n_users,
        data.data.n_items,
        data.data.n_interactions,
        data.data.density * 100,
    )

    # --- Step 2: Split data (once for fair comparison) ---
    split_data = random_split(data.data, test_ratio=0.1, val_ratio=0.1, seed=seed)
    train_data = split_data.train
    val_data = split_data.val
    test_data = split_data.test
    logger.info(
        "Split: train=%d, val=%d, test=%d",
        train_data.n_interactions,
        val_data.n_interactions,
        test_data.n_interactions,
    )

    # --- Step 3: Train and evaluate all models ---
    models = build_models(device)
    metrics_list = ["ndcg@10", "hr@10", "mrr"]
    all_results = {}
    all_timing = {}
    all_per_user = {}

    for name, model in models.items():
        logger.info("Training %s...", name)
        t0 = time.time()
        model.fit(train_data, epochs=epochs, val_data=val_data)
        train_time = time.time() - t0

        t0 = time.time()
        metrics = compute_metrics(model, test_data, metrics_list)
        eval_time = time.time() - t0

        all_results[name] = metrics
        all_timing[name] = {"train_s": round(train_time, 2), "eval_s": round(eval_time, 2)}

        # Compute per-user scores for significance testing
        per_user = compute_per_user_metrics(model, test_data, metric="ndcg@10")
        all_per_user[name] = per_user

        logger.info(
            "%s -> NDCG@10=%.4f, HR@10=%.4f, MRR=%.4f (train: %.1fs, eval: %.1fs)",
            name,
            metrics["ndcg@10"],
            metrics["hr@10"],
            metrics["mrr"],
            train_time,
            eval_time,
        )

    # --- Step 4: Statistical significance tests ---
    logger.info("Running significance tests...")
    model_names = list(all_results.keys())
    significance_results = []

    # Find best model by NDCG@10
    best_model = max(all_results, key=lambda m: all_results[m]["ndcg@10"])
    best_per_user = all_per_user[best_model]

    for name in model_names:
        if name == best_model:
            continue
        other_per_user = all_per_user[name]

        # Align per-user scores (only users present in both)
        common_users = sorted(set(best_per_user.keys()) & set(other_per_user.keys()))
        if len(common_users) < 10:
            logger.warning(
                "Too few common users (%d) for %s vs %s",
                len(common_users), best_model, name,
            )
            continue

        scores_best = np.array([best_per_user[u] for u in common_users])
        scores_other = np.array([other_per_user[u] for u in common_users])

        t_result = paired_t_test(
            scores_best, scores_other,
            model_a_name=best_model, model_b_name=name,
            metric_name="ndcg@10",
        )
        w_result = wilcoxon_test(
            scores_best, scores_other,
            model_a_name=best_model, model_b_name=name,
            metric_name="ndcg@10",
        )
        b_result = bootstrap_ci(
            scores_best, scores_other,
            model_a_name=best_model, model_b_name=name,
            metric_name="ndcg@10",
        )

        significance_results.append({
            "comparison": f"{best_model} vs {name}",
            "paired_t_p": t_result.p_value,
            "paired_t_sig": t_result.significant,
            "wilcoxon_p": w_result.p_value,
            "wilcoxon_sig": w_result.significant,
            "bootstrap_sig": b_result["significant"],
            "effect_size_d": t_result.effect_size,
        })

        logger.info(
            "%s vs %s: t-test p=%.4f (%s), Wilcoxon p=%.4f (%s), Cohen's d=%.3f",
            best_model, name,
            t_result.p_value, "SIG" if t_result.significant else "NS",
            w_result.p_value, "SIG" if w_result.significant else "NS",
            t_result.effect_size,
        )

    # --- Step 5: Export results ---
    # Metrics table (CSV-friendly)
    csv_lines = ["model," + ",".join(metrics_list) + ",train_s,eval_s"]
    for name in model_names:
        m = all_results[name]
        t = all_timing[name]
        vals = [f"{m[metric]:.4f}" for metric in metrics_list]
        csv_lines.append(f"{name},{','.join(vals)},{t['train_s']},{t['eval_s']}")
    csv_text = "\n".join(csv_lines)

    csv_path = output_path / f"benchmark_{dataset.replace('-', '_')}.csv"
    csv_path.write_text(csv_text)
    logger.info("Results saved to %s", csv_path)

    # Full results as JSON
    full_results = {
        "dataset": dataset,
        "seed": seed,
        "epochs": epochs,
        "device": device,
        "data_stats": {
            "n_users": data.data.n_users,
            "n_items": data.data.n_items,
            "n_interactions": data.data.n_interactions,
        },
        "metrics": all_results,
        "timing": all_timing,
        "significance": significance_results,
        "best_model": best_model,
    }

    json_path = output_path / f"benchmark_{dataset.replace('-', '_')}.json"
    json_path.write_text(json.dumps(full_results, indent=2))
    logger.info("Full results saved to %s", json_path)

    # Print summary table
    print("\n" + "=" * 70)
    print(f"BENCHMARK RESULTS: {dataset} (seed={seed}, epochs={epochs})")
    print("=" * 70)
    print(f"{'Model':<15} {'NDCG@10':>10} {'HR@10':>10} {'MRR':>10} {'Train(s)':>10}")
    print("-" * 70)
    for name in model_names:
        m = all_results[name]
        t = all_timing[name]
        marker = " *" if name == best_model else ""
        print(
            f"{name:<15} {m['ndcg@10']:>10.4f} {m['hr@10']:>10.4f} "
            f"{m['mrr']:>10.4f} {t['train_s']:>10.1f}{marker}"
        )
    print("-" * 70)
    print(f"* Best model by NDCG@10: {best_model}")

    if significance_results:
        print(f"\nSignificance tests ({best_model} vs others, alpha=0.05):")
        for s in significance_results:
            print(
                f"  {s['comparison']}: "
                f"t-test {'SIG' if s['paired_t_sig'] else 'NS'} (p={s['paired_t_p']:.4f}), "
                f"Wilcoxon {'SIG' if s['wilcoxon_sig'] else 'NS'} (p={s['wilcoxon_p']:.4f}), "
                f"d={s['effect_size_d']:.3f}"
            )
    print("=" * 70)

    return full_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RecLLM MovieLens Benchmark")
    parser.add_argument("--dataset", default="ml-100k", choices=["ml-100k", "ml-1m"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    run_benchmark(
        dataset=args.dataset,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
    )
