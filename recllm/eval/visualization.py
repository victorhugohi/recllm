"""Visualization utilities for recommendation experiment results.

Generates publication-ready plots for thesis and papers.
Requires matplotlib: pip install recllm[viz]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def plot_model_comparison(
    results: dict[str, dict[str, float]],
    metrics: list[str] | None = None,
    title: str = "Model Comparison",
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> Any:
    """Bar chart comparing models across metrics.

    Args:
        results: Dict mapping model names to metric dicts.
            E.g. {"BPR": {"ndcg@10": 0.35, "hr@10": 0.55}, ...}
        metrics: Which metrics to plot. If None, uses all metrics
            from the first model.
        title: Plot title.
        save_path: If provided, saves the figure to this path.
        figsize: Figure size in inches.

    Returns:
        matplotlib Figure object.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if metrics is None:
        metrics = list(next(iter(results.values())).keys())

    model_names = list(results.keys())
    n_models = len(model_names)
    n_metrics = len(metrics)

    x = np.arange(n_metrics)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=figsize)

    for i, name in enumerate(model_names):
        values = [results[name].get(m, 0) for m in metrics]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=name)
        # Add value labels on top of bars
        for bar, val in zip(bars, values, strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=7,
            )

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_training_curves(
    histories: dict[str, list[float]],
    ylabel: str = "Loss",
    title: str = "Training Curves",
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (8, 5),
) -> Any:
    """Line plot of training loss/metric over epochs.

    Args:
        histories: Dict mapping model names to lists of per-epoch values.
        ylabel: Y-axis label.
        title: Plot title.
        save_path: If provided, saves the figure.
        figsize: Figure size.

    Returns:
        matplotlib Figure object.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    for name, values in histories.items():
        ax.plot(range(1, len(values) + 1), values, marker="o", markersize=3, label=name)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_metric_heatmap(
    results: dict[str, dict[str, float]],
    title: str = "Model-Metric Heatmap",
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (8, 5),
    cmap: str = "YlOrRd",
) -> Any:
    """Heatmap of models (rows) vs metrics (columns).

    Args:
        results: Dict mapping model names to metric dicts.
        title: Plot title.
        save_path: If provided, saves the figure.
        figsize: Figure size.
        cmap: Colormap name.

    Returns:
        matplotlib Figure object.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    model_names = list(results.keys())
    metrics = list(next(iter(results.values())).keys())

    data = np.array([
        [results[m].get(metric, 0) for metric in metrics]
        for m in model_names
    ])

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, cmap=cmap, aspect="auto")

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names)

    # Annotate cells
    for i in range(len(model_names)):
        for j in range(len(metrics)):
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", fontsize=9)

    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def results_to_latex(
    results: dict[str, dict[str, float]],
    caption: str = "Model comparison results",
    label: str = "tab:results",
    bold_best: bool = True,
) -> str:
    """Convert results dict to a LaTeX table string.

    Args:
        results: Dict mapping model names to metric dicts.
        caption: Table caption.
        label: LaTeX label for referencing.
        bold_best: Whether to bold the best value in each metric column.

    Returns:
        LaTeX table string.
    """
    import numpy as np

    model_names = list(results.keys())
    metrics = list(next(iter(results.values())).keys())

    # Find best per metric
    best = {}
    if bold_best:
        for metric in metrics:
            values = [results[m].get(metric, 0) for m in model_names]
            best[metric] = model_names[int(np.argmax(values))]

    # Build table
    col_spec = "l" + "c" * len(metrics)
    header = " & ".join(["Model"] + [m.replace("@", r"@") for m in metrics])

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
    ]

    for name in model_names:
        cells = [name]
        for metric in metrics:
            val = results[name].get(metric, 0)
            formatted = f"{val:.4f}"
            if bold_best and best.get(metric) == name:
                formatted = f"\\textbf{{{formatted}}}"
            cells.append(formatted)
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)
