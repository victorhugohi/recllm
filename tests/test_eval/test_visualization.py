"""Tests for visualization utilities (non-rendering tests)."""

import pytest

from recllm.eval.visualization import results_to_latex


@pytest.fixture()
def sample_results():
    return {
        "BPR": {"ndcg@10": 0.35, "hr@10": 0.55, "mrr": 0.30},
        "NCF": {"ndcg@10": 0.40, "hr@10": 0.60, "mrr": 0.33},
        "DeepFM": {"ndcg@10": 0.38, "hr@10": 0.58, "mrr": 0.35},
    }


def test_results_to_latex_basic(sample_results):
    latex = results_to_latex(sample_results)
    assert r"\begin{table}" in latex
    assert r"\end{table}" in latex
    assert "BPR" in latex
    assert "NCF" in latex
    assert "DeepFM" in latex


def test_results_to_latex_bold_best(sample_results):
    latex = results_to_latex(sample_results, bold_best=True)
    # NCF has best ndcg@10 (0.40)
    assert r"\textbf{0.4000}" in latex
    # DeepFM has best mrr (0.35)
    assert r"\textbf{0.3500}" in latex


def test_results_to_latex_no_bold(sample_results):
    latex = results_to_latex(sample_results, bold_best=False)
    assert r"\textbf" not in latex


def test_results_to_latex_custom_caption(sample_results):
    latex = results_to_latex(
        sample_results,
        caption="Custom caption",
        label="tab:custom",
    )
    assert "Custom caption" in latex
    assert "tab:custom" in latex
