"""RecLLM Interactive Dashboard.

A Streamlit web application for exploring RecLLM library capabilities,
running experiments, and visualizing results.

Run with:
    streamlit run app/app.py
"""
# ruff: noqa: E501
from __future__ import annotations

import json
import time
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RecLLM Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("RecLLM")
st.sidebar.caption("LLM-Enhanced Recommendation Systems")

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Run Experiment",
        "Results Explorer",
        "Dataset Inspector",
        "Model Gallery",
        "Architecture",
    ],
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

MODELS_INFO = {
    "PopularityBaseline": {
        "type": "Non-personalized",
        "paradigm": "Frequency",
        "reference": "—",
        "params": {},
        "description": "Ranks items by interaction count. No personalization — serves as a lower-bound baseline.",
    },
    "BPR": {
        "type": "Matrix Factorization",
        "paradigm": "Collaborative Filtering",
        "reference": "Rendle et al. (2009)",
        "params": {"embedding_dim": 64, "learning_rate": 0.01},
        "description": "Bayesian Personalized Ranking learns user/item embeddings optimized for pairwise ranking.",
    },
    "NCF": {
        "type": "Deep Learning",
        "paradigm": "Neural CF",
        "reference": "He et al. (2017)",
        "params": {"gmf_dim": 32, "mlp_dims": [64, 32, 16], "loss": "bce"},
        "description": "Neural Collaborative Filtering fuses GMF (element-wise product) with MLP for nonlinear interaction modeling.",
    },
    "DeepFM": {
        "type": "FM + Deep",
        "paradigm": "Hybrid",
        "reference": "Guo et al. (2017)",
        "params": {"embed_dim": 32, "mlp_dims": [128, 64, 32], "loss": "bce"},
        "description": "Combines Factorization Machines (explicit feature interactions) with a Deep network (implicit high-order patterns), sharing embeddings.",
    },
    "SASRec": {
        "type": "Sequential / Transformer",
        "paradigm": "Self-Attention",
        "reference": "Kang & McAuley (2018)",
        "params": {"embedding_dim": 64, "n_heads": 2, "n_layers": 2, "max_seq_len": 50},
        "description": "Self-Attentive Sequential Recommendation uses causal Transformers to model item sequences.",
    },
    "LightGCN": {
        "type": "Graph Neural Network",
        "paradigm": "GNN",
        "reference": "He et al. (2020)",
        "params": {"embedding_dim": 64, "n_layers": 3},
        "description": "Light Graph Convolution Network simplifies GCN by removing nonlinearities, propagating embeddings on the user-item bipartite graph.",
    },
}

LLM_PATTERNS = {
    "Feature Enhancement": {
        "module": "FeatureEnhancer",
        "reference": "RLMRec (Ren 2024), KAR (Xi 2024)",
        "description": "LLM generates rich text features for items/users, which are embedded and concatenated with collaborative signals.",
    },
    "Ranking": {
        "module": "LLMRanker",
        "reference": "TALLRec (Bao 2024), Hou et al. (2024)",
        "description": "LLM re-ranks candidate items using pointwise scoring, pairwise comparison, or listwise ordering.",
    },
    "Explanation": {
        "module": "LLMExplainer",
        "reference": "—",
        "description": "LLM generates natural language explanations for why an item was recommended, in conversational, analytical, or brief styles.",
    },
}


def load_previous_results() -> dict | None:
    """Load results from the most recent benchmark run."""
    results_dir = Path("results")
    if not results_dir.exists():
        return None
    json_files = sorted(results_dir.glob("benchmark_*.json"), reverse=True)
    if not json_files:
        return None
    return json.loads(json_files[0].read_text())


def run_mini_experiment(
    dataset_version: str,
    selected_models: list[str],
    epochs: int,
    seed: int,
) -> dict:
    """Run a small-scale experiment and return results."""
    from recllm.data.movielens import MovieLens
    from recllm.data.splitting import random_split
    from recllm.eval.metrics import compute_metrics
    from recllm.utils.reproducibility import set_seed

    set_seed(seed)

    # Load data
    loader = MovieLens(dataset_version).load()
    data = loader.preprocess(
        min_user_interactions=5, min_item_interactions=5
    )
    split = random_split(
        data.data, test_ratio=0.1, val_ratio=0.1, seed=seed
    )
    train = split.train
    val = split.val
    test = split.test

    # Build selected models
    model_instances = {}
    if "PopularityBaseline" in selected_models:
        from recllm.models.popularity import PopularityBaseline
        model_instances["Popularity"] = PopularityBaseline()
    if "BPR" in selected_models:
        from recllm.models.bpr import BPR
        model_instances["BPR"] = BPR(
            embedding_dim=64, learning_rate=0.01, device="cpu"
        )
    if "NCF" in selected_models:
        from recllm.models.ncf import NCF
        model_instances["NCF"] = NCF(
            gmf_dim=32, mlp_dims=[64, 32, 16], loss="bce", device="cpu"
        )
    if "DeepFM" in selected_models:
        from recllm.models.deepfm import DeepFM
        model_instances["DeepFM"] = DeepFM(
            embed_dim=32, mlp_dims=[128, 64, 32], device="cpu"
        )
    if "SASRec" in selected_models:
        from recllm.models.sasrec import SASRec
        model_instances["SASRec"] = SASRec(
            embedding_dim=64, n_heads=2, n_layers=2, device="cpu"
        )
    if "LightGCN" in selected_models:
        from recllm.models.lightgcn import LightGCN
        model_instances["LightGCN"] = LightGCN(
            embedding_dim=64, n_layers=3, device="cpu"
        )

    metrics_list = ["ndcg@10", "hr@10", "mrr"]
    all_results = {}
    all_timing = {}

    progress = st.progress(0, text="Starting experiment...")
    total = len(model_instances)

    for idx, (name, model) in enumerate(model_instances.items()):
        progress.progress((idx) / total, text=f"Training {name}...")
        t0 = time.time()
        model.fit(train, epochs=epochs, val_data=val)
        train_time = time.time() - t0

        progress.progress(
            (idx + 0.5) / total, text=f"Evaluating {name}..."
        )
        t0 = time.time()
        metrics = compute_metrics(model, test, metrics_list)
        eval_time = time.time() - t0

        all_results[name] = metrics
        all_timing[name] = {
            "train_s": round(train_time, 2),
            "eval_s": round(eval_time, 2),
        }

    progress.progress(1.0, text="Done!")

    return {
        "metrics": all_results,
        "timing": all_timing,
        "data_stats": {
            "n_users": data.data.n_users,
            "n_items": data.data.n_items,
            "n_interactions": data.data.n_interactions,
            "density": data.data.density,
        },
        "config": {
            "dataset": f"ml-{dataset_version}",
            "epochs": epochs,
            "seed": seed,
        },
    }


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

if page == "Overview":
    st.title("RecLLM: LLM-Enhanced Recommendation Systems")
    st.markdown("""
    **RecLLM** is an open-source Python library for integrating Large Language Models
    into recommendation pipelines, with a focus on local inference.
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Models", "6", help="Spanning 5 paradigms")
    col2.metric("LLM Patterns", "3", help="Feature, Ranking, Explanation")
    col3.metric("Datasets", "4", help="MovieLens, Amazon, Yelp, BookCrossing")
    col4.metric("Tests", "85", help="All passing")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Library Architecture")
        arch_data = {
            "Layer": ["Data", "Model", "LLM", "Enhancement", "Evaluation", "Pipeline"],
            "Module": [
                "recllm.data",
                "recllm.models",
                "recllm.llm",
                "recllm.enhance",
                "recllm.eval",
                "recllm.pipeline",
            ],
            "Components": [
                "MovieLens, Amazon, Yelp, BookCrossing",
                "Popularity, BPR, NCF, DeepFM, SASRec, LightGCN",
                "Ollama, OpenAI, llama.cpp",
                "FeatureEnhancer, LLMRanker, LLMExplainer",
                "NDCG, HR, MRR, P, R + significance tests",
                "Pipeline, YAML config, visualization",
            ],
        }
        st.dataframe(arch_data, use_container_width=True, hide_index=True)

    with col_right:
        st.subheader("Research Context")
        st.markdown("""
        **Thesis:** *Design and Implementation of an Open-Source Python Library
        for LLM-Enhanced Recommendation Systems with Local Model Inference*

        **Candidate:** Victor Hugo
        **Program:** Doctorado en Informatica Aplicada, UMSA, La Paz, Bolivia
        **Methodology:** Design Science Research (DSR) + AI-assisted
        **Repository:** [github.com/victorhugohi/recllm](https://github.com/victorhugohi/recllm)
        """)

    st.divider()
    st.subheader("Model Paradigms")

    paradigm_data = []
    for name, info in MODELS_INFO.items():
        paradigm_data.append({
            "Model": name,
            "Paradigm": info["paradigm"],
            "Type": info["type"],
            "Reference": info["reference"],
        })
    st.dataframe(paradigm_data, use_container_width=True, hide_index=True)

elif page == "Run Experiment":
    st.title("Run Experiment")
    st.markdown("Train and compare recommendation models on MovieLens.")

    with st.sidebar:
        st.subheader("Experiment Settings")
        dataset = st.selectbox("Dataset", ["100k", "1m"], index=0)
        epochs = st.slider("Epochs", 1, 50, 10)
        seed = st.number_input("Seed", value=42, min_value=0)

        st.subheader("Select Models")
        model_options = list(MODELS_INFO.keys())
        selected = []
        for m in model_options:
            default_on = m in ["PopularityBaseline", "BPR", "NCF"]
            if st.checkbox(m, value=default_on):
                selected.append(m)

    if st.button(
        "Run Experiment", type="primary", disabled=len(selected) == 0
    ):
        if len(selected) == 0:
            st.warning("Select at least one model.")
        else:
            with st.spinner("Running experiment..."):
                results = run_mini_experiment(
                    dataset, selected, epochs, seed
                )

            # Store in session state
            st.session_state["last_results"] = results

            # Display results
            st.subheader("Results")

            # Metrics table
            metrics_data = []
            for name, m in results["metrics"].items():
                row = {"Model": name}
                row.update({k: round(v, 4) for k, v in m.items()})
                row["Train (s)"] = results["timing"][name]["train_s"]
                metrics_data.append(row)
            st.dataframe(
                metrics_data, use_container_width=True, hide_index=True
            )

            # Bar chart
            metric_names = list(
                next(iter(results["metrics"].values())).keys()
            )
            chart_metric = st.selectbox("Chart metric", metric_names)

            chart_data = {
                "Model": list(results["metrics"].keys()),
                "Score": [
                    results["metrics"][m][chart_metric]
                    for m in results["metrics"]
                ],
            }
            fig = px.bar(
                chart_data, x="Model", y="Score",
                title=f"{chart_metric} by Model",
                color="Model",
                text_auto=".4f",
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            # Radar chart
            if len(selected) >= 2 and len(metric_names) >= 3:
                st.subheader("Radar Comparison")
                fig_radar = go.Figure()
                for name in results["metrics"]:
                    values = [
                        results["metrics"][name][m] for m in metric_names
                    ]
                    values.append(values[0])
                    fig_radar.add_trace(go.Scatterpolar(
                        r=values,
                        theta=metric_names + [metric_names[0]],
                        fill="toself",
                        name=name,
                    ))
                fig_radar.update_layout(
                    polar={
                        "radialaxis": {"visible": True, "range": [0, 1]}
                    },
                    title="Multi-Metric Radar",
                )
                st.plotly_chart(fig_radar, use_container_width=True)

            # Timing
            st.subheader("Timing")
            timing_data = {
                "Model": list(results["timing"].keys()),
                "Training (s)": [
                    results["timing"][m]["train_s"]
                    for m in results["timing"]
                ],
                "Evaluation (s)": [
                    results["timing"][m]["eval_s"]
                    for m in results["timing"]
                ],
            }
            fig_timing = px.bar(
                timing_data,
                x="Model",
                y=["Training (s)", "Evaluation (s)"],
                title="Time by Model",
                barmode="group",
            )
            st.plotly_chart(fig_timing, use_container_width=True)

            # Dataset stats
            st.subheader("Dataset Statistics")
            ds = results["data_stats"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Users", f"{ds['n_users']:,}")
            c2.metric("Items", f"{ds['n_items']:,}")
            c3.metric("Interactions", f"{ds['n_interactions']:,}")
            c4.metric("Density", f"{ds['density']:.4%}")

            # Save results
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)
            out_path = results_dir / f"dashboard_ml_{dataset}.json"
            out_path.write_text(
                json.dumps(results, indent=2, default=str)
            )
            st.success(f"Results saved to {out_path}")

    else:
        st.info(
            "Configure models in the sidebar, "
            "then click **Run Experiment**."
        )

elif page == "Results Explorer":
    st.title("Results Explorer")

    # Try loading from session state or disk
    results = (
        st.session_state.get("last_results")
        or load_previous_results()
    )

    if results is None:
        st.info(
            "No results found. Run an experiment first, "
            "or place a benchmark JSON in `results/`."
        )
    else:
        dataset_name = results.get("config", {}).get("dataset", "unknown")
        st.subheader(f"Dataset: {dataset_name}")

        # Metrics comparison
        metrics = results["metrics"]
        model_names = list(metrics.keys())
        metric_names = list(next(iter(metrics.values())).keys())

        # Heatmap
        z = [
            [metrics[m][metric] for metric in metric_names]
            for m in model_names
        ]
        fig_heat = go.Figure(data=go.Heatmap(
            z=z,
            x=metric_names,
            y=model_names,
            texttemplate="%{z:.4f}",
            colorscale="YlOrRd",
        ))
        fig_heat.update_layout(title="Model-Metric Heatmap", height=400)
        st.plotly_chart(fig_heat, use_container_width=True)

        # Grouped bar chart
        import pandas as pd
        rows = []
        for name in model_names:
            for metric in metric_names:
                rows.append({
                    "Model": name,
                    "Metric": metric,
                    "Score": metrics[name][metric],
                })
        df = pd.DataFrame(rows)
        fig_grouped = px.bar(
            df, x="Metric", y="Score", color="Model",
            barmode="group", text_auto=".3f",
        )
        fig_grouped.update_layout(title="All Metrics by Model")
        st.plotly_chart(fig_grouped, use_container_width=True)

        # Best model highlight
        best_metric = (
            "ndcg@10" if "ndcg@10" in metric_names else metric_names[0]
        )
        best_model = max(
            model_names, key=lambda m: metrics[m][best_metric]
        )
        best_score = metrics[best_model][best_metric]
        st.success(
            f"Best model by {best_metric}: "
            f"**{best_model}** ({best_score:.4f})"
        )

        # Significance results if available
        if "significance" in results and results["significance"]:
            st.subheader("Statistical Significance")
            st.dataframe(
                results["significance"],
                use_container_width=True,
                hide_index=True,
            )

elif page == "Dataset Inspector":
    st.title("Dataset Inspector")
    st.markdown("Load and explore recommendation datasets.")

    dataset_choice = st.selectbox(
        "Dataset",
        ["MovieLens-100K", "MovieLens-1M"],
    )

    if st.button("Load Dataset"):
        with st.spinner("Loading..."):
            from recllm.data.movielens import MovieLens
            version = "100k" if "100K" in dataset_choice else "1m"
            loader = MovieLens(version).load()
            data = loader.data

        st.subheader("Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Users", f"{data.n_users:,}")
        col2.metric("Items", f"{data.n_items:,}")
        col3.metric("Interactions", f"{data.n_interactions:,}")
        col4.metric("Density", f"{data.density:.4%}")

        # Rating distribution
        st.subheader("Rating Distribution")
        ratings = data.interactions["rating"].to_list()
        fig_ratings = px.histogram(
            x=ratings, nbins=10,
            title="Rating Distribution",
            labels={"x": "Rating", "y": "Count"},
        )
        st.plotly_chart(fig_ratings, use_container_width=True)

        # Interactions per user
        st.subheader("User Activity Distribution")
        user_counts = (
            data.interactions.group_by("user_id").len()["len"].to_list()
        )
        fig_users = px.histogram(
            x=user_counts, nbins=50,
            title="Interactions per User",
            labels={
                "x": "Number of Interactions",
                "y": "Number of Users",
            },
        )
        st.plotly_chart(fig_users, use_container_width=True)

        # Interactions per item
        st.subheader("Item Popularity Distribution")
        item_counts = (
            data.interactions.group_by("item_id").len()["len"].to_list()
        )
        fig_items = px.histogram(
            x=item_counts, nbins=50,
            title="Interactions per Item",
            labels={
                "x": "Number of Interactions",
                "y": "Number of Items",
            },
        )
        st.plotly_chart(fig_items, use_container_width=True)

        # Item features if available
        if data.item_features is not None:
            st.subheader("Item Features (Movies)")
            st.dataframe(
                data.item_features.head(20).to_pandas(),
                use_container_width=True,
            )

            if "genres" in data.item_features.columns:
                st.subheader("Genre Distribution")
                genres_raw = data.item_features["genres"].to_list()
                genre_counts: dict[str, int] = {}
                for g in genres_raw:
                    if g:
                        for genre in str(g).split("|"):
                            genre = genre.strip()
                            if genre:
                                genre_counts[genre] = (
                                    genre_counts.get(genre, 0) + 1
                                )
                if genre_counts:
                    sorted_genres = sorted(
                        genre_counts.items(), key=lambda x: -x[1]
                    )
                    fig_genres = px.bar(
                        x=[g[0] for g in sorted_genres],
                        y=[g[1] for g in sorted_genres],
                        title="Movies per Genre",
                        labels={"x": "Genre", "y": "Count"},
                    )
                    st.plotly_chart(fig_genres, use_container_width=True)

        # Raw data sample
        st.subheader("Data Sample")
        st.dataframe(
            data.interactions.head(100).to_pandas(),
            use_container_width=True,
        )

elif page == "Model Gallery":
    st.title("Model Gallery")
    st.markdown(
        "Explore the 6 recommendation models and "
        "3 LLM integration patterns in RecLLM."
    )

    st.subheader("Recommendation Models")

    for name, info in MODELS_INFO.items():
        with st.expander(
            f"**{name}** — {info['type']} ({info['paradigm']})"
        ):
            st.markdown(f"**Reference:** {info['reference']}")
            st.markdown(info["description"])
            if info["params"]:
                st.markdown("**Default hyperparameters:**")
                st.json(info["params"])

            # Show code example
            if name == "PopularityBaseline":
                st.code("""
from recllm.models.popularity import PopularityBaseline
model = PopularityBaseline()
model.fit(train_data)
recs = model.recommend(user_id=1, n=10)
""", language="python")
            elif name == "BPR":
                st.code("""
from recllm.models import BPR
model = BPR(embedding_dim=64, learning_rate=0.01)
model.fit(train_data, epochs=20)
recs = model.recommend(user_id=1, n=10)
""", language="python")
            elif name == "NCF":
                st.code("""
from recllm.models import NCF
model = NCF(gmf_dim=32, mlp_dims=[64, 32, 16], loss="bce")
model.fit(train_data, epochs=20)
""", language="python")
            elif name == "DeepFM":
                st.code("""
from recllm.models import DeepFM
model = DeepFM(embed_dim=32, mlp_dims=[128, 64, 32])
model.fit(train_data, epochs=20)
""", language="python")
            elif name == "SASRec":
                st.code("""
from recllm.models import SASRec
model = SASRec(embedding_dim=64, n_heads=2, n_layers=2)
model.fit(train_data, epochs=20)
""", language="python")
            elif name == "LightGCN":
                st.code("""
from recllm.models import LightGCN
model = LightGCN(embedding_dim=64, n_layers=3)
model.fit(train_data, epochs=20)
""", language="python")

    st.divider()
    st.subheader("LLM Integration Patterns")

    for pattern_name, info in LLM_PATTERNS.items():
        with st.expander(f"**{pattern_name}** — `{info['module']}`"):
            st.markdown(f"**Reference:** {info['reference']}")
            st.markdown(info["description"])

            if pattern_name == "Feature Enhancement":
                st.code("""
from recllm.llm import OllamaClient
from recllm.enhance import FeatureEnhancer

llm = OllamaClient(model="mistral:7b")
enhancer = FeatureEnhancer(llm, cache_dir="llm_cache")
features = enhancer.enhance_items(train_data, feature_col="title")
""", language="python")
            elif pattern_name == "Ranking":
                st.code("""
from recllm.enhance import LLMRanker

ranker = LLMRanker(llm, mode="listwise")
reranked = ranker.rerank(
    user_history=["The Matrix", "Inception"],
    candidates=["Titanic", "Interstellar", "The Notebook"],
)
""", language="python")
            elif pattern_name == "Explanation":
                st.code("""
from recllm.enhance import LLMExplainer

explainer = LLMExplainer(llm, style="conversational")
explanation = explainer.explain(
    user_history=["The Matrix", "Blade Runner"],
    recommended_item="Interstellar",
    score=0.92,
)
""", language="python")

elif page == "Architecture":
    st.title("Library Architecture")

    st.markdown("""
    RecLLM follows a **5-layer modular architecture** designed for
    extensibility and the scikit-learn
    `fit() -> predict() -> recommend() -> evaluate()` pattern.
    """)

    # Architecture diagram using Mermaid
    st.subheader("Module Dependency Graph")

    mermaid_code = """
    graph TD
        A[recllm.data] -->|InteractionData| B[recllm.models]
        A -->|InteractionData| C[recllm.enhance]
        D[recllm.llm] -->|LLMClient| C
        B -->|BaseModel| E[recllm.eval]
        C -->|EnhancedFeatures| B
        A -->|InteractionData| E
        B -->|BaseModel| F[recllm.pipeline]
        E -->|metrics| F
        C -->|enhancer| F
        A -->|data| F

        style A fill:#4CAF50,color:white
        style B fill:#2196F3,color:white
        style C fill:#FF9800,color:white
        style D fill:#9C27B0,color:white
        style E fill:#F44336,color:white
        style F fill:#607D8B,color:white
    """
    st.code(mermaid_code, language="mermaid")

    st.subheader("Design Decisions (ADRs)")

    adrs = [
        ("ADR-001", "Polars for DataFrames",
         "Fast, lazy eval, zero-copy Arrow interop"),
        ("ADR-002", "Ollama as primary LLM backend",
         "Free, local, GPU-aware, simple HTTP API"),
        ("ADR-003", "Scikit-learn-style API",
         "fit/predict/recommend/evaluate — familiar to ML practitioners"),
        ("ADR-004", "Content-addressed disk caching",
         "SHA-256 keyed cache avoids redundant LLM calls"),
        ("ADR-005", "Sequential GPU time-multiplexing",
         "PyTorch and LLM share GPU by alternating"),
    ]

    for adr_id, title, rationale in adrs:
        with st.expander(f"**{adr_id}: {title}**"):
            st.markdown(f"**Rationale:** {rationale}")

    st.divider()

    st.subheader("Pipeline Flow")
    st.markdown("""
    ```
    Load -> Preprocess -> Split -> [LLM Enhance] -> Train -> Evaluate
     |          |           |           |              |          |
    ML-100K  filter_min  random     Enhancer      model.fit  NDCG@K
    Amazon              temporal    Ranker                    HR@K
    Yelp                LOO        Explainer                  MRR
    BookX                                                    P@K,R@K
    ```
    """)

    st.subheader("YAML Configuration")
    st.code("""
seed: 42
epochs: 20
metrics: [ndcg@10, hr@10, mrr]
data:
  type: movielens
  version: 100k
models:
  - type: popularity
  - type: bpr
    embedding_dim: 64
  - type: deepfm
    embed_dim: 32
    mlp_dims: [128, 64, 32]
""", language="yaml")

    st.code("""
from recllm.pipeline import run_from_config
results = run_from_config("experiment.yaml")
""", language="python")
