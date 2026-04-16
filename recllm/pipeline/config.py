"""YAML/JSON pipeline configuration loader.

Enables declarative experiment definitions for reproducible research.
A single config file specifies dataset, model, LLM backend, enhancement
settings, evaluation metrics, and output paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from recllm.data.base import InteractionData
from recllm.models.base import BaseModel
from recllm.pipeline.recommendation import PipelineResult, RecommendationPipeline


def _build_model(model_config: dict) -> BaseModel:
    """Instantiate a model from config dict.

    Args:
        model_config: Dict with "type" key and optional hyperparameters.

    Returns:
        Instantiated model.
    """
    model_type = model_config.pop("type")
    model_map = {
        "popularity": "recllm.models.popularity.PopularityBaseline",
        "bpr": "recllm.models.bpr.BPR",
        "ncf": "recllm.models.ncf.NCF",
        "deepfm": "recllm.models.deepfm.DeepFM",
        "sasrec": "recllm.models.sasrec.SASRec",
        "lightgcn": "recllm.models.lightgcn.LightGCN",
    }

    qualified_name = model_map.get(model_type.lower())
    if qualified_name is None:
        raise ValueError(
            f"Unknown model type: {model_type!r}. "
            f"Available: {sorted(model_map.keys())}"
        )

    module_path, class_name = qualified_name.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)
    return model_class(**model_config)


def _build_dataset(data_config: dict) -> InteractionData:
    """Load a dataset from config dict.

    Args:
        data_config: Dict with "type", "version"/"category", and optional params.

    Returns:
        Loaded InteractionData.
    """
    data_type = data_config.pop("type").lower()

    if data_type == "movielens":
        from recllm.data.movielens import MovieLens
        version = data_config.pop("version", "100k")
        loader = MovieLens(version=version, **data_config).load()
        min_user = data_config.get("min_user", 5)
        min_item = data_config.get("min_item", 5)
        loader.preprocess(min_user_interactions=min_user, min_item_interactions=min_item)
        return loader.data
    elif data_type == "amazon":
        from recllm.data.amazon import AmazonReviews
        category = data_config.pop("category", "digital_music")
        loader = AmazonReviews(category=category, **data_config).load()
        return loader.data
    elif data_type == "bookcrossing":
        from recllm.data.bookcrossing import BookCrossing
        explicit_only = data_config.pop("explicit_only", True)
        loader = BookCrossing(explicit_only=explicit_only, **data_config).load()
        return loader.data
    elif data_type == "yelp":
        from recllm.data.yelp import YelpDataset
        data_dir = data_config.pop("data_dir")
        loader = YelpDataset(data_dir=data_dir, **data_config).load()
        return loader.data
    else:
        raise ValueError(
            f"Unknown dataset type: {data_type!r}. "
            "Available: movielens, amazon, bookcrossing, yelp"
        )


def _build_llm(llm_config: dict):
    """Instantiate an LLM client from config dict.

    Args:
        llm_config: Dict with "type" key and backend-specific params.

    Returns:
        LLMClient instance.
    """
    llm_type = llm_config.pop("type").lower()

    if llm_type == "ollama":
        from recllm.llm.ollama import OllamaClient
        return OllamaClient(**llm_config)
    elif llm_type == "openai":
        from recllm.llm.openai_client import OpenAIClient
        return OpenAIClient(**llm_config)
    elif llm_type == "llamacpp":
        from recllm.llm.llamacpp import LlamaCppClient
        return LlamaCppClient(**llm_config)
    else:
        raise ValueError(
            f"Unknown LLM type: {llm_type!r}. Available: ollama, openai, llamacpp"
        )


def load_config(path: str | Path) -> dict[str, Any]:
    """Load experiment configuration from YAML or JSON file.

    Args:
        path: Path to config file (.yaml, .yml, or .json).

    Returns:
        Parsed configuration dict.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    if path.suffix == ".json":
        return json.loads(text)
    elif path.suffix in (".yaml", ".yml"):
        return yaml.safe_load(text)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}. Use .yaml or .json")


def run_from_config(
    config: dict[str, Any] | str | Path,
) -> dict[str, PipelineResult]:
    """Run one or more experiments from a configuration dict or file.

    Config structure:
        ```yaml
        seed: 42
        epochs: 20
        metrics: [ndcg@10, hr@10, mrr]
        data:
          type: movielens
          version: 100k
        models:
          - type: popularity
          - type: bpr
            embed_dim: 64
          - type: ncf
            gmf_dim: 32
        llm:  # optional
          type: ollama
          model: mistral:7b
        enhance:  # optional
          items: true
          users: false
        ```

    Args:
        config: Configuration dict, or path to YAML/JSON config file.

    Returns:
        Dict mapping model names to PipelineResult.
    """
    if isinstance(config, (str, Path)):
        config = load_config(config)

    seed = config.get("seed", 42)
    epochs = config.get("epochs", 20)
    metrics = config.get("metrics", ["ndcg@10", "hr@10"])
    split_ratios = tuple(config.get("split_ratios", [0.8, 0.1, 0.1]))

    # Load dataset
    data_config = dict(config["data"])
    data = _build_dataset(data_config)

    # Build models
    model_configs = config.get("models", [{"type": "popularity"}])
    models: dict[str, BaseModel] = {}
    for mc in model_configs:
        mc = dict(mc)  # copy to avoid mutating original
        model_type = mc["type"]
        model = _build_model(dict(mc))
        # Allow multiple models of same type with different configs
        name = model_type.upper()
        if name in models:
            name = f"{name}_{len(models)}"
        models[name] = model

    # Optional LLM enhancement
    enhancer = None
    enhance_items = False
    enhance_users = False
    if "llm" in config and "enhance" in config:
        llm_config = dict(config["llm"])
        llm_client = _build_llm(llm_config)

        from recllm.enhance.feature_enhancer import FeatureEnhancer
        enhancer = FeatureEnhancer(llm_client=llm_client)
        enhance_cfg = config["enhance"]
        enhance_items = enhance_cfg.get("items", False)
        enhance_users = enhance_cfg.get("users", False)

    # Run pipeline for each model
    results: dict[str, PipelineResult] = {}
    for name, model in models.items():
        pipeline = RecommendationPipeline(
            model=model, seed=seed, eval_metrics=metrics
        )
        result = pipeline.run(
            data,
            epochs=epochs,
            enhancer=enhancer,
            enhance_items=enhance_items,
            enhance_users=enhance_users,
            split_ratios=split_ratios,
        )
        results[name] = result

    return results
