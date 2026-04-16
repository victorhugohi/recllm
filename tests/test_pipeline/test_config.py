"""Tests for YAML/JSON pipeline configuration."""

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from recllm.pipeline.config import _build_model, load_config


def test_build_model_popularity():
    model = _build_model({"type": "popularity"})
    assert model.__class__.__name__ == "PopularityBaseline"


def test_build_model_bpr():
    model = _build_model({"type": "bpr", "embedding_dim": 32})
    assert model.__class__.__name__ == "BPR"
    assert model.embedding_dim == 32


def test_build_model_ncf():
    model = _build_model({"type": "ncf", "gmf_dim": 16, "mlp_dims": [32, 16]})
    assert model.__class__.__name__ == "NCF"
    assert model.gmf_dim == 16


def test_build_model_deepfm():
    model = _build_model({"type": "deepfm", "embed_dim": 16})
    assert model.__class__.__name__ == "DeepFM"


def test_build_model_sasrec():
    model = _build_model({"type": "sasrec", "embedding_dim": 16, "n_heads": 2})
    assert model.__class__.__name__ == "SASRec"


def test_build_model_lightgcn():
    model = _build_model({"type": "lightgcn", "embedding_dim": 16})
    assert model.__class__.__name__ == "LightGCN"


def test_build_model_unknown():
    with pytest.raises(ValueError, match="Unknown model type"):
        _build_model({"type": "nonexistent"})


def test_load_config_yaml():
    config = {"seed": 42, "epochs": 10, "data": {"type": "movielens"}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        f.flush()
        loaded = load_config(f.name)
    assert loaded["seed"] == 42
    assert loaded["epochs"] == 10
    Path(f.name).unlink()


def test_load_config_json():
    config = {"seed": 123, "epochs": 5}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f)
        f.flush()
        loaded = load_config(f.name)
    assert loaded["seed"] == 123
    Path(f.name).unlink()


def test_load_config_unsupported():
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"hello")
        f.flush()
        with pytest.raises(ValueError, match="Unsupported config format"):
            load_config(f.name)
    Path(f.name).unlink()
