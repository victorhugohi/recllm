"""Recommendation model implementations."""

from recllm.models.base import BaseModel
from recllm.models.popularity import PopularityBaseline

__all__ = ["BaseModel", "PopularityBaseline"]

# Lazy imports for torch-dependent models
def __getattr__(name: str):
    if name == "BPR":
        from recllm.models.bpr import BPR
        return BPR
    if name == "NCF":
        from recllm.models.ncf import NCF
        return NCF
    if name == "SASRec":
        from recllm.models.sasrec import SASRec
        return SASRec
    if name == "LightGCN":
        from recllm.models.lightgcn import LightGCN
        return LightGCN
    if name == "DeepFM":
        from recllm.models.deepfm import DeepFM
        return DeepFM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
