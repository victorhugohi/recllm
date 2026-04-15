"""Hardware profiling and model selection guidance."""

from __future__ import annotations

import platform
from dataclasses import dataclass, field

import torch


@dataclass
class ModelRecommendation:
    """A recommended model configuration for the detected hardware."""

    model_name: str
    quantization: str
    estimated_vram_gb: float
    estimated_tokens_per_sec: float
    note: str = ""


@dataclass
class HardwareProfile:
    """Detected hardware capabilities and model recommendations."""

    gpu_name: str | None
    gpu_vram_gb: float
    cuda_version: str | None
    cpu_cores: int
    ram_total_gb: float
    platform: str
    recommended_models: list[ModelRecommendation] = field(default_factory=list)
    recommended_strategy: str = "sequential"

    def summary(self) -> str:
        """Return a human-readable hardware summary."""
        lines = [
            "RecLLM Hardware Profile",
            "=" * 40,
            f"GPU: {self.gpu_name or 'None detected'}"
            + (f" ({self.gpu_vram_gb:.1f} GB VRAM)" if self.gpu_name else ""),
        ]
        if self.cuda_version:
            lines.append(f"CUDA: {self.cuda_version}")
        lines.append(f"CPU: {self.cpu_cores} cores, {self.ram_total_gb:.0f} GB RAM")
        lines.append(f"Platform: {self.platform}")
        lines.append("")

        if self.recommended_models:
            lines.append("Recommended Models:")
            for rec in self.recommended_models:
                star = " *" if rec.note else ""
                lines.append(
                    f"  {rec.model_name} {rec.quantization}"
                    f"  (~{rec.estimated_vram_gb:.1f} GB, ~{rec.estimated_tokens_per_sec:.0f} tok/s)"
                    f"{star}"
                )
                if rec.note:
                    lines.append(f"    {rec.note}")
        lines.append(f"\nRecommended Strategy: {self.recommended_strategy}")
        return "\n".join(lines)


def profile_hardware() -> HardwareProfile:
    """Detect available hardware and recommend LLM configurations.

    Returns:
        HardwareProfile with detected specs and model recommendations.
    """
    import os

    cpu_cores = os.cpu_count() or 1
    # Estimate RAM (platform-dependent)
    try:
        import psutil

        ram_total_gb = psutil.virtual_memory().total / (1024**3)
    except ImportError:
        ram_total_gb = 0.0

    gpu_name = None
    gpu_vram_gb = 0.0
    cuda_version = None

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        cuda_version = torch.version.cuda

    profile = HardwareProfile(
        gpu_name=gpu_name,
        gpu_vram_gb=gpu_vram_gb,
        cuda_version=cuda_version,
        cpu_cores=cpu_cores,
        ram_total_gb=ram_total_gb,
        platform=platform.system(),
    )

    # Generate model recommendations based on available VRAM
    if gpu_vram_gb >= 20:
        profile.recommended_models = [
            ModelRecommendation("Mistral 7B", "FP16", 14.0, 80, "Full precision"),
            ModelRecommendation("Llama 3.1 8B", "Q4_K_M", 4.5, 60, "Best balance"),
            ModelRecommendation("Mistral 7B", "Q4_K_M", 4.1, 65, "Fast + quality"),
        ]
        profile.recommended_strategy = "shared"
    elif gpu_vram_gb >= 10:
        profile.recommended_models = [
            ModelRecommendation("Mistral 7B", "Q4_K_M", 4.1, 30, "Best balance"),
            ModelRecommendation("Llama 3.1 8B", "Q4_K_M", 4.5, 25, "Strong reasoning"),
            ModelRecommendation("Phi-3 Mini", "Q4_K_M", 2.2, 45, "Fastest"),
        ]
        profile.recommended_strategy = "sequential"
    elif gpu_vram_gb >= 3:
        profile.recommended_models = [
            ModelRecommendation("Phi-3 Mini", "Q4_K_M", 2.2, 12, "Best for low VRAM"),
            ModelRecommendation("SmolLM2 1.7B", "Q4_K_M", 1.0, 18, "Ultra-compact"),
        ]
        profile.recommended_strategy = "sequential"
    else:
        profile.recommended_models = [
            ModelRecommendation("SmolLM2 1.7B", "Q4_K_M", 1.0, 5, "CPU inference"),
            ModelRecommendation("Phi-3 Mini", "Q4_K_M", 2.2, 3, "CPU, slower"),
        ]
        profile.recommended_strategy = "sequential"

    return profile
