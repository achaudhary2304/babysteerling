"""babysteerling: a small, laptop-trainable implementation of the Steerling architecture
(concept bottleneck language model) and the Atlas-inspired pipeline used to build its concept
dataset. Implements ideas from Guide Labs' "Scaling Inherently Interpretable Language Models"
technical report (see NOTICE).
"""
from .loss import ConceptLoss, IndependenceLoss, ReconstructionLoss, compute_losses
from . import nn
from .model import BaseLM, Diffusion, GPT, build_model
from .training import estimate_loss, get_lr, run_batch
from .steering import steered, injected_at, InterventionModule, AddDirectionStrategy

__version__ = "0.1.0"

__all__ = [
    "nn",

    "BaseLM",
    "GPT",
    "Diffusion",
    "build_model",

    "steered",
    "injected_at",
    "InterventionModule",
    "AddDirectionStrategy",

    "ConceptLoss",
    "ReconstructionLoss",
    "IndependenceLoss",
    "compute_losses",

    "get_lr",
    "run_batch",
    "estimate_loss",
]
