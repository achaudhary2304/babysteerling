"""babysteerling: a small, laptop-trainable implementation of the Steerling architecture
(concept-bottleneck language model) and the Atlas-inspired pipeline used to build its concept
dataset. Independently implements ideas described in Guide Labs' "Scaling Inherently
Interpretable Language Models" technical report -- see NOTICE for attribution.
"""
from .loss import ConceptLoss, IndependenceLoss, ReconstructionLoss, compute_losses
from .nn import (
    ConceptBottleneck,
    ConceptLMHead,
    SteerlingGPT,
    SupervisedConceptHead,
    TransformerModel,
    UnsupervisedConceptHead,
    build_model,
)
from .training import estimate_loss, get_lr, run_batch

__version__ = "0.1.0"

__all__ = [
    "SteerlingGPT",
    "build_model",
    "ConceptBottleneck",
    "SupervisedConceptHead",
    "UnsupervisedConceptHead",
    "ConceptLMHead",
    "TransformerModel",
    "ConceptLoss",
    "ReconstructionLoss",
    "IndependenceLoss",
    "compute_losses",
    "get_lr",
    "run_batch",
    "estimate_loss",
]
