import torch
from torch import nn
from .encoder import SparseEmbeddingToConcept
from .prototype import (
    LinearSelector, LiftedTokenPredictor, ProductKeySelector, PrototypeConceptEncoder,
    PrototypeCrossAttention, PrototypePredictor,
)

_SELECTOR_TYPES = ("linear_selector", "product_key")
_PREDICTOR_TYPES = ("prototype", "lifted_tokens")


def _build_known_encoder(known_encoder_type, d, n, top_k_known, proto_token_ids, backbone,
                          topk_axis, chunk_size, key_dim, use_checkpoint, candidates_per_token,
                          predictor_type="prototype", lifted_top_k=5):
    """Builds self.known. All four encoder types share the same activation()/embed()/forward()
    signature, so nothing downstream (loss.py, steering.py, training.py) needs to know which one
    is in use.

    predictor_type (linear_selector/product_key only): "prototype" (default -- PrototypePredictor,
    scores against concept_prototypes.json's LLM-generated sentences) or "lifted_tokens"
    (LiftedTokenPredictor, scores against each concept's own top lifted tokens instead --
    proto_token_ids must then come from data.utils.load_lifted_token_prototypes, not
    load_concept_prototype_tokens). Orthogonal to which selector is in use, same as the
    selector/predictor split itself.
    """
    if known_encoder_type == "dense":
        return SparseEmbeddingToConcept(d, n, top_k=top_k_known)

    if proto_token_ids is None or backbone is None:
        raise ValueError(
            f"known_encoder_type={known_encoder_type!r} requires proto_token_ids and "
            "backbone (see babysteerling.data.utils.load_concept_prototype_tokens)"
        )
    if known_encoder_type == "prototype_attention":
        return PrototypeCrossAttention(
            d, n, proto_token_ids, backbone, chunk_size=chunk_size, key_dim=key_dim,
            top_k=top_k_known, use_checkpoint=use_checkpoint,
        )
    if known_encoder_type in _SELECTOR_TYPES:
        # the selector (which candidates) and predictor (how well they match) are independent:
        # swapping one doesn't change the other
        if known_encoder_type == "linear_selector":
            selector = LinearSelector(d, n, candidates_per_token=candidates_per_token)
        else:
            selector = ProductKeySelector(d, n, topk_axis=topk_axis, key_dim=key_dim)
        if predictor_type == "prototype":
            predictor = PrototypePredictor(d, n, proto_token_ids, backbone, key_dim=key_dim)
        elif predictor_type == "lifted_tokens":
            predictor = LiftedTokenPredictor(
                d, n, proto_token_ids, backbone, key_dim=key_dim, top_k=lifted_top_k,
            )
        else:
            raise ValueError(
                f"unknown predictor_type: {predictor_type!r} (expected {_PREDICTOR_TYPES!r})"
            )
        return PrototypeConceptEncoder(d, n, selector, predictor)
    raise ValueError(
        f"unknown known_encoder_type: {known_encoder_type!r} "
        "(expected 'dense', 'linear_selector', 'product_key', or 'prototype_attention')"
    )


class ResidualModule(nn.Module):
    """epsilon = h - k - u: whatever the two concept heads don't reconstruct.

    Dropout on epsilon discourages the model from routing information through this
    uninterpretable channel just because it's easier than using a concept. It only bites when the
    whole residual is gone, since h_bar is then k + u and those two have to carry the
    sequence on their own. Per-element dropout doesn't achieve that: it zeroes 10% of individual
    numbers and rescales the rest by 1/(1-p), so h_bar stays close to h on every sequence. Dropping
    the residual for 10% of *sequences* instead, all or nothing.
    """

    def __init__(self, p_epsilon=0.1):
        super().__init__()
        self.p_epsilon = p_epsilon

    def forward(self, h, k, u):
        epsilon = h - k - u  # shape: [B, T, d], same shape as h
        if self.training and self.p_epsilon > 0:
            keep = (torch.rand(epsilon.shape[0], 1, 1, device=epsilon.device)
                    >= self.p_epsilon).to(epsilon.dtype)  # shape: [B, 1, 1], one draw per sequence
            epsilon = epsilon * keep
        return epsilon


class ConceptBottleneck(nn.Module):
    """Composes the three heads: h_bar = k + u + epsilon.

    known_encoder_type picks self.known: "dense" (default, a small MLP over the known concept
    library), "linear_selector"/"product_key" (score a subset of concepts against their
    prototype texts), or "prototype_attention" (score all concepts against their prototypes).
    The last three need proto_token_ids and the model's own backbone; see
    babysteerling.nn.prototype.
    """

    def __init__(self, d, n, unknown_ratio=3, p_epsilon=0.1, unknown_rank=None,
                 top_k_known=None, top_k_unknown=None, known_encoder_type="dense",
                 proto_token_ids=None, backbone=None, topk_axis=5, chunk_size=4096,
                 key_dim=None, use_checkpoint=True, candidates_per_token=25,
                 predictor_type="prototype", lifted_top_k=5):
        super().__init__()
        self.n = n
        self.m = unknown_ratio * n
        self.known = _build_known_encoder(
            known_encoder_type, d, n, top_k_known, proto_token_ids, backbone,
            topk_axis, chunk_size, key_dim, use_checkpoint, candidates_per_token,
            predictor_type=predictor_type, lifted_top_k=lifted_top_k,
        )
        self.unknown = SparseEmbeddingToConcept(d, self.m, rank=unknown_rank, top_k=top_k_unknown)
        self.residual = ResidualModule(p_epsilon)

    def forward(self, h, known_labels=None):
        alpha_k, k = self.known(h)
        # h is detached into the unknown head, so its gradient doesn't reach the backbone and it
        # can't pull h toward "easy to reconstruct"
        alpha_u, u = self.unknown(h.detach())

        k_gt, u_target = None, None
        if known_labels is not None:
            k_gt = self.known.ground_truth_embedding(known_labels)  # shape: [B, T, d]
            u_target = h - k_gt  # shape: [B, T, d], target for the unknown head's reconstruction loss

        # teacher forcing: feed the LM head the ground-truth concept embedding while training, so it
        # isn't learning from an unconverged concept head. Gated on self.training, since
        # build_supervision provides known_labels at eval too and without the gate validation would
        # be handed the correct concepts, making val/lm measure an easier task than the
        # no-bottleneck baseline does.
        k_lm = k_gt if (self.training and k_gt is not None) else k

        epsilon = self.residual(h, k_lm, u)  # shape: [B, T, d]
        h_bar = k_lm + u + epsilon  # shape: [B, T, d], exactly reconstructs h in expectation

        intermediates = {
            'alpha_k': alpha_k, 'alpha_u': alpha_u, 'k': k, 'u': u,  # predicted: what the losses score
            'k_lm': k_lm,  # what actually formed h_bar (ground truth while training)
            'k_gt': k_gt, 'u_target': u_target, 'epsilon': epsilon,
        }
        return h_bar, intermediates
