from torch import nn
from .encoder import SparseEmbeddingToConcept
from .prototype import (
    LinearSelector, ProductKeySelector, PrototypeConceptEncoder, PrototypeCrossAttention,
    PrototypePredictor,
)

_SELECTOR_TYPES = ("linear_selector", "product_key")


def _build_known_encoder(known_encoder_type, d, n, top_k_known, proto_token_ids, backbone,
                          topk_axis, chunk_size, key_dim, use_checkpoint, candidates_per_token):
    """Builds self.known. All four encoder types share the same activation()/embed()/forward()
    signature, so nothing downstream (loss.py, steering.py, training.py) needs to know which one
    is in use.
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
        predictor = PrototypePredictor(d, n, proto_token_ids, backbone, key_dim=key_dim)
        return PrototypeConceptEncoder(d, n, selector, predictor)
    raise ValueError(
        f"unknown known_encoder_type: {known_encoder_type!r} "
        "(expected 'dense', 'linear_selector', 'product_key', or 'prototype_attention')"
    )


class ResidualModule(nn.Module):
    """epsilon = h - k_hat - u_hat: whatever the two concept heads don't reconstruct.

    Dropout on epsilon discourages the model from routing information through this
    uninterpretable channel just because it's easier than using a concept.
    """

    def __init__(self, p_epsilon=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p_epsilon)

    def forward(self, h, k_hat, u_hat):
        epsilon = h - k_hat - u_hat  # shape: [B, T, d], same shape as h
        return self.dropout(epsilon)


class ConceptBottleneck(nn.Module):
    """Composes the three heads: h_bar = k_hat + u_hat + epsilon.

    known_encoder_type picks self.known: "dense" (default, a small MLP over the known concept
    library), "linear_selector"/"product_key" (score a subset of concepts against their
    prototype texts), or "prototype_attention" (score all concepts against their prototypes).
    The last three need proto_token_ids and the model's own backbone; see
    babysteerling.nn.prototype.
    """

    def __init__(self, d, n, unknown_ratio=3, p_epsilon=0.1, unknown_rank=None,
                 top_k_known=None, top_k_unknown=None, known_encoder_type="dense",
                 proto_token_ids=None, backbone=None, topk_axis=5, chunk_size=4096,
                 key_dim=None, use_checkpoint=True, candidates_per_token=25):
        super().__init__()
        self.n = n
        self.m = unknown_ratio * n
        self.known = _build_known_encoder(
            known_encoder_type, d, n, top_k_known, proto_token_ids, backbone,
            topk_axis, chunk_size, key_dim, use_checkpoint, candidates_per_token,
        )
        self.unknown = SparseEmbeddingToConcept(d, self.m, rank=unknown_rank, top_k=top_k_unknown)
        self.residual = ResidualModule(p_epsilon)

    def forward(self, h, known_labels=None):
        k, k_hat = self.known(h)
        u, u_hat = self.unknown(h)

        k_hat_gt, u_hat_gt = None, None
        if known_labels is not None:
            k_hat_gt = self.known.ground_truth_embedding(known_labels)  # shape: [B, T, d]
            u_hat_gt = h - k_hat_gt  # shape: [B, T, d], target for the unknown head's reconstruction loss

        epsilon = self.residual(h, k_hat, u_hat)  # shape: [B, T, d]
        h_bar = k_hat + u_hat + epsilon  # shape: [B, T, d], exactly reconstructs h in expectation

        intermediates = {
            'k': k, 'u': u, 'k_hat': k_hat, 'u_hat': u_hat,
            'k_hat_gt': k_hat_gt, 'u_hat_gt': u_hat_gt, 'epsilon': epsilon,
        }
        return h_bar, intermediates
