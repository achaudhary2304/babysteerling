"""Model architecture: transformer backbone + Steerling concept bottleneck + head.

Independently implements the architecture from Guide Labs' "Scaling Inherently Interpretable
Language Models" (see NOTICE), at a scale that trains in minutes on a laptop.

The concept bottleneck splits the backbone's hidden state into three parts that add back up to
it, before the final projection to vocabulary logits:

    h_bar = k_hat (known concepts) + u_hat (unknown concepts) + epsilon (residual)

Idea: instead of letting the model use its hidden state however it wants, force part of it
through a small set of human-labeled "known" concepts (see babysteerling.data.atlas for how the
concept library is built), part through a larger set of "unknown" concepts the model discovers on
its own, and let a residual catch whatever's left. Since the final head is linear, every output
logit is an exact sum of a known-concept part, an unknown-concept part, and a residual part. That
is what lets us trace a prediction back to specific concepts.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

from .nn.backbone import TransformerModel
from .nn.bottleneck import ConceptBottleneck
from .nn.predictor import LinearEmbeddingToConcept, ReluEmbeddingToConcepts


class SteerlingGPT(nn.Module):
    """Backbone + concept bottleneck + head, as one module.

    One nn.Module instead of three separate objects, so model.parameters() and
    model.state_dict() automatically dedupe the tied embedding/head weight.

    backbone_type="causal" (default): normal next-token attention. backbone_type="diffusion":
    block-causal attention (bidirectional inside a block of `diff_block_len` tokens, causal
    across blocks), for the masked-diffusion objective in babysteerling.diffusion. This class
    only builds the attention mask; corruption and sampling live in babysteerling.diffusion.
    """

    def __init__(self, vocab_size, block_size, n_embed, num_heads, num_kv_heads, n_layers,
                 dropout, n_concepts, unknown_ratio=3, p_epsilon=0.1, unknown_rank=None,
                 top_k_known=None, top_k_unknown=None, head_type="linear", tie_weights=True,
                 head_mlp_hidden=None, backbone_type="causal",
                 known_encoder_type="dense", proto_token_ids=None, topk_axis=5, chunk_size=4096,
                 known_key_dim=None, use_checkpoint=True, candidates_per_token=25,
                 predictor_type="prototype", lifted_top_k=5):
        super().__init__()
        self.block_size = block_size
        self.backbone_type = backbone_type
        self.backbone = TransformerModel(vocab_size, n_embed, block_size, num_heads, n_layers, dropout, num_kv_heads)
        self.bottleneck = ConceptBottleneck(
            n_embed, n_concepts, unknown_ratio=unknown_ratio, p_epsilon=p_epsilon,
            unknown_rank=unknown_rank, top_k_known=top_k_known, top_k_unknown=top_k_unknown,
            known_encoder_type=known_encoder_type, proto_token_ids=proto_token_ids,
            backbone=self.backbone, topk_axis=topk_axis,
            chunk_size=chunk_size, key_dim=known_key_dim, use_checkpoint=use_checkpoint,
            candidates_per_token=candidates_per_token,
            predictor_type=predictor_type, lifted_top_k=lifted_top_k,
        )
        tied_embedding = self.backbone.token_embedding_table.weight if tie_weights else None
        if head_type == "linear":
            self.head = LinearEmbeddingToConcept(
                n_embed, vocab_size, tie_weights=tie_weights, tied_embedding=tied_embedding,
            )
        elif head_type == "mlp":
            self.head = ReluEmbeddingToConcepts(
                n_embed, vocab_size, mlp_hidden=head_mlp_hidden,
            )

        attn_mask = None
        # non-persistent: it's cheap to rebuild and shouldn't be saved into/loaded from checkpoints
        self.register_buffer("attn_mask", attn_mask, persistent=False)

    def forward(self, idx, known_labels=None):
        h = self.backbone(idx, attn_mask=self.attn_mask)  # shape: [B, T, n_embed]
        h_bar, intermediates = self.bottleneck(h, known_labels=known_labels)
        logits = self.head(h_bar)  # shape: [B, T, vocab_size]
        return logits, intermediates

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressive sampling, one token at a time, cropping context to block_size."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]  # shape: [B, <=block_size], keep only the last block_size tokens
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # shape: [B, T, vocab] -> [B, vocab], last-position logits only

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))  # shape: [B, top_k], top-k logit values
                logits[logits < v[:, [-1]]] = float('-inf')  # mask out everything below the k-th largest logit

            probs = F.softmax(logits, dim=-1)  # shape: [B, vocab]
            idx_next = torch.multinomial(probs, num_samples=1)  # shape: [B, 1]
            idx = torch.cat((idx, idx_next), dim=1)  # shape: [B, T] -> [B, T+1]
        return idx


class SteerlingDiffusion(SteerlingGPT):
    """SteerlingGPT with a diffusion-specific generate(). Everything else (forward, backbone,
    bottleneck, head) is unchanged; only the sampling procedure differs.
    """

    def __init__(self, vocab_size, block_size, n_embed, num_heads, num_kv_heads, n_layers,
                 dropout, n_concepts, unknown_ratio=3, p_epsilon=0.1, unknown_rank=None,
                 top_k_known=None, top_k_unknown=None, head_type="linear", tie_weights=True,
                 head_mlp_hidden=None, backbone_type="causal", diff_block_len=None,
                 known_encoder_type="dense", proto_token_ids=None, topk_axis=5, chunk_size=4096,
                 known_key_dim=None, use_checkpoint=True, candidates_per_token=25,
                 predictor_type="prototype", lifted_top_k=5):
        super().__init__(
            vocab_size=vocab_size,
            block_size=block_size,
            n_embed=n_embed,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            n_layers=n_layers,
            dropout=dropout,
            n_concepts=n_concepts,
            unknown_ratio=unknown_ratio,
            p_epsilon=p_epsilon,
            unknown_rank=unknown_rank,
            top_k_known=top_k_known,
            top_k_unknown=top_k_unknown,
            head_type=head_type,
            tie_weights=tie_weights,
            head_mlp_hidden=head_mlp_hidden,
            backbone_type=backbone_type,
            known_encoder_type=known_encoder_type,
            proto_token_ids=proto_token_ids,
            topk_axis=topk_axis,
            chunk_size=chunk_size,
            known_key_dim=known_key_dim,
            use_checkpoint=use_checkpoint,
            candidates_per_token=candidates_per_token,
            predictor_type=predictor_type,
            lifted_top_k=lifted_top_k,
        )
        assert diff_block_len is not None, "diff_block_len is required when backbone_type='diffusion'"
        from .diffusion import build_block_causal_mask  # local import: diffusion.py doesn't need to import nn.py
        attn_mask = build_block_causal_mask(block_size, diff_block_len)

        # non-persistent: it's cheap to rebuild and shouldn't be saved into/loaded from checkpoints
        self.register_buffer("attn_mask", attn_mask, persistent=False)

    @torch.no_grad()
    def generate(model, mask_token_id, seq_len, vocab_size, gen_steps=32, temperature=0.8, top_k=50):
        """Simple random-remasking sampler for the diffusion model, just to sanity-check output.
        Not the paper's efficient block-wise, KV-cached inference. Works on any model with
        SteerlingGPT's forward(idx) -> (logits, intermediates) interface.
        """
        device = next(model.parameters()).device
        model.eval()
        x = torch.full((1, seq_len), mask_token_id, dtype=torch.long,
                       device=device)  # shape: [1, seq_len], start fully masked
        masked = torch.ones_like(x, dtype=torch.bool)  # shape: [1, seq_len], tracks which positions are still masked

        for step in range(1, gen_steps + 1):
            logits, _ = model(x)  # shape: [1, seq_len, vocab_size]
            target_masked_count = round(
                seq_len * (1 - step / gen_steps))  # shrink the masked budget linearly over gen_steps

            probs = F.softmax(logits / temperature, dim=-1)  # shape: [1, seq_len, vocab_size]
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)  # shape: [1, seq_len, top_k]
                probs = torch.where(logits < v[..., [-1]], torch.zeros_like(probs), probs)  # zero out everything below the k-th largest logit
                probs = probs / probs.sum(dim=-1, keepdim=True)  # renormalize after truncation
            sampled = torch.multinomial(probs.view(-1, vocab_size), 1).view(1,
                                                                            seq_len)  # shape: [seq_len, vocab_size] -> [seq_len, 1] -> [1, seq_len]

            masked_positions = masked[0].nonzero(as_tuple=True)[
                0]  # shape: [n_still_masked], indices of masked positions
            num_to_reveal = max(len(masked_positions) - target_masked_count, 0)
            if num_to_reveal > 0:
                # reveal a random subset of currently-masked positions, not necessarily the most
                # confident ones: simplest possible sampler, good enough for a sanity check
                reveal_idx = masked_positions[torch.randperm(len(masked_positions))[:num_to_reveal]]
                x[0, reveal_idx] = sampled[0, reveal_idx]
                masked[0, reveal_idx] = False

        model.train()
        return x[0].tolist()


def build_model(vocab_size, n_concepts, block_size, n_embed=128, num_heads=4, num_kv_heads=2,
                 n_layers=4, dropout=0.2, unknown_ratio=3, p_epsilon=0.1, unknown_rank=None,
                 top_k_known=None, top_k_unknown=None, head_type="linear", tie_weights=True,
                 head_mlp_hidden=None, backbone_type="causal", diff_block_len=None,
                 known_encoder_type="dense", proto_token_ids=None, topk_axis=5, chunk_size=4096,
                 known_key_dim=None, use_checkpoint=True, candidates_per_token=25,
                 predictor_type="prototype", lifted_top_k=5):
    """Builds a SteerlingGPT from plain keyword arguments. No config object needed, so it works
    the same whether the caller uses Hydra or not (see experiments/train.py for the adapter that
    unpacks `cfg.model` into this call).

    known_encoder_type: "dense" (default, a small MLP over the known concept library),
    "linear_selector" or "product_key" (score a subset of concepts against their prototype
    texts), or "prototype_attention" (score all concepts against their prototypes, chunked). See
    babysteerling.nn.prototype. The last three need proto_token_ids (see
    babysteerling.data.utils.load_concept_prototype_tokens).

    predictor_type (linear_selector/product_key only): "prototype" (default) or "lifted_tokens"
    (score against each concept's own top lifted tokens instead -- proto_token_ids must then come
    from babysteerling.data.utils.load_lifted_token_prototypes). See nn.bottleneck's
    _build_known_encoder.
    """
    if backbone_type.lower() == "causal":
        return SteerlingGPT(
            vocab_size=vocab_size,
            block_size=block_size,
            n_embed=n_embed,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            n_layers=n_layers,
            dropout=dropout,
            n_concepts=n_concepts,
            unknown_ratio=unknown_ratio,
            p_epsilon=p_epsilon,
            unknown_rank=unknown_rank,
            top_k_known=top_k_known,
            top_k_unknown=top_k_unknown,
            head_type=head_type,
            tie_weights=tie_weights,
            head_mlp_hidden=head_mlp_hidden,
            backbone_type=backbone_type,
            known_encoder_type=known_encoder_type,
            proto_token_ids=proto_token_ids,
            topk_axis=topk_axis,
            chunk_size=chunk_size,
            known_key_dim=known_key_dim,
            use_checkpoint=use_checkpoint,
            candidates_per_token=candidates_per_token,
            predictor_type=predictor_type,
            lifted_top_k=lifted_top_k,
        )
    elif backbone_type.lower() == "diffusion":
        return SteerlingDiffusion(
            vocab_size=vocab_size,
            block_size=block_size,
            n_embed=n_embed,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            n_layers=n_layers,
            dropout=dropout,
            n_concepts=n_concepts,
            unknown_ratio=unknown_ratio,
            p_epsilon=p_epsilon,
            unknown_rank=unknown_rank,
            top_k_known=top_k_known,
            top_k_unknown=top_k_unknown,
            head_type=head_type,
            tie_weights=tie_weights,
            head_mlp_hidden=head_mlp_hidden,
            backbone_type=backbone_type,
            diff_block_len=diff_block_len,
            known_encoder_type=known_encoder_type,
            proto_token_ids=proto_token_ids,
            topk_axis=topk_axis,
            chunk_size=chunk_size,
            known_key_dim=known_key_dim,
            use_checkpoint=use_checkpoint,
            candidates_per_token=candidates_per_token,
            predictor_type=predictor_type,
            lifted_top_k=lifted_top_k,
        )
    else:
        raise ValueError(f"Unsupported backbone_type: {backbone_type}. Supported types are 'causal' and 'diffusion'.")
