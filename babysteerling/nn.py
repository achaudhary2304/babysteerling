"""Model architecture: causal transformer backbone + Steerling concept bottleneck + head.

Independently implements the architecture described in Guide Labs' technical report "Scaling
Inherently Interpretable Language Models" (see the project's NOTICE for attribution) at a scale
that trains in minutes on a laptop. The concept bottleneck decomposes the backbone's hidden state
into three additive, inspectable pieces before the final projection to vocabulary logits:

    h_bar = k_hat (known concepts) + u_hat (unknown concepts) + epsilon (residual)

Intuition: instead of letting the model use its hidden state however it wants, we force part of
it to route through a small set of human-labeled "known" concepts (supervised by a concept
library -- see babysteerling.data.atlas for how to build one), part through a larger set of free
"unknown" concepts the model discovers on its own, and let a residual mop up whatever's left.
Because the final head is linear, every output logit is then an exact sum of a known-concept
contribution, an unknown-concept contribution, and a residual contribution -- which is what makes
the model's predictions attributable back to specific concepts.
"""
import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class MultiHeadAttention(nn.Module):
    """Grouped-query causal self-attention (num_kv_heads < num_heads shares key/value heads
    across multiple query heads, trading a little quality for a smaller KV cache)."""

    def __init__(self, num_heads, head_size, n_embed, dropout, num_kv_heads=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads or num_heads
        assert num_heads % self.num_kv_heads == 0

        self.query = nn.Linear(n_embed, n_embed, bias=False)
        self.key = nn.Linear(n_embed, self.num_kv_heads * head_size, bias=False)
        self.value = nn.Linear(n_embed, self.num_kv_heads * head_size, bias=False)
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout_p = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape  # batch, sequence length, n_embed

        # project then split into per-head slices: [B, T, C] -> [B, T, heads, head_size] -> [B, heads, T, head_size]
        q = self.query(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k = self.key(x).view(B, T, self.num_kv_heads, self.head_size).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_kv_heads, self.head_size).transpose(1, 2)

        # duplicate each kv head across the query heads that share it: [B, num_kv_heads, T, head_size] -> [B, num_heads, T, head_size]
        repeat_factor = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)

        # fused causal attention kernel; is_causal=True masks each position to only see itself and earlier tokens
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )  # shape: [B, num_heads, T, head_size]

        out = out.transpose(1, 2).contiguous().view(B, T, C)  # merge heads back: -> [B, T, C]
        out = self.proj(out)
        out = self.resid_dropout(out)
        return out


class FeedForward(nn.Module):
    """SwiGLU MLP: gated activation tends to outperform plain ReLU/GELU MLPs at equal params."""

    def __init__(self, n_embed, dropout):
        super().__init__()
        hidden_dim = int(4 * n_embed * 2 / 3)  # keep param count close to a standard 4x MLP despite the extra gate projection
        self.w1 = nn.Linear(n_embed, hidden_dim, bias=False)
        self.w2 = nn.Linear(n_embed, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, n_embed, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = F.silu(self.w1(x)) * self.w2(x)  # shape: [B, T, hidden_dim], elementwise gate
        out = self.w3(out)  # shape: [B, T, hidden_dim] -> [B, T, n_embed]
        out = self.dropout(out)
        return out


class Block(nn.Module):
    """One pre-norm transformer block: attention sub-layer, then feedforward sub-layer."""

    def __init__(self, n_embed, num_heads, dropout, num_kv_heads):
        super().__init__()
        head_size = n_embed // num_heads
        self.sa_head = MultiHeadAttention(num_heads, head_size, n_embed, dropout, num_kv_heads=num_kv_heads)
        self.ffwd = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = self.ln1(x)
        x = x + self.sa_head(x)  # residual connection around attention
        x = self.ln2(x)
        x = x + self.ffwd(x)  # residual connection around the feedforward
        return x


class TransformerModel(nn.Module):
    """Backbone only: token/position embeddings through the transformer stack.

    Stops before any LM head on purpose -- in this architecture the head applies to the
    *bottlenecked* hidden state (see ConceptBottleneck below), not this raw backbone output.
    """

    def __init__(self, vocab_size, n_embed, block_size, num_heads, n_layers, dropout, num_kv_heads):
        super().__init__()
        self.block_size = block_size
        self.n_layers = n_layers
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, num_heads, dropout, num_kv_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embed)

        self.apply(self._init_weights)
        # scale down residual-stream-writing projections so the residual stream doesn't blow up
        # in variance as depth increases (standard GPT-2-style init trick)
        for name, p in self.named_parameters():
            if name.endswith('proj.weight') or name.endswith('w3.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        token_emb = self.token_embedding_table(idx)  # shape: [B, T] -> [B, T, n_embed]
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # shape: [T, n_embed]
        x = token_emb + pos_emb  # broadcast add: [B, T, n_embed] + [T, n_embed] -> [B, T, n_embed]
        x = self.blocks(x)
        x = self.ln_f(x)
        return x  # shape: [B, T, n_embed], hidden state ready for the concept bottleneck


def sparsify_top_k(activations, k):
    """Zero out every activation except the top-k per token (optional, off by default).

    Intuition: forces each token to "explain itself" via a small number of active concepts
    instead of a dense mixture, which is closer to how a human would describe a piece of text
    (a few salient concepts, not a weighted blend of the entire library).
    """
    if k is None or k >= activations.shape[-1]:
        return activations
    top_vals, top_idx = torch.topk(activations, k, dim=-1)  # shape: [..., n] -> [..., k] (values and their indices)
    sparse = torch.zeros_like(activations)
    sparse.scatter_(-1, top_idx, top_vals)  # write the top-k values back into their original positions, rest stay 0
    return sparse


class SupervisedConceptHead(nn.Module):
    """Known concepts (size n, fixed by an externally-provided concept library).

    A small MLP predicts, for every token, how strongly each of the n known concepts is
    "active" there (sigmoid, so activations are independent probabilities, not a distribution
    over mutually exclusive classes -- multiple concepts can be present at once). Those
    activations then weight a learned embedding per concept, the same way attention weights
    combine value vectors.
    """

    def __init__(self, d, n, hidden_dim=None, top_k=None):
        super().__init__()
        hidden_dim = hidden_dim or d
        self.f = nn.Sequential(nn.Linear(d, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, n))
        self.K = nn.Parameter(torch.randn(n, d) * 0.02)  # known concept embedding table, shape: [n, d]
        self.top_k = top_k

    def forward(self, h):
        k = torch.sigmoid(self.f(h))  # shape: [B, T, d] -> [B, T, n], per-token concept activations in [0, 1]
        k = sparsify_top_k(k, self.top_k)
        k_hat = k @ self.K  # shape: [B, T, n] @ [n, d] -> [B, T, d], weighted sum of concept embeddings
        return k, k_hat

    def ground_truth_embedding(self, known_labels):
        # weighted sum of K by the ground-truth chunk-level labels, broadcast to every token
        # position of the chunk -- this is the reconstruction loss's target for the unknown head
        return known_labels.float() @ self.K  # shape: [B, T, n] @ [n, d] -> [B, T, d]


class UnsupervisedConceptHead(nn.Module):
    """Unknown concepts (size m, typically a multiple of n): free capacity for the model to
    discover its own concepts, unconstrained by the labeled library. Same activation ->
    weighted-embedding mechanism as the known head, with an optional low-rank embedding table
    for when m is large enough that a dense [m, d] table would dominate the parameter count.
    """

    def __init__(self, d, m, hidden_dim=None, rank=None, top_k=None):
        super().__init__()
        hidden_dim = hidden_dim or d
        self.g = nn.Sequential(nn.Linear(d, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, m))
        self.rank = rank
        if rank is None:
            self.U = nn.Parameter(torch.randn(m, d) * 0.02)  # shape: [m, d]
        else:
            # low-rank factorization U = A @ B: cuts params from m*d to rank*(m+d), and turns
            # the per-token [m, d] matmul into two smaller ones -- worthwhile once m >> rank
            self.A = nn.Parameter(torch.randn(m, rank) * 0.02)  # shape: [m, rank]
            self.B = nn.Parameter(torch.randn(rank, d) * 0.02)  # shape: [rank, d]
        self.top_k = top_k

    def _embed(self, u):
        if self.rank is None:
            return u @ self.U  # shape: [B, T, m] @ [m, d] -> [B, T, d]
        return (u @ self.A) @ self.B  # shape: [B, T, m] @ [m, rank] -> [B, T, rank] -> @ [rank, d] -> [B, T, d]

    def forward(self, h):
        u = torch.sigmoid(self.g(h))  # shape: [B, T, d] -> [B, T, m]
        u = sparsify_top_k(u, self.top_k)
        u_hat = self._embed(u)
        return u, u_hat


class ResidualModule(nn.Module):
    """epsilon = h - k_hat - u_hat: whatever the two concept heads fail to reconstruct.

    Dropout on epsilon discourages the model from routing information through this
    uninterpretable channel just because it's easier than going through a concept -- it should
    only carry what genuinely can't be expressed as a combination of concepts.
    """

    def __init__(self, p_epsilon=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p_epsilon)

    def forward(self, h, k_hat, u_hat):
        epsilon = h - k_hat - u_hat  # shape: [B, T, d], same shape as h
        return self.dropout(epsilon)


class ConceptBottleneck(nn.Module):
    """Composes the three heads: h_bar = k_hat + u_hat + epsilon."""

    def __init__(self, d, n, unknown_ratio=3, p_epsilon=0.1, unknown_rank=None,
                 top_k_known=None, top_k_unknown=None):
        super().__init__()
        self.n = n
        self.m = unknown_ratio * n
        self.known = SupervisedConceptHead(d, n, top_k=top_k_known)
        self.unknown = UnsupervisedConceptHead(d, self.m, rank=unknown_rank, top_k=top_k_unknown)
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


class ConceptLMHead(nn.Module):
    """Projects the bottlenecked hidden state to vocabulary logits.

    head_type="linear" (default): a single bias-free Linear layer, optionally weight-tied to
    the backbone's token embedding. Because it's linear with no bias, logits are an *exact*
    linear function of h_bar, so decompose() below can attribute any output logit exactly to
    the known-concept, unknown-concept, and residual contributions that produced it.

    head_type="mlp": a deeper, nonlinear head -- more expressive, but decompose() is then only
    an approximation, since MLP(a+b+c) != MLP(a)+MLP(b)+MLP(c). Kept for experimentation; see
    the docstring on decompose() below.
    """

    def __init__(self, d, vocab_size, head_type="linear", tie_weights=True,
                 tied_embedding=None, mlp_hidden=None):
        super().__init__()
        self.head_type = head_type
        if head_type == "linear":
            self.head = nn.Linear(d, vocab_size, bias=False)
            if tie_weights and tied_embedding is not None:
                self.head.weight = tied_embedding  # share the tensor, not a copy
        elif head_type == "mlp":
            mlp_hidden = mlp_hidden or d
            # deeper, nonlinear head; weight tying isn't meaningful here since the final
            # layer's input space isn't the embedding space
            self.head = nn.Sequential(
                nn.Linear(d, mlp_hidden, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(mlp_hidden, mlp_hidden, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(mlp_hidden, vocab_size, bias=False),
            )
        else:
            raise ValueError(f"unknown head_type: {head_type!r} (expected 'linear' or 'mlp')")

    def forward(self, h_bar):
        return self.head(h_bar)  # shape: [B, T, d] -> [B, T, vocab_size]

    def decompose(self, k_hat, u_hat, epsilon):
        """Split logits into known/unknown/residual contributions.

        Exact (sums to forward(k_hat + u_hat + epsilon)) when head_type="linear", since the head
        is then a single linear map with no bias. Only approximate for head_type="mlp".
        """
        return self.head(k_hat), self.head(u_hat), self.head(epsilon)  # each: [B, T, d] -> [B, T, vocab_size]


class SteerlingGPT(nn.Module):
    """Backbone + concept bottleneck + head, as one module.

    Wrapping everything in a single nn.Module (rather than passing three separate objects
    around) means model.parameters() and model.state_dict() naturally deduplicate the tied
    embedding/head weight via PyTorch's built-in traversal.
    """

    def __init__(self, vocab_size, block_size, n_embed, num_heads, num_kv_heads, n_layers,
                 dropout, n_concepts, unknown_ratio=3, p_epsilon=0.1, unknown_rank=None,
                 top_k_known=None, top_k_unknown=None, head_type="linear", tie_weights=True,
                 head_mlp_hidden=None):
        super().__init__()
        self.block_size = block_size
        self.backbone = TransformerModel(vocab_size, n_embed, block_size, num_heads, n_layers, dropout, num_kv_heads)
        self.bottleneck = ConceptBottleneck(
            n_embed, n_concepts, unknown_ratio=unknown_ratio, p_epsilon=p_epsilon,
            unknown_rank=unknown_rank, top_k_known=top_k_known, top_k_unknown=top_k_unknown,
        )
        tied_embedding = self.backbone.token_embedding_table.weight if tie_weights else None
        self.head = ConceptLMHead(
            n_embed, vocab_size, head_type=head_type, tie_weights=tie_weights,
            tied_embedding=tied_embedding, mlp_hidden=head_mlp_hidden,
        )

    def forward(self, idx, known_labels=None):
        h = self.backbone(idx)  # shape: [B, T, n_embed]
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


def build_model(vocab_size, n_concepts, block_size, n_embed=128, num_heads=4, num_kv_heads=2,
                 n_layers=4, dropout=0.2, unknown_ratio=3, p_epsilon=0.1, unknown_rank=None,
                 top_k_known=None, top_k_unknown=None, head_type="linear", tie_weights=True,
                 head_mlp_hidden=None):
    """Factory: construct a SteerlingGPT from plain keyword arguments (defaults match this
    project's baseline config). Takes no framework-specific config object, so it can be called
    the same way whether or not the caller uses Hydra -- see experiments/train.py for the
    Hydra-config adapter that unpacks a `cfg.model` group into this call.
    """
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
    )
