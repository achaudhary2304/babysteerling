import torch
import torch.nn as nn

from torch_concepts.nn import BaseConceptLayer


def sparsify_top_k(activations, k):
    """Zeros out every activation except the top-k per token. Off by default.

    Forces each token to explain itself with a few active concepts instead of a dense mixture,
    closer to how a person would describe a piece of text.
    """
    if k is None or k >= activations.shape[-1]:
        return activations
    top_vals, top_idx = torch.topk(activations, k, dim=-1)  # shape: [..., n] -> [..., k] (values and their indices)
    sparse = torch.zeros_like(activations)
    sparse.scatter_(-1, top_idx, top_vals)  # write the top-k values back into their original positions, rest stay 0
    return sparse


class SparseEmbeddingToConcept(BaseConceptLayer):
    """Known-concept encoder: a small MLP scores each concept (sigmoid activation), then a
    learned [m, d] table turns that score vector into an embedding. Set `rank` for a low-rank
    table when m is large enough that a dense [m, d] table would dominate the parameter count.
    """

    def __init__(self, in_embeddings, out_concepts, hidden_dim=None, rank=None, top_k=None):
        super().__init__(
            out_concepts=out_concepts,
            in_concepts=None,  # Concepts come from encoder, not traditional input
            in_embeddings=in_embeddings
        )
        m = self.out_concepts_shape
        d = self.in_embeddings_shape
        hidden_dim = hidden_dim or d
        self.g = nn.Sequential(nn.Linear(d, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, m))
        self.rank = rank
        if rank is None:
            self.K = nn.Parameter(torch.randn(m, d) * 0.02)  # shape: [m, d]
        else:
            # low-rank factorization A @ B: cuts params from m*d to rank*(m+d), worth it once m >> rank
            self.A = nn.Parameter(torch.randn(m, rank) * 0.02)  # shape: [m, rank]
            self.B = nn.Parameter(torch.randn(rank, d) * 0.02)  # shape: [rank, d]
        self.top_k = top_k

    def activation(self, embeddings):
        u = torch.sigmoid(self.g(embeddings))  # shape: [B, T, d] -> [B, T, m]
        return sparsify_top_k(u, self.top_k)

    def embed(self, u):
        if self.rank is None:
            return u @ self.K  # shape: [B, T, m] @ [m, d] -> [B, T, d]
        return (u @ self.A) @ self.B  # shape: [B, T, m] @ [m, rank] -> [B, T, rank] -> @ [rank, d] -> [B, T, d]

    def forward(self, embeddings):
        # split into activation()/embed() so babysteerling.steering's InterventionModule can
        # wrap activation() alone, to intervene on concept activations without touching the
        # embedding step
        u = self.activation(embeddings)
        u_hat = self.embed(u)
        return u, u_hat

    def ground_truth_embedding(self, known_labels):
        # weighted sum of K by the ground-truth labels; this is the reconstruction loss's target
        # for the unknown head
        if self.rank is None:
            return known_labels.float() @ self.K  # shape: [B, T, n] @ [n, d] -> [B, T, d]
        return (known_labels.float() @ self.A) @ self.B
