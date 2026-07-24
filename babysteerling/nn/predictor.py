from torch import nn
from torch_concepts.nn import BaseConceptLayer


class LinearEmbeddingToConcept(BaseConceptLayer):
    """Projects the bottlenecked hidden state to vocabulary logits.
    """

    def __init__(self, in_embeddings, out_concepts, tie_weights=True, tied_embedding=None):
        super().__init__(
            in_embeddings=in_embeddings,
            out_concepts=out_concepts,
            in_concepts=None
        )
        d = self.in_embeddings_shape
        vocab_size = self.out_concepts_shape
        self.head_type = "linear"
        self.head = nn.Linear(d, vocab_size, bias=False)
        if tie_weights and tied_embedding is not None:
            self.head.weight = tied_embedding  # share the tensor, not a copy

    def forward(self, embeddings):
        return self.head(embeddings)  # shape: [B, T, d] -> [B, T, vocab_size]

    def decompose(self, k_hat, u_hat, epsilon):
        """Split logits into known/unknown/residual contributions.

        Exact (sums to forward(k_hat + u_hat + epsilon)) when head_type="linear", since the head
        is then a single linear map with no bias. Only approximate for head_type="mlp".
        """
        return self.head(k_hat), self.head(u_hat), self.head(epsilon)  # each: [B, T, d] -> [B, T, vocab_size]


class ReluEmbeddingToConcepts(LinearEmbeddingToConcept):
    """Projects the bottlenecked hidden state to vocabulary logits with relu network (locally linear).
    """

    def __init__(self, in_embeddings, out_concepts, mlp_hidden=None):
        super().__init__(
            in_embeddings=in_embeddings,
            out_concepts=out_concepts,
        )
        d = self.in_embeddings_shape
        vocab_size = self.out_concepts_shape
        self.head_type = "non-linear"
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
