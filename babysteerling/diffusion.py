"""Masked-diffusion training objective: everything specific to the `backbone_type="diffusion"`
option in nn.py's SteerlingGPT, kept out of nn.py/training.py so those stay simple for the
(default) causal case. Nothing here is required to use or understand the causal path.

Ported from 3_steerling/steerling.py's flat-script implementation of Guide Labs' Causal
Diffusion architecture (see NOTICE for attribution), reorganized as free functions that operate
on a plain SteerlingGPT via its public forward(idx) interface -- this module (not nn.py) is the
only place that needs to know about corruption, masking schedules, or denoising sampling.
"""
import torch
from torch.nn import functional as F


def build_block_causal_mask(block_size, diff_block_len, device=None):
    """Bidirectional within a block of `diff_block_len` tokens, causal across blocks (paper
    Figure 16d). Built once at model-construction time in SteerlingGPT and reused every step."""
    assert block_size % diff_block_len == 0, "block_size must be divisible by diff_block_len"
    block_ids = torch.arange(block_size, device=device) // diff_block_len  # shape: [block_size], each position's block index
    # mask[i, j] True means query i may attend to key j: allowed iff j's block <= i's block
    mask = block_ids.unsqueeze(1) >= block_ids.unsqueeze(0)  # shape: [block_size] -> [block_size, block_size]
    return mask


def sample_noise_levels(num_blocks, device):
    """Per-block noise level t_b ~ U(0,1) (paper Section 5.4.2's per-block masking schedule).
    The paper's final model instead uses a moving Gaussian curriculum (center 0.2->0.8,
    sigma=0.3); swap this out for that later without touching corrupt()."""
    return torch.rand(num_blocks, device=device)


def corrupt(x0, mask_token_id, diff_block_len):
    """Independently mask tokens with a per-block noise level.

    x0: LongTensor [B, T], T must be divisible by diff_block_len.
    Returns (x_t, mask): corrupted sequence and boolean mask of replaced positions -- `mask` is
    what loss.compute_losses(..., mask=mask) needs to restrict the LM/reconstruction losses to
    the positions that actually had something to predict.
    """
    B, T = x0.shape
    assert T % diff_block_len == 0, "sequence length must be divisible by the diffusion block length"
    num_blocks = T // diff_block_len

    t = sample_noise_levels(B * num_blocks, x0.device).view(B, num_blocks)  # shape: [B*num_blocks] -> [B, num_blocks]
    t_per_token = t.repeat_interleave(diff_block_len, dim=1)  # shape: [B, num_blocks] -> [B, T], broadcast each block's t to its tokens

    mask = torch.rand(B, T, device=x0.device) < t_per_token  # shape: [B, T], True where this token gets masked
    x_t = x0.clone()
    x_t[mask] = mask_token_id
    return x_t, mask


def ensure_mask_token(tokenizer, mask_token="[MASK]"):
    """Add `[MASK]` as a special token if the tokenizer doesn't already have one, and return its
    id. Call this before computing vocab_size / building the model, so the embedding table and
    LM head are sized to include it. Kept out of data/utils.py's generic load_tokenizer() since
    it's only needed for the diffusion backbone.
    """
    if tokenizer.token_to_id(mask_token) is None:
        tokenizer.add_special_tokens([mask_token])
    return tokenizer.token_to_id(mask_token)


@torch.no_grad()
def generate(model, mask_token_id, seq_len, vocab_size, gen_steps=32, temperature=0.8, top_k=50):
    """Basic random-remasking MDM sampler for sanity-checking output only -- not the paper's
    efficient block-wise KV-cached inference procedure (out of scope here). Operates on any
    model exposing SteerlingGPT's forward(idx) -> (logits, intermediates) interface; doesn't
    need to know anything about the concept bottleneck.
    """
    device = next(model.parameters()).device
    model.eval()
    x = torch.full((1, seq_len), mask_token_id, dtype=torch.long, device=device)  # shape: [1, seq_len], start fully masked
    masked = torch.ones_like(x, dtype=torch.bool)  # shape: [1, seq_len], tracks which positions are still masked

    for step in range(1, gen_steps + 1):
        logits, _ = model(x)  # shape: [1, seq_len, vocab_size]
        target_masked_count = round(seq_len * (1 - step / gen_steps))  # shrink the masked budget linearly over gen_steps

        probs = F.softmax(logits / temperature, dim=-1)  # shape: [1, seq_len, vocab_size]
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)  # shape: [1, seq_len, top_k]
            probs = torch.where(logits < v[..., [-1]], torch.zeros_like(probs), probs)  # zero out everything below the k-th largest logit
            probs = probs / probs.sum(dim=-1, keepdim=True)  # renormalize after truncation
        sampled = torch.multinomial(probs.view(-1, vocab_size), 1).view(1, seq_len)  # shape: [seq_len, vocab_size] -> [seq_len, 1] -> [1, seq_len]

        masked_positions = masked[0].nonzero(as_tuple=True)[0]  # shape: [n_still_masked], indices of masked positions
        num_to_reveal = max(len(masked_positions) - target_masked_count, 0)
        if num_to_reveal > 0:
            # reveal a random subset of currently-masked positions (not necessarily the most
            # confident ones) -- simplest possible sampler, good enough for a sanity check
            reveal_idx = masked_positions[torch.randperm(len(masked_positions))[:num_to_reveal]]
            x[0, reveal_idx] = sampled[0, reveal_idx]
            masked[0, reveal_idx] = False

    model.train()
    return x[0].tolist()
