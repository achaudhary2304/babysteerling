"""Masked-diffusion training objective for backbone_type="diffusion". Kept out of nn.py/
training.py so the default causal path stays simple; you can ignore this module entirely if
you're only using the causal backbone.

Ported from 3_steerling/steerling.py's Causal Diffusion architecture (see NOTICE), as free
functions over SteerlingGPT's plain forward(idx) interface. This is the only module that needs
to know about corruption, masking schedules, or denoising sampling.
"""
import torch
from torch.nn import functional as F


def build_block_causal_mask(block_size, diff_block_len, device=None):
    """Attention mask: bidirectional inside each block of `diff_block_len` tokens, causal across
    blocks (paper Figure 16d). Built once when SteerlingGPT is constructed, reused every step."""
    assert block_size % diff_block_len == 0, "block_size must be divisible by diff_block_len"
    block_ids = torch.arange(block_size, device=device) // diff_block_len  # shape: [block_size], each position's block index
    # mask[i, j] True means query i may attend to key j: allowed iff j's block <= i's block
    mask = block_ids.unsqueeze(1) >= block_ids.unsqueeze(0)  # shape: [block_size] -> [block_size, block_size]
    return mask


def sample_noise_levels(num_blocks, device):
    """Per-block noise level t_b ~ U(0,1) (paper Section 5.4.2). The paper's final model uses a
    moving Gaussian curriculum instead; swap this function for that later without touching
    corrupt()."""
    return torch.rand(num_blocks, device=device)


def corrupt(x0, mask_token_id, diff_block_len, t_min=1e-3):
    """Masks tokens independently, using a per-block noise level.

    x0: LongTensor [B, T], T must be divisible by diff_block_len.
    Returns (x_t, mask, p_mask): the corrupted sequence, a boolean mask of which positions were
    replaced, and each position's masking probability. loss.compute_losses(..., mask=mask,
    mask_weights=p_mask) uses the mask to only score positions that actually had something to
    predict, and p_mask to weight them.

    t is clamped to at least t_min, since 1/p_mask is the ELBO's importance weight and t ~ 0
    would make it explode.
    """
    B, T = x0.shape
    assert T % diff_block_len == 0, "sequence length must be divisible by the diffusion block length"
    num_blocks = T // diff_block_len

    t = sample_noise_levels(B * num_blocks, x0.device).view(B, num_blocks).clamp(min=t_min)  # shape: [B*num_blocks] -> [B, num_blocks]
    t_per_token = t.repeat_interleave(diff_block_len, dim=1)  # shape: [B, num_blocks] -> [B, T], broadcast each block's t to its tokens

    mask = torch.rand(B, T, device=x0.device) < t_per_token  # shape: [B, T], True where this token gets masked
    x_t = x0.clone()
    x_t[mask] = mask_token_id
    return x_t, mask, t_per_token


def ensure_mask_token(tokenizer, mask_token="[MASK]"):
    """Adds `[MASK]` as a special token if the tokenizer doesn't have one, and returns its id.
    Call this before computing vocab_size, so the embedding table and LM head are sized to
    include it. Not in data/utils.py's load_tokenizer() since only the diffusion backbone
    needs it.
    """
    if tokenizer.token_to_id(mask_token) is None:
        tokenizer.add_special_tokens([mask_token])
    return tokenizer.token_to_id(mask_token)
