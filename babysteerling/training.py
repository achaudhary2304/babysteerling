"""Generic training-loop utilities: LR schedule, one training/eval batch, and averaged eval.

Plain-argument functions -- no config-framework object -- so this package works whether or not
the caller uses Hydra. experiments/train.py is the Hydra-specific adapter that unpacks a `cfg`
into these calls; anyone using babysteerling directly can call them with ordinary Python values.
"""
import math

import torch

from . import diffusion
from .data.utils import build_supervision, get_batch
from .loss import compute_losses


def get_lr(step, lr, min_lr, warmup_steps, max_steps):
    """Linear warmup then cosine decay down to a floor -- standard LLM-training schedule."""
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


def run_batch(model, tokens, doc_records, doc_starts, n_train, n_concepts, split, block_size,
              batch_size, device, lambda_concept=1.0, lambda_rec=1.0, lambda_indep=1.0):
    """Sample one batch for `split`, run the forward pass, and score it.

    Shared by the training step and the evaluation loop below, so "how a batch turns into a
    loss" is defined in exactly one place.
    """
    xb, yb, starts = get_batch(
        tokens, split, block_size, batch_size, n_train, device,
    )  # xb, yb: [batch_size, block_size]; starts: list[int] of window offsets, one per batch item
    doc_spans, known_labels = build_supervision(
        doc_records, doc_starts, starts, block_size, n_concepts, device,
    )  # known_labels: [batch_size, block_size, n_concepts]

    logits, intermediates = model(xb, known_labels=known_labels)  # logits: [batch_size, block_size, vocab_size]
    total_loss, components = compute_losses(
        logits, yb, intermediates, doc_spans,
        lambda_concept=lambda_concept, lambda_rec=lambda_rec, lambda_indep=lambda_indep,
    )
    return total_loss, components


@torch.no_grad()
def estimate_loss(model, tokens, doc_records, doc_starts, n_train, n_concepts, block_size,
                   batch_size, device, eval_iters, lambda_concept=1.0, lambda_rec=1.0, lambda_indep=1.0):
    """Average the loss (and each component) over several fresh batches per split, so the
    number reported at each eval step is a smoothed estimate rather than one noisy batch."""
    model.eval()
    out = {}
    for split in ('train', 'val'):
        totals = {'total': 0.0, 'lm': 0.0, 'concept': 0.0, 'rec': 0.0, 'indep': 0.0}
        for _ in range(eval_iters):
            _, components = run_batch(
                model, tokens, doc_records, doc_starts, n_train, n_concepts, split, block_size,
                batch_size, device,
                lambda_concept=lambda_concept, lambda_rec=lambda_rec, lambda_indep=lambda_indep,
            )
            for key in totals:
                totals[key] += components[key]
        out[split] = {key: value / eval_iters for key, value in totals.items()}
    model.train()
    return out


# ---------------------------------------------------------------------------
# Diffusion-backbone counterparts of the two functions above. Kept as separate functions rather
# than branching inside run_batch/estimate_loss, so the (default) causal path above stays exactly
# as simple as it already is -- these are purely additive, only called when
# cfg.model.backbone_type == "diffusion" (see experiments/train.py).
# ---------------------------------------------------------------------------

def run_diffusion_batch(model, tokens, doc_records, doc_starts, n_train, n_concepts, split,
                         block_size, batch_size, device, mask_token_id, diff_block_len,
                         lambda_concept=1.0, lambda_rec=1.0, lambda_indep=1.0):
    """Sample one batch, corrupt it, run the forward pass, and score it against the masked
    positions only. The diffusion counterpart of run_batch() above -- same shape, different
    corruption process and a masked (rather than every-position) loss.
    """
    x0, _, starts = get_batch(
        tokens, split, block_size, batch_size, n_train, device,
    )  # x0: [batch_size, block_size]; the shift-by-one target from get_batch is unused here --
    # diffusion predicts the *clean* x0 at masked positions, not the next token
    x_t, mask = diffusion.corrupt(x0, mask_token_id, diff_block_len)  # x_t, mask: [batch_size, block_size]

    doc_spans, known_labels = build_supervision(
        doc_records, doc_starts, starts, block_size, n_concepts, device,
    )  # known_labels: [batch_size, block_size, n_concepts]

    logits, intermediates = model(x_t, known_labels=known_labels)  # logits: [batch_size, block_size, vocab_size]
    total_loss, components = compute_losses(
        logits, x0, intermediates, doc_spans,
        lambda_concept=lambda_concept, lambda_rec=lambda_rec, lambda_indep=lambda_indep, mask=mask,
    )
    return total_loss, components


@torch.no_grad()
def estimate_diffusion_loss(model, tokens, doc_records, doc_starts, n_train, n_concepts, block_size,
                             batch_size, device, eval_iters, mask_token_id, diff_block_len,
                             lambda_concept=1.0, lambda_rec=1.0, lambda_indep=1.0):
    """Diffusion counterpart of estimate_loss() above."""
    model.eval()
    out = {}
    for split in ('train', 'val'):
        totals = {'total': 0.0, 'lm': 0.0, 'concept': 0.0, 'rec': 0.0, 'indep': 0.0}
        for _ in range(eval_iters):
            _, components = run_diffusion_batch(
                model, tokens, doc_records, doc_starts, n_train, n_concepts, split, block_size,
                batch_size, device, mask_token_id, diff_block_len,
                lambda_concept=lambda_concept, lambda_rec=lambda_rec, lambda_indep=lambda_indep,
            )
            for key in totals:
                totals[key] += components[key]
        out[split] = {key: value / eval_iters for key, value in totals.items()}
    model.train()
    return out
