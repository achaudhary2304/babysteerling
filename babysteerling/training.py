"""Training-loop utilities: LR schedule, one training/eval batch, and averaged eval.

Plain arguments, no config object, so this works whether or not the caller uses Hydra.
experiments/train.py unpacks a Hydra `cfg` into these calls.
"""
import math

import torch

from . import diffusion
from .data.utils import build_supervision, get_batch
from .loss import compute_losses


def get_lr(step, lr, min_lr, warmup_steps, max_steps):
    """Linear warmup, then cosine decay down to a floor."""
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


def concept_contribution(head, intermediates, targets, mask=None):
    """For each target token, what fraction of the |known| + |unknown| + |residual| logit
    magnitude at that token's target id comes from the known-concept path, and separately from
    the unknown-concept path (Section 6.1.2 / Eq. 22). Exact for head_type="linear"; approximate
    for "mlp". Doesn't depend on babysteerling.steering, so it's always computed.
    """
    k_logits, u_logits, eps_logits = head.decompose(intermediates['k_hat'], intermediates['u_hat'], intermediates['epsilon'])
    k_term = k_logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1).abs()    # shape: [B,T,vocab] -> [B,T]
    u_term = u_logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1).abs()
    eps_term = eps_logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1).abs()
    if mask is not None:
        if mask.sum() == 0:
            return 0.0, 0.0
        k_term, u_term, eps_term = k_term[mask], u_term[mask], eps_term[mask]
    denom = k_term + u_term + eps_term + 1e-8
    return (k_term / denom).mean().item(), (u_term / denom).mean().item()


def run_batch(model, tokens, doc_records, doc_starts, n_train, n_concepts, split, block_size,
              batch_size, device, lambda_concept=1.0, lambda_rec=1.0, lambda_indep=1.0,
              use_concept_loss=True):
    """Sample one batch for `split`, run the forward pass, and score it.

    Shared by the training step and the eval loop below, so "how a batch turns into a loss" is
    defined in one place.
    """
    xb, yb, starts = get_batch(
        tokens, split, block_size, batch_size, n_train, device,
    )  # xb, yb: [batch_size, block_size]; starts: list[int] of window offsets, one per batch item
    doc_spans, known_labels = build_supervision(
        doc_records, doc_starts, starts, block_size, n_concepts, device,
    )  # known_labels: [batch_size, block_size, n_concepts]

    logits, intermediates = model(xb, known_labels=known_labels)  # logits: [batch_size, block_size, vocab_size]
    total_loss, components = compute_losses(
        logits, yb, intermediates, doc_spans, known_labels=known_labels,
        lambda_concept=lambda_concept, lambda_rec=lambda_rec, lambda_indep=lambda_indep,
        use_concept_loss=use_concept_loss,
    )
    components['known_contribution'], components['unknown_contribution'] = concept_contribution(
        model.head, intermediates, yb,
    )
    return total_loss, components


@torch.no_grad()
def estimate_loss(model, tokens, doc_records, doc_starts, n_train, n_concepts, block_size,
                   batch_size, device, eval_iters, lambda_concept=1.0, lambda_rec=1.0, lambda_indep=1.0,
                   steering_metrics_fn=None, use_concept_loss=True):
    """Average the loss and each component over several fresh batches per split, so the number
    reported at each eval step is a smoothed estimate, not one noisy batch.

    steering_metrics_fn, if given, is called once per split and its result (or None) is merged
    into that split's output. This is how experiments/train.py adds
    babysteerling.steering's metrics without this module importing steering itself.
    """
    model.eval()
    out = {}
    for split in ('train', 'val'):
        totals = {
            'total': 0.0, 'lm': 0.0, 'lm_accuracy': 0.0, 'concept': 0.0, 'concept_accuracy': 0.0,
            'rec': 0.0, 'indep': 0.0, 'known_contribution': 0.0, 'unknown_contribution': 0.0,
        }
        for _ in range(eval_iters):
            _, components = run_batch(
                model, tokens, doc_records, doc_starts, n_train, n_concepts, split, block_size,
                batch_size, device,
                lambda_concept=lambda_concept, lambda_rec=lambda_rec, lambda_indep=lambda_indep,
                use_concept_loss=use_concept_loss,
            )
            for key in totals:
                totals[key] += components[key]
        out[split] = {key: value / eval_iters for key, value in totals.items()}
        if steering_metrics_fn is not None:
            extra = steering_metrics_fn(split)
            if extra is not None:
                out[split].update(extra)
    model.train()
    return out


# ---------------------------------------------------------------------------
# Diffusion counterparts of the two functions above. Kept separate rather than branching inside
# run_batch/estimate_loss, so the causal path stays simple. Only called when
# cfg.model.backbone_type == "diffusion" (see experiments/train.py).
# ---------------------------------------------------------------------------

def run_diffusion_batch(model, tokens, doc_records, doc_starts, n_train, n_concepts, split,
                         block_size, batch_size, device, mask_token_id, diff_block_len,
                         lambda_concept=1.0, lambda_rec=1.0, lambda_indep=1.0, use_concept_loss=True):
    """Sample one batch, corrupt it, run the forward pass, and score it against the masked
    positions only. The diffusion counterpart of run_batch() above: same shape, but the input
    is corrupted first and the loss only looks at masked positions.
    """
    x0, _, starts = get_batch(
        tokens, split, block_size, batch_size, n_train, device,
    )  # x0: [batch_size, block_size]; get_batch's shift-by-one target is unused here, since
    # diffusion predicts the clean x0 at masked positions, not the next token
    x_t, mask = diffusion.corrupt(x0, mask_token_id, diff_block_len)  # x_t, mask: [batch_size, block_size]

    doc_spans, known_labels = build_supervision(
        doc_records, doc_starts, starts, block_size, n_concepts, device,
    )  # known_labels: [batch_size, block_size, n_concepts]

    logits, intermediates = model(x_t, known_labels=known_labels)  # logits: [batch_size, block_size, vocab_size]
    total_loss, components = compute_losses(
        logits, x0, intermediates, doc_spans, known_labels=known_labels,
        lambda_concept=lambda_concept, lambda_rec=lambda_rec, lambda_indep=lambda_indep, mask=mask,
        use_concept_loss=use_concept_loss,
    )
    components['known_contribution'], components['unknown_contribution'] = concept_contribution(
        model.head, intermediates, x0, mask=mask,
    )
    return total_loss, components


@torch.no_grad()
def estimate_diffusion_loss(model, tokens, doc_records, doc_starts, n_train, n_concepts, block_size,
                             batch_size, device, eval_iters, mask_token_id, diff_block_len,
                             lambda_concept=1.0, lambda_rec=1.0, lambda_indep=1.0,
                             steering_metrics_fn=None, use_concept_loss=True):
    """Diffusion counterpart of estimate_loss() above, same steering_metrics_fn hook included."""
    model.eval()
    out = {}
    for split in ('train', 'val'):
        totals = {
            'total': 0.0, 'lm': 0.0, 'lm_accuracy': 0.0, 'concept': 0.0, 'concept_accuracy': 0.0,
            'rec': 0.0, 'indep': 0.0, 'known_contribution': 0.0, 'unknown_contribution': 0.0,
        }
        for _ in range(eval_iters):
            _, components = run_diffusion_batch(
                model, tokens, doc_records, doc_starts, n_train, n_concepts, split, block_size,
                batch_size, device, mask_token_id, diff_block_len,
                lambda_concept=lambda_concept, lambda_rec=lambda_rec, lambda_indep=lambda_indep,
                use_concept_loss=use_concept_loss,
            )
            for key in totals:
                totals[key] += components[key]
        out[split] = {key: value / eval_iters for key, value in totals.items()}
        if steering_metrics_fn is not None:
            extra = steering_metrics_fn(split)
            if extra is not None:
                out[split].update(extra)
    model.train()
    return out
