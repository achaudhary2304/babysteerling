"""Steering: push the model's output toward a concept by adding that concept's learned
embedding direction into the hidden state, at inference time or during training. No prompting,
no weight updates. Implements Guide Labs' Section 6.2 (see NOTICE): injection h_t^(l) +=
gamma*e_c for l >= L_inj (Eq. 18), calibrated strength gamma = tau/peak(e_c) (Eq. 19), and
ReLU-gated logit suppression (Eq. 20-21). Also implements Section 10.2.4's steering-training
(respond/express losses, Eq. 31-32), using the token-level concept attribution the atlas
pipeline already computes (Section 4.4's lift metric), instead of a new LLM-tagging stage.

InterventionModule works on any module that returns a [B, T, F] or [B, F] tensor, gated by any
pytorch_concepts Strategy/Policy pair. It doesn't care if it's wrapping a transformer block
(hidden-state steering) or a concept head's activation() (concept-level intervention). Plug in a
new model by giving it a single-tensor-in/out submodule; plug in new intervention behavior by
subclassing pytorch_concepts' BaseConceptInterventionStrategy/BaseInterventionPolicy.
"""
import random
from contextlib import contextmanager

import torch
import torch.nn as nn
from torch.nn import functional as F

from torch_concepts.nn import BaseConceptInterventionStrategy, UniformPolicy

from . import diffusion
from .data.utils import build_supervision, get_batch


def _flatten_to_2d(x):
    """[B, T, F] -> [B*T, F] (a 2-D [B, F] input is left as is). Returns (flat, original_shape)
    so the caller can restore it after. pytorch_concepts' build_mask() picks rows by quantile,
    so treating each token as its own row here is correct, not a hack.
    """
    if x.dim() == 2:
        return x, None
    if x.dim() == 3:
        B, T, F_ = x.shape
        return x.reshape(B * T, F_), (B, T, F_)
    raise ValueError(f"expected a 2-D [B,F] or 3-D [B,T,F] tensor, got shape {tuple(x.shape)}")


def _unflatten(x, original_shape):
    if original_shape is None:
        return x
    B, T, F_ = original_shape
    return x.reshape(B, T, F_)


class AddDirectionStrategy(BaseConceptInterventionStrategy):
    """x -> x + gamma*direction (Eq. 18). pytorch_concepts' own strategies replace a value
    outright; this one adds a direction on top of whatever value is already there.
    """

    def __init__(self, direction, gamma):
        super().__init__()
        self.register_buffer('direction', direction.detach())  # buffer, so it moves with .to(device)
        self.gamma = gamma

    def forward(self, x, *args, **kwargs):
        return x + self.gamma * self.direction


class InterventionModule(nn.Module):
    """Wraps a module that outputs [B, T, F] or [B, F], and applies intervention_strategy to
    its output, gated per row by intervention_policy. Not a subclass of pytorch_concepts'
    BaseInterventionModule, since that requires a 2-D [B, F] output and can't wrap a transformer
    block or a per-token activation(). This flattens to 2-D, calls their unmodified
    build_mask(), and flattens back. Works with any of their (or our) Strategy/Policy classes.
    """

    def __init__(self, original_module, intervention_strategy, intervention_policy,
                 sel_idx=None, quantile=1.0, eps=1e-12):
        super().__init__()
        self.original_module = original_module
        self.intervention_strategy = intervention_strategy
        self.intervention_policy = intervention_policy
        self.sel_idx = sel_idx
        self.quantile = quantile
        self.eps = eps

    def forward(self, *args, **kwargs):
        output = self.original_module(*args, **kwargs)
        flat, shape = _flatten_to_2d(output)

        policy_scores = self.intervention_policy(flat)
        # build_mask() uses torch.kthvalue, which MPS doesn't support, so run it on CPU and
        # move the result back. Cheap either way, since policy_scores is only [rows, F].
        mask = self.intervention_policy.build_mask(
            policy_scores.cpu(), sel_idx=self.sel_idx, quantile=self.quantile, eps=self.eps,
        ).to(device=flat.device, dtype=flat.dtype)  # 1 = keep original, 0 = replace (pytorch_concepts' convention)
        intervened = self.intervention_strategy(flat)

        result = flat * mask + intervened * (1.0 - mask)
        return _unflatten(result, shape)


@contextmanager
def steered(model, direction, gamma, inj_layer):
    """Temporarily wraps model.backbone.blocks[inj_layer:] so every position gets the
    AddDirectionStrategy injection (Eq. 18). Restores the original blocks on exit either way,
    so it's safe to use inside eval/metric code.
    """
    blocks = model.backbone.blocks
    layer_ids = range(inj_layer, len(blocks))
    originals = {i: blocks[i] for i in layer_ids}
    try:
        for i in layer_ids:
            blocks[i] = InterventionModule(
                originals[i],
                AddDirectionStrategy(direction, gamma),
                UniformPolicy(),
                quantile=1.0,
            )
        yield model
    finally:
        for i, orig in originals.items():
            blocks[i] = orig


@contextmanager
def injected_at(model, direction, gamma, position_mask, inj_layer):
    """Like steered(), but injects only at a given position_mask [B, T] instead of everywhere.
    Used by run_steering_batch, where the positions come from positions_for_concept and are
    already known, so no Policy is needed to choose them. Uses a plain forward hook instead of
    InterventionModule, since there's no strategy/policy choice to make here.
    """
    def hook(module, inputs, output):
        # These blocks are shared with prototype-based known encoders (nn.prototype.
        # PrototypePredictor), which re-enter them mid-forward to encode prototype texts at a
        # different length. Skip those calls: only the real batch matches position_mask's shape.
        if output.shape[:2] != position_mask.shape:
            return output
        return inject_at_positions(output, direction, gamma, position_mask)

    handles = [block.register_forward_hook(hook) for block in list(model.backbone.blocks)[inj_layer:]]
    try:
        yield model
    finally:
        for h in handles:
            h.remove()


def concept_direction(embedding_table, concept_ids):
    """e_c = K_c / ||K_c|| (Eq. 18). Pass a list of concept_ids to steer toward several concepts
    at once (their embeddings are summed then normalized). embedding_table: a head's own
    [n_or_m, D] embedding, e.g. model.bottleneck.known.K.
    """
    if isinstance(concept_ids, int):
        concept_ids = [concept_ids]
    vec = embedding_table[concept_ids].sum(dim=0)  # shape: [len(concept_ids), D] -> [D]
    return vec / vec.norm()


def calibrate_gamma(direction, head, tau=4.0):
    """gamma = tau / peak(e_c), where peak(e_c) = max_y (e_c . W_y) (Eq. 19). Scales the
    injection so its largest effect on any output logit equals tau, giving comparable steering
    strength across concepts without tuning gamma by hand. Needs head_type="linear", since the
    calibration assumes a real linear projection.
    """
    if head.head_type != "linear":
        raise ValueError("calibrate_gamma requires head_type='linear' (Eq. 19 assumes a linear LM head)")
    alignment = head.head.weight @ direction  # shape: [vocab, D] @ [D] -> [vocab], e_c . W_y for every y
    peak = alignment.max().item()
    return tau / peak


def suppress_logits(logits, direction, head, strength):
    """Suppresses a concept in the output: l_v -= strength*ReLU(a_c[v]) for every logit v,
    where a_c = W.e_c is the concept's alignment with each token (Eq. 20-21). The ReLU stops
    this from boosting anti-aligned tokens instead (see paper Figure 20). Acts directly on
    logits, so it's a plain function rather than going through InterventionModule.
    """
    if head.head_type != "linear":
        raise ValueError("suppress_logits requires head_type='linear'")
    a_c = head.head.weight @ direction  # shape: [vocab, D] @ [D] -> [vocab]
    return logits - strength * F.relu(a_c)


def positions_for_concept(token_window, doc_spans, concept_id, lifted_token_ids):
    """Boolean mask [B, T], True where a token (a) is inside a document tagged with
    concept_id (doc_spans, from data.utils.build_supervision), and (b) is one of concept_id's
    lifted tokens (Section 4.4's lift metric). This is the token-level attribution
    steering-training needs, built entirely from data the pipeline already produces.
    """
    doc_mask = torch.zeros_like(token_window, dtype=torch.bool)
    for batch_idx, tok_start, tok_end, concept_ids in doc_spans:
        if concept_id in concept_ids:
            doc_mask[batch_idx, tok_start:tok_end] = True
    if not lifted_token_ids:
        return torch.zeros_like(doc_mask)
    lifted = torch.as_tensor(list(lifted_token_ids), device=token_window.device)
    return doc_mask & torch.isin(token_window, lifted)


def inject_at_positions(h, direction, gamma, position_mask):
    """h + gamma*direction (Eq. 18), only where position_mask is True. Same idea as
    AddDirectionStrategy, but for when the positions are already known instead of chosen by a
    Policy (see injected_at()).
    """
    delta = gamma * direction  # shape: [D]
    return h + position_mask.unsqueeze(-1).to(h.dtype) * delta


def sample_steering_target(doc_spans, lifted_tokens):
    """Picks which concept to steer toward this step: a random choice among concepts present
    in this batch that also have at least one lifted token. This decides which concept to
    target, a different question than a Policy answers (it picks positions within an
    already-chosen target).
    """
    candidates = sorted({c for _, _, _, concept_ids in doc_spans for c in concept_ids if lifted_tokens.get(c)})
    if not candidates:
        return None
    return random.choice(candidates)


def respond_loss(alpha_k, concept_id, position_mask):
    """Eq. 31: pushes the injected concept's own activation k_{c,t} toward 1 at its attributed
    positions, training the concept module to notice the concept it was just steered toward.
    k: the known head's activation [B, T, n] from a post-injection forward pass. position_mask:
    [B, T] bool, from positions_for_concept.
    """
    if position_mask.sum() == 0:
        return torch.tensor(0.0, device=alpha_k.device)
    alpha_c = alpha_k[..., concept_id]  # shape: [B, T, n] -> [B, T]
    return -torch.log(alpha_c[position_mask].clamp(1e-6, 1.0)).mean()


def express_loss(logits, lifted_token_ids, position_mask):
    """Eq. 32: pushes the model's output distribution, not just its internal activation,
    toward the concept's lifted tokens at its attributed positions.
    """
    if position_mask.sum() == 0 or not lifted_token_ids:
        return torch.tensor(0.0, device=logits.device)
    probs = F.softmax(logits, dim=-1)  # shape: [B, T, vocab]
    lifted = torch.as_tensor(list(lifted_token_ids), device=logits.device)
    mass = probs.index_select(-1, lifted).sum(dim=-1)  # shape: [B, T, vocab] -> [B, T]
    return -torch.log(mass[position_mask].clamp(1e-6, 1.0)).mean()


@torch.no_grad()
def steering_ability_metrics(model, xb, doc_spans, lifted_tokens, inj_layer=1, tau=4.0):
    """Two cheap steering-ability checks, no LLM judge needed: after injecting a sampled
    concept everywhere via steered(), does the concept's own activation rise (respond_delta),
    and does the output shift toward its lifted tokens (output_change_delta)? Both come from
    the same before/after pair of forward passes, so they're directly comparable. Returns None
    if no concept in this batch has lifted tokens to steer toward.
    """
    concept_id = sample_steering_target(doc_spans, lifted_tokens)
    if concept_id is None:
        return None
    direction = concept_direction(model.bottleneck.known.K, concept_id)
    gamma = calibrate_gamma(direction, model.head, tau)
    lifted = torch.as_tensor(lifted_tokens[concept_id], device=xb.device)

    logits_before, intermediates_before = model(xb)
    alpha_before = intermediates_before['alpha_k'][..., concept_id].mean()
    mass_before = F.softmax(logits_before, dim=-1).index_select(-1, lifted).sum(dim=-1).mean()

    with steered(model, direction, gamma, inj_layer):
        logits_after, intermediates_after = model(xb)
    alpha_after = intermediates_after['alpha_k'][..., concept_id].mean()
    mass_after = F.softmax(logits_after, dim=-1).index_select(-1, lifted).sum(dim=-1).mean()

    return {
        'respond_delta': (alpha_after - alpha_before).item(),
        'output_change_delta': (mass_after - mass_before).item(),
    }


def run_steering_batch(model, tokens, doc_records, doc_starts, n_train, n_concepts, split,
                        block_size, batch_size, device, lifted_tokens, lambda_respond=1.0,
                        lambda_express=1.0, inj_layer=1, tau=4.0, is_diffusion=False,
                        mask_token_id=None, diff_block_len=None):
    """One steering-training step (Section 10.2.4): pick a target concept present in this
    batch, inject its direction at its attributed positions through the backbone, and score
    with L_respond/L_express on top of the normal LM loss. Concept/reconstruction/independence
    losses are left out here, matching the paper's steering phase. Branches on is_diffusion the
    same way run_batch/run_diffusion_batch do.
    """
    xb, yb, starts = get_batch(tokens, split, block_size, batch_size, n_train, device)
    doc_spans, _ = build_supervision(doc_records, doc_starts, starts, block_size, n_concepts, device)

    if is_diffusion:
        x_t, corrupt_mask = diffusion.corrupt(xb, mask_token_id, diff_block_len)
        model_input, lm_target, lm_mask = x_t, xb, corrupt_mask
    else:
        model_input, lm_target, lm_mask = xb, yb, None

    def lm_loss_from(logits):
        B, T, C = logits.shape
        if lm_mask is None:
            return F.cross_entropy(logits.view(B * T, C), lm_target.view(B * T))
        if lm_mask.sum() == 0:
            return torch.tensor(0.0, device=device)
        return F.cross_entropy(logits[lm_mask], lm_target[lm_mask])

    concept_id = sample_steering_target(doc_spans, lifted_tokens)
    if concept_id is None:
        # nothing steerable in this batch; fall back to a plain LM step instead of stalling
        logits, _ = model(model_input)
        lm_loss = lm_loss_from(logits)
        return lm_loss, {'total': lm_loss.item(), 'lm': lm_loss.item(), 'respond': 0.0, 'express': 0.0, 'concept_id': None}

    direction = concept_direction(model.bottleneck.known.K, concept_id)
    gamma = calibrate_gamma(direction, model.head, tau)
    position_mask = positions_for_concept(xb, doc_spans, concept_id, lifted_tokens.get(concept_id, []))

    with injected_at(model, direction, gamma, position_mask, inj_layer):
        logits, intermediates = model(model_input)

    lm_loss = lm_loss_from(logits)
    r_loss = respond_loss(intermediates['alpha_k'], concept_id, position_mask)
    e_loss = express_loss(logits, lifted_tokens.get(concept_id, []), position_mask)
    total_loss = lm_loss + lambda_respond * r_loss + lambda_express * e_loss

    components = {
        'total': total_loss.item(), 'lm': lm_loss.item(),
        'respond': r_loss.item(), 'express': e_loss.item(), 'concept_id': concept_id,
    }
    return total_loss, components
