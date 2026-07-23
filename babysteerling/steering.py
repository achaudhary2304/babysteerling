"""Steering: control model output at inference time (or during training) by injecting a
concept's own learned embedding direction into the transformer's hidden state -- no prompting,
no weight updates. Implements Section 6.2 of Guide Labs' technical report (see the project's
NOTICE for attribution): injection h_t^(l) += gamma*e_c for l >= L_inj (Eq. 18), calibrated
strength gamma = tau/peak(e_c) (Eq. 19), and ReLU-gated logit suppression (Eq. 20-21). Also
implements Section 10.2.4's steering-training (respond/express losses, Eq. 31-32), using
token-level concept attribution derived from data the atlas pipeline already produces (Section
4.4's lift metric) rather than a new LLM-tagging stage.

Optional module: requires the `steering` extra (`pip install -e ".[steering]"`), which pulls in
pytorch-concepts. Not imported by babysteerling's top-level __init__.py or by training.py, so a
core install never touches this dependency; experiments/train.py only imports this module when
`cfg.steering.enabled` is true.

Modularity: the injection mechanism (InterventionModule) is generic over any module returning a
[B, T, F] or [B, F] tensor, gated by any pytorch_concepts Strategy/Policy pair -- it doesn't know
or care whether it's wrapping a transformer block (hidden-state steering) or a concept head's
activation() (concept-level intervention, e.g. teacher forcing via GroundTruthIntervention). New
models plug in by exposing single-tensor-in/out submodules; new intervention behaviors plug in
by subclassing pytorch_concepts' own BaseConceptInterventionStrategy/BaseInterventionPolicy.
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
    """[B, T, F] -> [B*T, F] (or leaves an already-2-D [B, F] tensor as is). Returns (flat,
    original_shape) so the caller can restore it afterwards. pytorch_concepts' own
    BaseInterventionPolicy.build_mask() is a pure, row-wise quantile selection over its input's
    first dimension -- treating "one row per token" instead of "one row per example" is exactly
    correct for that computation, not a reinterpretation of it, which is what makes reusing it
    unchanged (see InterventionModule) valid for sequence tensors.
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
    """x -> x + gamma*direction (Eq. 18). The one intervention strategy pytorch_concepts doesn't
    ship: its own strategies (DoIntervention, GroundTruthIntervention) replace a value outright,
    where steering instead adds a calibrated direction to whatever value is already there.
    """

    def __init__(self, direction, gamma):
        super().__init__()
        self.direction = direction.detach()  # never used under grad (see steered()); detach defensively
        self.gamma = gamma

    def forward(self, x, *args, **kwargs):
        return x + self.gamma * self.direction


class InterventionModule(nn.Module):
    """Wraps a module producing [B, T, F] or [B, F], and applies intervention_strategy to its
    output, gated per-row by intervention_policy. Our own module, not a subclass of
    pytorch_concepts' BaseInterventionModule: theirs hard-asserts a 2-D [B, F] output (a concept
    encoder's prediction), so it can't wrap a transformer block or a concept head's per-token
    activation() -- both [B, T, F]. This generalizes it by flattening to 2-D and back around
    their unmodified BaseInterventionPolicy.build_mask(), and otherwise works identically:
    accepts any of their (or our) Strategy/Policy subclasses interchangeably.
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
        # pytorch_concepts' build_mask() uses torch.kthvalue, which MPS doesn't implement -- run
        # it on CPU and move the result back, rather than requiring every caller to set
        # PYTORCH_ENABLE_MPS_FALLBACK=1. Cheap either way: policy_scores is only [rows, F].
        mask = self.intervention_policy.build_mask(
            policy_scores.cpu(), sel_idx=self.sel_idx, quantile=self.quantile, eps=self.eps,
        ).to(device=flat.device, dtype=flat.dtype)  # 1 = keep original, 0 = replace (pytorch_concepts' own convention)
        intervened = self.intervention_strategy(flat)

        result = flat * mask + intervened * (1.0 - mask)
        return _unflatten(result, shape)


@contextmanager
def steered(model, direction, gamma, inj_layer):
    """Temporarily wrap model.backbone.blocks[inj_layer:] with an always-on, everywhere
    AddDirectionStrategy injection -- InterventionModule + UniformPolicy(quantile=1.0). This *is*
    Section 6.2's inference-time steering operation (Eq. 18); restores the original blocks on
    exit either way, so it's safe to use inside eval/metric code.

    No custom "AlwaysPolicy" needed: tracing build_mask's arithmetic shows that at quantile=1.0,
    the selection threshold becomes the row-wise max of whatever scores the policy produced, and
    the strict '>' comparison then excludes that max everywhere -- so quantile=1.0 alone already
    forces "intervene at every position" for *any* policy's scores. UniformPolicy (all-zero
    scores, i.e. "no preference") is used here simply because its scores are the cheapest to
    compute; the result is identical to any other policy at this quantile.
    """
    blocks = model.backbone.blocks
    layer_ids = range(inj_layer, len(blocks))
    originals = {i: blocks[i] for i in layer_ids}
    try:
        for i in layer_ids:
            blocks[i] = InterventionModule(
                originals[i], AddDirectionStrategy(direction, gamma), UniformPolicy(), quantile=1.0,
            )
        yield model
    finally:
        for i, orig in originals.items():
            blocks[i] = orig


@contextmanager
def injected_at(model, direction, gamma, position_mask, inj_layer):
    """Like steered(), but injects only at the given (precomputed, exact) position_mask [B, T]
    rather than everywhere. Used by run_steering_batch, where positions come from
    positions_for_concept -- already known exactly, so a Policy's quantile selection would be the
    wrong tool (it chooses *which* rows to intervene on; here that choice is already made by the
    data). Applies the injection via a plain forward hook instead of InterventionModule, since no
    strategy/policy indirection is needed for a fixed, precomputed mask.
    """
    def hook(module, inputs, output):
        return inject_at_positions(output, direction, gamma, position_mask)

    handles = [block.register_forward_hook(hook) for block in list(model.backbone.blocks)[inj_layer:]]
    try:
        yield model
    finally:
        for h in handles:
            h.remove()


def concept_direction(embedding_table, concept_ids):
    """e_c = K_c / ||K_c|| (Eq. 18), or the normalized sum of several embeddings to steer toward
    multiple concepts at once. embedding_table: a head's own [n_or_m, D] embedding parameter
    (e.g. model.bottleneck.known.K); concept_ids: one int or a list of ints.
    """
    if isinstance(concept_ids, int):
        concept_ids = [concept_ids]
    vec = embedding_table[concept_ids].sum(dim=0)  # shape: [len(concept_ids), D] -> [D]
    return vec / vec.norm()


def calibrate_gamma(direction, head, tau=4.0):
    """gamma = tau / peak(e_c), peak(e_c) = max_y (e_c . W_y) (Eq. 19): calibrates injection
    strength so its largest effect on any output logit equals a fixed target tau, giving
    adaptive steering strength across concepts without per-concept tuning. Requires
    head_type="linear" -- the calibration assumes a genuine linear projection, the same
    restriction ConceptLMHead.decompose() notes for the same reason.
    """
    if head.head_type != "linear":
        raise ValueError("calibrate_gamma requires head_type='linear' (Eq. 19 assumes a linear LM head)")
    alignment = head.head.weight @ direction  # shape: [vocab, D] @ [D] -> [vocab], e_c . W_y for every y
    peak = alignment.max().item()
    return tau / peak


def suppress_logits(logits, direction, head, strength):
    """ReLU-gated concept suppression (Eq. 20-21): l_v -= strength*ReLU(a_c[v]) for every vocab
    logit v, where a_c = W.e_c is the concept's own alignment with each token. The ReLU keeps
    this from promoting anti-aligned tokens -- naive subtraction's failure mode (Figure 20).
    Applied directly to final logits, not routed through InterventionModule: this acts on
    neither a hidden state nor a concept activation, so the strategy/policy abstraction doesn't
    fit it -- it's a plain function instead.
    """
    if head.head_type != "linear":
        raise ValueError("suppress_logits requires head_type='linear'")
    a_c = head.head.weight @ direction  # shape: [vocab, D] @ [D] -> [vocab]
    return logits - strength * F.relu(a_c)


def positions_for_concept(token_window, doc_spans, concept_id, lifted_token_ids):
    """Boolean mask [B, T]: True at positions that (a) fall within a document tagged with
    concept_id in this batch of windows (doc_spans, from data.utils.build_supervision), and (b)
    whose own token is one of concept_id's lifted tokens (Section 4.4's lift metric). This is
    the token-level concept attribution steering-training needs -- derived entirely from data
    the pipeline already produces (chunk-level labels + lifted-token statistics), not a new
    LLM-tagging stage.
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
    """h + gamma*direction (Eq. 18), applied only where position_mask is True. The direct,
    non-InterventionModule counterpart of AddDirectionStrategy, for when the positions are
    already known exactly rather than chosen by a Policy (see injected_at()).
    """
    delta = gamma * direction  # shape: [D]
    return h + position_mask.unsqueeze(-1).to(h.dtype) * delta


def sample_steering_target(doc_spans, lifted_tokens):
    """Which concept to steer toward this step: a plain random choice among concepts actually
    present (chunk-level) in the current batch of windows that also have at least one lifted
    token. This is a higher-level "which concept" decision, not a per-position selection, so it
    deliberately isn't routed through a pytorch_concepts Policy -- those choose rows/positions
    within an already-fixed target, not which concept to target in the first place.
    """
    candidates = sorted({c for _, _, _, concept_ids in doc_spans for c in concept_ids if lifted_tokens.get(c)})
    if not candidates:
        return None
    return random.choice(candidates)


def respond_loss(k, concept_id, position_mask):
    """Eq. 31: NLL of the injected concept's own activation at its attributed positions -- push
    k_{c,t} toward 1 there, training the concept module to actually "notice" the concept it was
    just steered toward. k: the known head's activation tensor [B, T, n] from a post-injection
    forward pass; position_mask: [B, T] bool, from positions_for_concept.
    """
    if position_mask.sum() == 0:
        return torch.tensor(0.0, device=k.device)
    k_c = k[..., concept_id]  # shape: [B, T, n] -> [B, T]
    return -torch.log(k_c[position_mask].clamp(1e-6, 1.0)).mean()


def express_loss(logits, lifted_token_ids, position_mask):
    """Eq. 32: negative log probability mass on the concept's lifted tokens at its attributed
    positions -- push the model's output distribution, not just its internal activation, toward
    tokens that express the concept.
    """
    if position_mask.sum() == 0 or not lifted_token_ids:
        return torch.tensor(0.0, device=logits.device)
    probs = F.softmax(logits, dim=-1)  # shape: [B, T, vocab]
    lifted = torch.as_tensor(list(lifted_token_ids), device=logits.device)
    mass = probs.index_select(-1, lifted).sum(dim=-1)  # shape: [B, T, vocab] -> [B, T]
    return -torch.log(mass[position_mask].clamp(1e-6, 1.0)).mean()


@torch.no_grad()
def steering_ability_metrics(model, xb, doc_spans, lifted_tokens, inj_layer=1, tau=4.0):
    """Two cheap, always-on steering-ability proxies (no LLM judge, no generation loop): after
    injecting a sampled concept everywhere via steered(), does the concept's own activation rise
    (respond_delta), and does the output distribution actually shift toward its lifted tokens
    (output_change_delta)? Both deltas come from the same before/after pair of forward passes on
    the same sampled concept, so they're directly comparable. Returns None if no concept present
    in this batch has any lifted tokens to steer toward.
    """
    concept_id = sample_steering_target(doc_spans, lifted_tokens)
    if concept_id is None:
        return None
    direction = concept_direction(model.bottleneck.known.K, concept_id)
    gamma = calibrate_gamma(direction, model.head, tau)
    lifted = torch.as_tensor(lifted_tokens[concept_id], device=xb.device)

    logits_before, intermediates_before = model(xb)
    k_before = intermediates_before['k'][..., concept_id].mean()
    mass_before = F.softmax(logits_before, dim=-1).index_select(-1, lifted).sum(dim=-1).mean()

    with steered(model, direction, gamma, inj_layer):
        logits_after, intermediates_after = model(xb)
    k_after = intermediates_after['k'][..., concept_id].mean()
    mass_after = F.softmax(logits_after, dim=-1).index_select(-1, lifted).sum(dim=-1).mean()

    return {
        'respond_delta': (k_after - k_before).item(),
        'output_change_delta': (mass_after - mass_before).item(),
    }


def run_steering_batch(model, tokens, doc_records, doc_starts, n_train, n_concepts, split,
                        block_size, batch_size, device, lifted_tokens, lambda_respond=1.0,
                        lambda_express=1.0, inj_layer=1, tau=4.0, is_diffusion=False,
                        mask_token_id=None, diff_block_len=None):
    """One steering-training step (Section 10.2.4): sample a target concept present in this
    batch, inject its direction at its attributed positions (positions_for_concept) throughout
    the backbone, and score with L_respond/L_express on top of the ordinary LM loss --
    concept/reconstruction/independence losses are intentionally left out here, matching the
    paper's steering phases. Branches on is_diffusion the same way run_batch/run_diffusion_batch
    do, so this works identically for either backbone.
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
        # nothing steerable in this batch of windows; fall back to a plain LM step so training
        # doesn't stall waiting for a concept that happens not to appear here
        logits, _ = model(model_input)
        lm_loss = lm_loss_from(logits)
        return lm_loss, {'total': lm_loss.item(), 'lm': lm_loss.item(), 'respond': 0.0, 'express': 0.0, 'concept_id': None}

    direction = concept_direction(model.bottleneck.known.K, concept_id)
    gamma = calibrate_gamma(direction, model.head, tau)
    position_mask = positions_for_concept(xb, doc_spans, concept_id, lifted_tokens.get(concept_id, []))

    with injected_at(model, direction, gamma, position_mask, inj_layer):
        logits, intermediates = model(model_input)

    lm_loss = lm_loss_from(logits)
    r_loss = respond_loss(intermediates['k'], concept_id, position_mask)
    e_loss = express_loss(logits, lifted_tokens.get(concept_id, []), position_mask)
    total_loss = lm_loss + lambda_respond * r_loss + lambda_express * e_loss

    components = {
        'total': total_loss.item(), 'lm': lm_loss.item(),
        'respond': r_loss.item(), 'express': e_loss.item(), 'concept_id': concept_id,
    }
    return total_loss, components
