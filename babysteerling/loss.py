"""Loss functions for the concept bottleneck model.

Pure functions only: everything here works from logits/intermediates already produced by the
model's forward pass. Nothing in this file runs the model.

Four terms make up the total training loss:
  1. Language modeling loss: the actual next-token prediction task. Plain F.cross_entropy,
     kept inline below (no need for its own class).
  2. Concept loss: did the known-concept head predict the right concepts?
  3. Reconstruction loss: does the unknown head capture what the known concepts leave out?
  4. Independence loss: are the known and unknown parts decorrelated, so they don't both encode
     the same information?
"""
import torch
import torch.nn as nn
from torch.nn import functional as F


class ConceptLoss(nn.Module):
    """BCE over per-document concept labels, aggregated across the document with a soft-OR.

    A document's label means "concept c appears somewhere in this document", not "at every
    token". So for each concept we combine its per-token predictions into one per-document
    probability with a soft-OR (1 minus the product of "concept absent" over every token), then
    compare that to the 0/1 label. The loss is satisfied as soon as one token confidently
    predicts the concept.
    """

    def forward(self, k, doc_spans, return_accuracy=False):
        """
        k: [B, T, n] predicted known-concept activations (post-sigmoid).
        doc_spans: list of (batch_idx, tok_start, tok_end, concept_ids), one per document that
            overlaps this batch of windows (see data/utils.py's build_supervision()).
        return_accuracy=True also returns two accuracies, computed in the same loop as the loss:
        OR-aggregated (matches the loss's own semantics: does the doc-level soft-OR prediction
        match the doc-level label) and per-token (does each individual token's own prediction
        match that same doc-level label, without aggregating first -- a stricter, more granular
        view, at the cost of scoring the same label once per token instead of once per document).
        """
        if not doc_spans:
            zero = torch.tensor(0.0, device=k.device)
            return (zero, zero, zero) if return_accuracy else zero
        n = k.shape[-1]
        total = k.new_zeros(())
        correct_or = k.new_zeros(())
        correct_per_token = k.new_zeros(())
        n_tokens = 0
        for batch_idx, tok_start, tok_end, concept_ids in doc_spans:
            k_span = k[batch_idx, tok_start:tok_end, :]  # shape: [B, T, n] -> [doc_len, n], this doc's own tokens only
            k_chunk = 1 - torch.prod(1 - k_span, dim=0)  # shape: [doc_len, n] -> [n], soft-OR aggregation over the doc
            y = k.new_zeros(n)  # shape: [n], multi-hot ground-truth label for this document
            y[concept_ids] = 1.0
            # clamp avoids log(0) in BCE when a prediction is fully saturated at 0 or 1
            total = total + F.binary_cross_entropy(k_chunk.clamp(1e-6, 1 - 1e-6), y, reduction='sum')
            if return_accuracy:
                correct_or = correct_or + ((k_chunk.detach() > 0.5) == y.bool()).float().sum()
                correct_per_token = correct_per_token + (
                    (k_span.detach() > 0.5) == y.bool().unsqueeze(0)
                ).float().sum()  # y broadcasts to every token in the doc's span
                n_tokens += k_span.shape[0]
        loss = total / len(doc_spans)  # average per-document loss, so batch size doesn't change the scale
        if return_accuracy:
            return loss, correct_or / (len(doc_spans) * n), correct_per_token / (n_tokens * n)
        return loss


def selected_concept_diagnostics(k, known_labels, doc_spans=None, eps=1e-6):
    """Diagnostic BCE and accuracy for sparse, prototype-routed encoders like
    known_encoder_type="linear_selector". Never contributes a gradient (see
    model.use_concept_loss).

    Only scores the concepts actually selected at each (batch, token) position. k is exactly 0
    everywhere else (see nn.prototype.PrototypePredictor), so scoring every slot the way
    ConceptLoss does would count every unselected concept as a free true negative and bias the
    metric.

    k's value axis is [-1, +1] (see nn.prototype._fixed_value_axis), rescaled to [0, 1] via
    (k+1)/2 before BCE.

    doc_spans, if given, also returns an OR-aggregated accuracy: per document, per concept that
    was selected at least once somewhere in it, "present" if any selected token's rescaled
    probability exceeds 0.5 (a soft-OR over selected positions only, same style as ConceptLoss's
    own aggregation) -- matches ConceptLoss's accuracy semantics instead of this function's usual
    per-token one. A concept never selected anywhere in a document isn't scored at all (same
    "only score what was actually selected" rule as the per-token accuracy above), rather than
    counted as a free true negative.
    """
    with torch.no_grad():
        mask = k != 0
        if mask.sum() == 0:
            zero = torch.tensor(0.0, device=k.device)
            return (zero, zero, zero) if doc_spans is not None else (zero, zero)
        probs = ((k[mask] + 1) / 2).clamp(eps, 1 - eps)
        target = known_labels[mask].float()
        bce = F.binary_cross_entropy(probs, target)
        accuracy = ((probs > 0.5) == target.bool()).float().mean()

        if doc_spans is None:
            return bce, accuracy

        n = k.shape[-1]
        prob_full = (k + 1) / 2  # shape: [B, T, n]; unscored (k==0) positions read 0.5, excluded below via mask
        or_correct = k.new_zeros(())
        n_scored = k.new_zeros(())
        for batch_idx, tok_start, tok_end, concept_ids in doc_spans:
            p_span = prob_full[batch_idx, tok_start:tok_end, :]  # shape: [doc_len, n]
            s_span = mask[batch_idx, tok_start:tok_end, :]  # shape: [doc_len, n]
            scored_here = s_span.any(dim=0)  # shape: [n], which concepts were selected anywhere in this doc
            if scored_here.sum() == 0:
                continue
            p_for_or = torch.where(s_span, p_span, torch.zeros_like(p_span))  # unselected -> 0, no effect on the product below
            doc_prob = 1 - torch.prod(1 - p_for_or, dim=0)  # shape: [n], soft-OR over selected positions only
            y = k.new_zeros(n)
            y[concept_ids] = 1.0
            or_correct = or_correct + ((doc_prob > 0.5) == y.bool())[scored_here].float().sum()
            n_scored = n_scored + scored_here.sum()
        or_accuracy = or_correct / n_scored.clamp(min=1)
        return bce, accuracy, or_accuracy


class ReconstructionLoss(nn.Module):
    """MSE between the unknown head's u_hat and its target (h minus the known concepts'
    contribution). Trains the unknown head to capture exactly what the known concepts miss.

    mask, if given (e.g. the diffusion corruption mask), restricts the loss to those positions.
    None (default) uses every position, which is correct for the causal backbone.
    """

    def forward(self, u_hat, u_hat_gt, mask=None):
        if u_hat_gt is None:
            return torch.tensor(0.0, device=u_hat.device)
        if mask is None:
            return ((u_hat - u_hat_gt) ** 2).mean()  # shape: [B, T, d] -> scalar
        if mask.sum() == 0:
            return torch.tensor(0.0, device=u_hat.device)
        diff = u_hat[mask] - u_hat_gt[mask]  # shape: [B, T, d] -> [n_masked, d]
        return (diff ** 2).mean()


class IndependenceLoss(nn.Module):
    """Penalizes correlation between k_hat and u_hat, so the unknown head doesn't just re-learn
    what the known head already captures.

    Only the unknown side gets gradient (k_hat is detached): we want the unknown head to adapt
    to the known head, not the other way around, since the known head is anchored to human
    labels.
    """

    def forward(self, k_hat, u_hat):
        d = k_hat.shape[-1]
        Hk = k_hat.detach().reshape(-1, d)  # shape: [B, T, d] -> [B*T, d], flatten batch+time into one axis of "samples"
        Hu = u_hat.reshape(-1, d)  # shape: [B, T, d] -> [B*T, d]
        num_tokens = Hk.shape[0]

        Phi = Hk - Hk.mean(dim=0, keepdim=True)  # shape: [B*T, d], center each feature across the batch
        Psi = Hu - Hu.mean(dim=0, keepdim=True)  # shape: [B*T, d]
        cross_cov = Psi.t() @ Phi  # shape: [d, B*T] @ [B*T, d] -> [d, d], empirical cross-covariance matrix
        return (cross_cov ** 2).sum() / (d ** 2 * max(num_tokens - 1, 1))  # normalized Frobenius norm^2


# shared singletons: these losses have no learnable parameters, so one instance is enough
_concept_loss_fn = ConceptLoss()
_rec_loss_fn = ReconstructionLoss()
_indep_loss_fn = IndependenceLoss()


def compute_losses(logits, targets, intermediates, doc_spans, known_labels=None,
                    lambda_concept=1.0, lambda_rec=1.0, lambda_indep=1.0, mask=None,
                    mask_weights=None, use_concept_loss=True):
    """Combines all four loss terms into the total training loss.

    mask, if given (e.g. the diffusion backbone's corruption mask), restricts the LM and
    reconstruction losses to those positions. None (default) scores every position, which is
    correct for the causal backbone.
    intermediates: the dict returned by ConceptBottleneck.forward().
    known_labels: [B, T, n] dense multi-hot ground truth, only used when use_concept_loss=False.
    use_concept_loss=False (see model.use_concept_loss) drops ConceptLoss from total_loss and
    reports selected_concept_diagnostics instead for components['concept'/'concept_accuracy_or'/
    'concept_accuracy_per_token'], purely for logging (see that function for why). Both
    accuracies are always reported regardless of use_concept_loss, so the two encoder families
    stay comparable on the same terms instead of each only ever exposing the one its own loss
    happens to aggregate by.
    Returns (total_loss, components), a plain dict of floats for logging each term.
    """
    B, T, C = logits.shape
    pred_ids = logits.argmax(-1)  # shape: [B, T, vocab] -> [B, T]
    if mask is None:
        lm_loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))  # shape: [B, T, vocab] -> [B*T, vocab] vs [B*T]
        lm_accuracy = (pred_ids == targets).float().mean()
    elif mask.sum() == 0:
        lm_loss = torch.tensor(0.0, device=logits.device)
        lm_accuracy = torch.tensor(0.0, device=logits.device)
    else:
        ce = F.cross_entropy(logits[mask], targets[mask], reduction='none')  # shape: [B, T, vocab] -> [n_masked, vocab] vs [n_masked]
        if mask_weights is None:
            lm_loss = ce.mean()
        else:
            # the masked-diffusion ELBO weights each masked position by 1/p_mask, so that blocks
            # with a low noise level aren't under-counted. Normalizing by the weight sum rather
            # than the token count keeps it a weighted mean: the large weights a small t produces
            # then appear on both sides and cancel, instead of dominating the gradient.
            w = 1.0 / mask_weights[mask]  # shape: [n_masked]
            lm_loss = (ce * w).sum() / w.sum()
        lm_accuracy = (pred_ids[mask] == targets[mask]).float().mean()

    if not intermediates:  # no bottleneck: cross-entropy is the whole loss
        components = {'total': lm_loss.item(), 'lm': lm_loss.item(), 'lm_accuracy': lm_accuracy.item()}
        components.update(concept=0.0, concept_accuracy_or=0.0, concept_accuracy_per_token=0.0,
                          rec=0.0, indep=0.0)
        return lm_loss, components

    if use_concept_loss:
        concept_loss, concept_accuracy_or, concept_accuracy_per_token = _concept_loss_fn(
            intermediates['k'], doc_spans, return_accuracy=True,
        )
    else:
        concept_loss, concept_accuracy_per_token, concept_accuracy_or = selected_concept_diagnostics(
            intermediates['k'], known_labels, doc_spans=doc_spans,
        )
    rec_loss = _rec_loss_fn(intermediates['u_hat'], intermediates['u_hat_gt'], mask=mask)
    indep_loss = _indep_loss_fn(intermediates['k_hat'], intermediates['u_hat'])

    total_loss = lm_loss + lambda_rec * rec_loss + lambda_indep * indep_loss
    if use_concept_loss:
        total_loss = total_loss + lambda_concept * concept_loss  # else: diagnostic only, never optimized (see docstring)
    components = {
        'total': total_loss.item(),
        'lm': lm_loss.item(),
        'lm_accuracy': lm_accuracy.item(),
        'concept': concept_loss.item(),
        'concept_accuracy_or': concept_accuracy_or.item(),
        'concept_accuracy_per_token': concept_accuracy_per_token.item(),
        'rec': rec_loss.item(),
        'indep': indep_loss.item(),
    }
    return total_loss, components
