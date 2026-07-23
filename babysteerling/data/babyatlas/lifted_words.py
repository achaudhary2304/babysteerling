"""Compute per-concept "lifted" tokens: vocabulary tokens most statistically associated with a
concept, following Section 4.4's lift metric: lift(w, c) = P(w|c) / P(w).

Used by babysteerling.steering as the token-level attribution signal for steering (a token in a
document already tagged with concept c counts as "attributed" to c if it's one of c's lifted
tokens) -- derived entirely from data the pipeline already produces (the final tokenized dataset
+ its per-document concept labels), rather than a new LLM-tagging stage.
"""
import json
import os
from collections import Counter

import torch


def compute_lifted_tokens(tokens_path, doc_records_path, output_path=None, top_k=50, min_support=5):
    """For each concept, rank vocabulary tokens by lift = P(token | concept) / P(token), using
    token frequency within documents tagged with that concept vs. corpus-wide token frequency.

    tokens_path / doc_records_path: the same steerling_tokens.pt / steerling_concepts.pt written
    by tokenize_dataset() -- reused directly, no re-tokenization needed.

    Returns {concept_id: [token_id, ...]} (top_k tokens per concept, each required to appear at
    least min_support times within that concept's documents, to keep idiosyncratic rare tokens
    from dominating the ranking with a spuriously high lift score).

    Idempotent: if output_path is given and already exists, loads and returns it instead of
    recomputing.
    """
    if output_path and os.path.exists(output_path):
        print(f"Found existing {output_path}, skipping lifted-token computation.")
        with open(output_path, 'r', encoding='utf-8') as f:
            return {int(k): v for k, v in json.load(f).items()}

    tokens = torch.load(tokens_path)
    doc_records = torch.load(doc_records_path)

    corpus_counts = Counter()   # token_id -> count across the whole corpus
    concept_counts = {}         # concept_id -> Counter(token_id -> count within that concept's documents)
    total_tokens = len(tokens)

    for doc in doc_records:
        doc_tokens = tokens[doc['start']:doc['end']].tolist()  # this document's own token ids
        corpus_counts.update(doc_tokens)
        for concept_id in doc['concept_ids']:
            concept_counts.setdefault(concept_id, Counter()).update(doc_tokens)

    lifted_tokens = {}
    for concept_id, counts in concept_counts.items():
        concept_total = sum(counts.values())
        scored = []
        for token_id, count in counts.items():
            if count < min_support:
                continue
            p_token_given_concept = count / concept_total
            p_token = corpus_counts[token_id] / total_tokens
            lift = p_token_given_concept / p_token
            scored.append((lift, token_id))
        scored.sort(reverse=True)  # highest lift first
        lifted_tokens[concept_id] = [token_id for _, token_id in scored[:top_k]]

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({str(k): v for k, v in lifted_tokens.items()}, f, indent=2)  # JSON keys must be strings
        print(f"Wrote lifted tokens for {len(lifted_tokens)} concepts to {output_path}")

    return lifted_tokens
