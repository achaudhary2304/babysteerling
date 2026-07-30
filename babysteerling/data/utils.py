"""Data loading for a concept-annotated corpus built by babysteerling.data.atlas (or anything
that writes the same steerling_tokens.pt / steerling_concepts.pt / concepts.json files).

Plain functions, no torch Dataset/DataLoader: a batch is a set of random token windows (like a
standard nanoGPT loader), plus a lookup for which documents overlap each window. That lookup
doesn't fit a map-style Dataset well, so plain functions are simpler here.
"""
import bisect
import json
import os

import torch
from tokenizers import Tokenizer


def combine_jsonl(input_paths, output_path):
    """Concatenates several JSONL files into one, renumbering each row's `chunk_id` in order.

    Used by build_dataset.py to merge tagged/assigned corpus sources into the single
    tags.jsonl/chunk_concepts.jsonl that build_concepts()/tokenize_dataset() expect, so the
    rest of the pipeline just sees one file.

    Idempotent: skips if `output_path` already exists.
    """
    if os.path.exists(output_path):
        print(f"Found existing {output_path}, skipping combine.")
        return
    next_chunk_id = 0
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as out_f:
        for input_path in input_paths:
            with open(input_path, 'r', encoding='utf-8') as in_f:
                for line in in_f:
                    row = json.loads(line)
                    row['chunk_id'] = next_chunk_id
                    out_f.write(json.dumps(row) + "\n")
                    next_chunk_id += 1
    print(f"Combined {len(input_paths)} file(s) into {output_path} ({next_chunk_id} rows).")


def load_tokenizer(data_dir, filename="tokenizer.json"):
    """Load the BPE tokenizer used to build the dataset. Returns the tokenizer object, its
    vocab size, and a decode function for turning generated ids back into text."""
    tok = Tokenizer.from_file(os.path.join(data_dir, filename))
    vocab_size = tok.get_vocab_size()
    decode = lambda ids: tok.decode(ids)
    return tok, vocab_size, decode


def load_dataset(data_dir):
    """Loads the token stream, per-document concept labels, and the concept library.

    tokens: 1D LongTensor, the whole corpus concatenated (documents separated by <|endoftext|>).
    doc_records: list of {chunk_id, start, end, concept_ids}, one per document: its [start, end)
        span in `tokens` and its ground-truth concept ids.
    n_concepts: size of the known-concept library, used to size the bottleneck's known head and
        the label tensors build_supervision() builds.
    """
    tokens = torch.load(os.path.join(data_dir, "steerling_tokens.pt"))  # shape: [N_total_tokens]
    doc_records = torch.load(os.path.join(data_dir, "steerling_concepts.pt"))
    with open(os.path.join(data_dir, "concepts.json")) as f:
        concept_library = json.load(f)
    n_concepts = len(concept_library)
    return tokens, doc_records, n_concepts


PROTOTYPE_VALUE_ORDER = ("negative", "unrelated", "positive")  # matches a fixed -1/0/+1 activation axis


def load_concept_prototype_tokens(data_dir, tokenizer, n_concepts, filename="concept_prototypes.json",
                                   max_tokens=32):
    """Loads per-concept prototype texts (babysteerling.data.babyatlas.build_concept_prototypes)
    and tokenizes them with the corpus's own BPE tokenizer.

    Returns an int32 tensor [n_concepts, 3, per_type, Tp]. The 3 axis is always
    PROTOTYPE_VALUE_ORDER (negative, unrelated, positive). A concept missing from the file gets
    an all-pad row.

    Tp is capped at max_tokens. This matters more than it looks: nn.prototype's Selectors gather
    per (candidate, token position), so one long, uncapped prototype anywhere in the library
    would multiply into every gather, not just its own row (an uncapped Tp near 300 has blown
    this up to double-digit GB for a modest batch). int32 instead of int64 halves that further.

    Returns None if the file doesn't exist, so callers that don't use a prototype-based encoder
    never have to check for it.
    """
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        by_concept = {int(k): v for k, v in json.load(f).items()}

    per_type = len(next(iter(by_concept.values()))[PROTOTYPE_VALUE_ORDER[0]])

    all_ids = []
    for concept_id in range(n_concepts):
        by_type = by_concept.get(concept_id, {})
        for ptype in PROTOTYPE_VALUE_ORDER:
            items = by_type.get(ptype) or [{"text": ""}] * per_type
            for item in items:
                all_ids.append(tokenizer.encode(item['text']).ids[:max_tokens])

    Tp = max((len(ids) for ids in all_ids), default=1) or 1
    padded = torch.tensor([ids + [0] * (Tp - len(ids)) for ids in all_ids], dtype=torch.int32)
    return padded.view(n_concepts, len(PROTOTYPE_VALUE_ORDER), per_type, Tp)


def load_lifted_tokens(data_dir):
    """Loads the per-concept lifted-token stats from babysteerling.data.atlas.compute_lifted_tokens
    (Section 4.4's lift metric), the token-level concept attribution babysteerling.steering uses.
    Returns {} if the dataset predates lifted tokens or steering was never enabled, so callers
    that don't need steering never have to check.
    """
    path = os.path.join(data_dir, "lifted_tokens.json")
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return {int(k): v for k, v in json.load(f).items()}


def overlapping_docs(doc_records, doc_starts, window_start, window_end):
    """Finds every document whose token span overlaps [window_start, window_end).

    Documents sit contiguously in `tokens` (one ends where the next starts), so the overlap set
    is always a short run of doc_records. `doc_starts` (sorted start offsets) lets us binary
    search straight to that run instead of scanning the whole corpus.

    Returns (local_start, local_end, concept_ids) tuples, with local_start/local_end relative to
    window_start.
    """
    # first candidate document: the last one starting at or before window_start
    i = max(bisect.bisect_right(doc_starts, window_start) - 1, 0)
    spans = []
    while i < len(doc_records) and doc_records[i]['start'] < window_end:
        d = doc_records[i]
        s, e = max(d['start'], window_start), min(d['end'], window_end)  # intersect with window
        if e > s:
            spans.append((s - window_start, e - window_start, d['concept_ids']))  # -> window-local offsets
        i += 1
    return spans


def get_batch(tokens, split, block_size, batch_size, n_train, device):
    """Samples a batch of (input, target) windows for next-token prediction.

    Picks `batch_size` random start offsets, takes `block_size` tokens as x, and the same span
    shifted one token right as y. `split` restricts which region windows are drawn from, so
    validation never overlaps training.
    """
    lo, hi = (0, n_train) if split == 'train' else (n_train, len(tokens))
    ix = torch.randint(lo, hi - block_size, (batch_size,))  # shape: [batch_size], window start offsets
    x = torch.stack([tokens[i:i + block_size] for i in ix])       # shape: [batch_size, block_size]
    y = torch.stack([tokens[i + 1:i + block_size + 1] for i in ix])  # shape: [batch_size, block_size], shifted by 1
    x, y = x.to(device), y.to(device)
    return x, y, ix.tolist()


def build_supervision(doc_records, doc_starts, starts, block_size, n_concepts, device):
    """Builds the concept-loss/reconstruction-loss supervision for a batch of sampled windows.

    Returns:
      doc_spans: (batch_idx, tok_start, tok_end, concept_ids), one per document overlapping the
          batch. Consumed by loss.py's ConceptLoss, which aggregates within each document's own
          span and never merges documents.
      known_labels: dense multi-hot ground-truth concept labels, broadcast to every token
          position of the document it belongs to. Used to build the unknown head's
          reconstruction target.
    """
    doc_spans = []
    known_labels = torch.zeros(len(starts), block_size, n_concepts, device=device)  # shape: [B, T, n_concepts]
    for b, window_start in enumerate(starts):
        for tok_start, tok_end, concept_ids in overlapping_docs(doc_records, doc_starts, window_start, window_start + block_size):
            doc_spans.append((b, tok_start, tok_end, concept_ids))
            known_labels[b, tok_start:tok_end, concept_ids] = 1.0  # broadcast labels over the doc's token span
    return doc_spans, known_labels
