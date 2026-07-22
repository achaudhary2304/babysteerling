"""Data loading for a concept-annotated corpus produced by babysteerling.data.atlas (or any
pipeline that writes the same steerling_tokens.pt / steerling_concepts.pt / concepts.json format).

Everything here is plain functions rather than a torch Dataset/DataLoader: batches are built
by sampling random contiguous token windows (like a standard nanoGPT-style loader) and then
looking up which documents (= concept-annotated "chunks") overlap that window. That lookup
doesn't map cleanly onto a map-style Dataset, so a few small functions are simpler and more
transparent than forcing an abstraction that doesn't fit.
"""
import bisect
import json
import os

import torch
from tokenizers import Tokenizer


def combine_jsonl(input_paths, output_path):
    """Concatenate several JSONL files into one, renumbering each row's `chunk_id` sequentially.

    Used by build_dataset.py to merge multiple corpus sources (each tagged/assigned
    independently) into the single tags.jsonl / chunk_concepts.jsonl that build_concepts() /
    tokenize_dataset() expect -- this is what makes "train on the union of several sources"
    possible: from this point on the pipeline just sees one combined file.

    Idempotent: skips entirely if `output_path` already exists.
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
    """Load the token stream, per-document concept labels, and the concept library.

    tokens: 1D LongTensor, the whole corpus concatenated (documents separated by <|endoftext|>).
    doc_records: list of {chunk_id, start, end, concept_ids} -- one entry per document, giving
        its [start, end) token span in `tokens` and its ground-truth concept ids. If the pipeline
        that built this dataset treats one document as one chunk (as babysteerling.data.atlas
        does for TinyStories), "document" and "chunk" mean the same thing here.
    n_concepts: size of the known-concept library (used to size the concept bottleneck's known
        head and the dense multi-hot label tensors built in build_supervision()).
    """
    tokens = torch.load(os.path.join(data_dir, "steerling_tokens.pt"))  # shape: [N_total_tokens]
    doc_records = torch.load(os.path.join(data_dir, "steerling_concepts.pt"))
    with open(os.path.join(data_dir, "concepts.json")) as f:
        concept_library = json.load(f)
    n_concepts = len(concept_library)
    return tokens, doc_records, n_concepts


def overlapping_docs(doc_records, doc_starts, window_start, window_end):
    """Find every document whose token span intersects [window_start, window_end).

    Documents are laid out contiguously and non-overlapping in `tokens` (each document's end
    equals the next one's start), so the overlap set is always a short contiguous run of
    doc_records. `doc_starts` (a sorted list of each document's start offset) lets us binary
    search straight to that run instead of scanning the whole corpus for every window.

    Returns a list of (local_start, local_end, concept_ids) tuples, where local_start/local_end
    are offsets relative to window_start (i.e. valid indices into a [block_size]-length window).
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
    """Sample a batch of (input, target) windows for next-token prediction.

    Standard autoregressive windowing: pick `batch_size` random start offsets, take
    `block_size` tokens as input x, and the same span shifted one token to the right as the
    target y. `split` restricts which region of the corpus the windows are drawn from, so
    validation windows never overlap the training region.
    """
    lo, hi = (0, n_train) if split == 'train' else (n_train, len(tokens))
    ix = torch.randint(lo, hi - block_size, (batch_size,))  # shape: [batch_size], window start offsets
    x = torch.stack([tokens[i:i + block_size] for i in ix])       # shape: [batch_size, block_size]
    y = torch.stack([tokens[i + 1:i + block_size + 1] for i in ix])  # shape: [batch_size, block_size], shifted by 1
    x, y = x.to(device), y.to(device)
    return x, y, ix.tolist()


def build_supervision(doc_records, doc_starts, starts, block_size, n_concepts, device):
    """Build the concept-loss/reconstruction-loss supervision for a sampled batch of windows.

    Returns:
      doc_spans: list of (batch_idx, tok_start, tok_end, concept_ids) -- one entry per document
          overlapping any window in the batch. Consumed by loss.py's ConceptLoss, which
          OR-aggregates predictions *within* each document's own span, never merging documents.
      known_labels: dense multi-hot tensor of ground-truth concept labels, broadcast to every
          token position of the document it belongs to. This is what the concept bottleneck uses
          to compute the reconstruction target for the unknown head (see nn.py).
    """
    doc_spans = []
    known_labels = torch.zeros(len(starts), block_size, n_concepts, device=device)  # shape: [B, T, n_concepts]
    for b, window_start in enumerate(starts):
        for tok_start, tok_end, concept_ids in overlapping_docs(doc_records, doc_starts, window_start, window_start + block_size):
            doc_spans.append((b, tok_start, tok_end, concept_ids))
            known_labels[b, tok_start:tok_end, concept_ids] = 1.0  # broadcast labels over the doc's token span
    return doc_spans, known_labels
