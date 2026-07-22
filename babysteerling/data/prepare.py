"""Stage 0: fetch a raw training corpus and train a tokenizer on it.

Kept separate from the Atlas stages (tag/cluster/assign/tokenize): this is one-time,
corpus-level setup, not part of concept annotation itself, and is the same regardless of which
concept-annotation hyperparameters a given build_dataset run uses.
"""
import os
import urllib.request

from tokenizers import ByteLevelBPETokenizer


def download_corpus(url, output_path):
    """Download a plain-text corpus from `url` to `output_path`.

    Works with any corpus, not just TinyStories -- which corpus to use is a config concern
    (see experiments/configs/corpus/) rather than something this function should hardcode.

    Idempotent: skips entirely if `output_path` already exists (large corpora shouldn't be
    redownloaded on every run).
    """
    if os.path.exists(output_path):
        print(f"Found existing {output_path}, skipping download.")
        return
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    print(f"Downloading corpus from {url}...")
    urllib.request.urlretrieve(url, output_path)
    print(f"Downloaded corpus to {output_path}")


def train_tokenizer(input_paths, output_path, vocab_size=2000, boundary_token="<|endoftext|>"):
    """Train a byte-level BPE tokenizer on one or more corpus files.

    `input_paths` accepts either a single path or a list -- pass every source's raw text file
    when building a union dataset (see build_dataset.py), so the resulting vocabulary covers all
    of them and everything can share one tokenizer.

    `boundary_token` is added as a special token so it survives tokenization intact. Unlike each
    source's own `document_delimiter` (used only to split that source's raw file into documents,
    and free to differ per source), this token is an artificial marker this pipeline inserts
    after every chunk in the *tokenized* stream -- it doesn't need to match any source's raw
    text at all, it just needs to be one consistent token that tokenize_dataset() can look up.

    Idempotent: skips entirely if `output_path` already exists.
    """
    if os.path.exists(output_path):
        print(f"Found existing {output_path}, skipping tokenizer training.")
        return
    if isinstance(input_paths, str):
        input_paths = [input_paths]
    print(f"Training a byte-level BPE tokenizer (vocab_size={vocab_size}) on {len(input_paths)} file(s)...")
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=input_paths,
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[boundary_token],
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tokenizer.save(output_path)
    print(f"Trained tokenizer with vocab_size={tokenizer.get_vocab_size()}, saved to {output_path}")
