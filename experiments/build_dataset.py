"""Minimal Hydra entry point for building a concept-annotated dataset (the Atlas pipeline).

Fully self-contained: downloads its own copy of each configured corpus source and trains its
own tokenizer, then runs the Atlas stages. Supports training on the *union* of several corpus
sources: each source is downloaded, tagged, and assigned independently, but all sources share
ONE concept library (built from every source's tags combined) and ONE tokenizer, then everything
is merged into a single training set. This is what makes "training on the union" meaningful --
it's not just concatenating unrelated datasets, every source's chunks are assigned concepts from
the same shared library.

Which corpus/corpora to build from is a config concern, not a code one: `corpus.sources` (see
configs/corpus/) is a list -- one entry for a single dataset, several for a union. `atlas.*` (see
configs/atlas/) supplies the dataset-agnostic pipeline hyperparameters, shared across every
source. To add a dataset to (or remove one from) the union, edit `corpus.sources` -- no code
changes needed.

Every stage (including download/tokenizer-training) is idempotent -- skips if its own output
already exists -- so re-running after a partial failure, or after only changing a later stage's
hyperparameters, doesn't redo already-finished work.

Run with defaults:            python build_dataset.py
Override any hyperparameter:  python build_dataset.py atlas.num_documents=2000 atlas.k=80
Build a different corpus:     python build_dataset.py corpus=my_corpus
(see README.md for the full explanation of config overrides, unions, and multirun)
"""
import os

import hydra
from omegaconf import DictConfig

from babysteerling.data.atlas import assign_concepts, build_concepts, tag_chunks, tokenize_dataset
from babysteerling.data.prepare import download_corpus, train_tokenizer
from babysteerling.data.utils import combine_jsonl


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    a, c = cfg.atlas, cfg.corpus
    os.makedirs(a.output_dir, exist_ok=True)
    raw_dir = os.path.join(a.output_dir, "raw")

    tokenizer_path = os.path.join(a.output_dir, "tokenizer.json")
    combined_tags_path = os.path.join(a.output_dir, "tags.jsonl")
    concepts_path = os.path.join(a.output_dir, "concepts.json")
    combined_chunk_concepts_path = os.path.join(a.output_dir, "chunk_concepts.jsonl")
    tokens_path = os.path.join(a.output_dir, "steerling_tokens.pt")
    doc_records_path = os.path.join(a.output_dir, "steerling_concepts.pt")

    # --- per-source: download raw text, then LLM-tag it independently ---
    input_paths, tags_paths = [], []
    for source in c.sources:
        input_path = os.path.join(raw_dir, f"{source.name}.txt")
        download_corpus(source.url, input_path)
        input_paths.append(input_path)

        tags_path = os.path.join(a.output_dir, f"tags_{source.name}.jsonl")
        tag_chunks(
            input_path=input_path, output_path=tags_path, num_documents=a.num_documents,
            document_delimiter=source.document_delimiter, prompt_template=a.tag_prompt_template,
            model_name=a.tagging_model, batch_size=a.tagging_batch_size,
            max_new_tokens=a.tagging_max_new_tokens, seed=a.seed,
        )
        tags_paths.append(tags_path)

    # one shared tokenizer across every source, so a union dataset tokenizes consistently;
    # boundary_token is independent of each source's own document_delimiter (see prepare.py)
    train_tokenizer(input_paths, tokenizer_path, vocab_size=c.tokenizer_vocab_size,
                     boundary_token=c.boundary_token)

    # one shared concept library built from every source's tags combined -- this is what makes
    # "training on the union" meaningful, rather than just concatenating unrelated datasets
    combine_jsonl(tags_paths, combined_tags_path)
    build_concepts(
        tags_path=combined_tags_path, output_path=concepts_path, embed_model_name=a.embed_model,
        label_model_name=a.tagging_model, k=a.k, min_cluster_size=a.min_cluster_size,
        tags_per_label_prompt=a.tags_per_label_prompt, dedup_threshold=a.dedup_threshold,
        batch_size=a.label_batch_size, max_new_tokens=a.label_max_new_tokens, seed=a.seed,
        label_prompt_template=a.label_prompt_template,
    )

    # --- per-source: assign against the shared library, then merge into one training set ---
    chunk_concepts_paths = []
    for source, tags_path, input_path in zip(c.sources, tags_paths, input_paths):
        chunk_concepts_path = os.path.join(a.output_dir, f"chunk_concepts_{source.name}.jsonl")
        assign_concepts(
            tags_path=tags_path, concepts_path=concepts_path, output_path=chunk_concepts_path,
            input_path=input_path, document_delimiter=source.document_delimiter,
            enable_scale_up=a.enable_scale_up, num_scaleup_documents=a.num_scaleup_documents,
            similarity_floor=a.similarity_floor, embed_model_name=a.embed_model,
        )
        chunk_concepts_paths.append(chunk_concepts_path)

    combine_jsonl(chunk_concepts_paths, combined_chunk_concepts_path)
    tokenize_dataset(
        chunk_concepts_path=combined_chunk_concepts_path, tokenizer_path=tokenizer_path,
        tokens_output_path=tokens_path, concepts_output_path=doc_records_path,
        boundary_token=c.boundary_token,
    )

    source_names = ", ".join(s.name for s in c.sources)
    print(f"\nDataset build complete in {a.output_dir!r} from source(s): {source_names}. "
          f"Train on it with: python train.py data.data_dir={a.output_dir}")


if __name__ == "__main__":
    main()
