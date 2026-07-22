# babysteerling

A didactic, from-scratch build-up of **interpretable language models**, small enough to train on
a laptop in a few minutes per step. The goal isn't state-of-the-art performance — it's to
actually implement, in readable code, the ideas behind making a language model's internals
inspectable (which concepts it represents, how much of its output traces back to them) rather
than treating interpretability as something bolted on after training with a post-hoc probe.

The numbered folders are a self-contained, step-by-step build-up (read top-to-bottom, one file
each, no shared code between them on purpose — see `CONTRIBUTING.md`). `babysteerling/` and
`experiments/` are where that build-up lands: a real installable package and a config-driven
runner around it.

- **`0_gpt_chars/`**: Karpathy-style bigram model and a tiny GPT, trained character-by-character
  on TinyShakespeare. The starting point: a transformer language model with no tokenizer, no
  interpretability, just next-character prediction.
- **`1_gpt_tokens/`**: the same GPT architecture, but tokenized (BPE) and trained on TinyStories
  instead of characters. Includes a couple of speed/efficiency variants of the training script.
- **`2_atlas/`**: before a model can be interpretable in terms of *concepts*, something has to
  define what the concepts are. This folder builds a small concept-annotated version of
  TinyStories, inspired by Guide Labs' Atlas pipeline (see NOTICE for attribution): tag text
  chunks with an LLM, cluster the tags into a canonical concept library, and assign concepts back
  to the training data — scaled down from Atlas's billion-chunk pipeline to something that runs
  on a laptop.
- **`3_steerling/`**: single-file implementations of the Steerling architecture (a concept
  bottleneck that decomposes the model's hidden state into known-concept, unknown-concept, and
  residual parts, trained with dedicated concept/reconstruction/independence losses so
  predictions are attributable back to specific concepts) — `steerling.py` follows the reference
  architecture closely (Causal Diffusion backbone, teacher forcing); `gpt_steerling.py`
  backtracks to a plain autoregressive GPT backbone with the same bottleneck and losses, simpler
  and faster to iterate on; `gpt_steerling_deephead.py` and `gpt_steerling_no_supervision.py` are
  small ablations (a deeper classification head; concept losses zeroed out as a baseline).
- **`babysteerling/`**: the same architecture and Atlas pipeline, refactored into a real,
  installable package (`pip install -e .`) so it can be used outside this repo. See NOTICE for
  attribution to the work this independently implements.
- **`experiments/`**: the Hydra + Weights & Biases runner around the `babysteerling` package —
  where to actually train models and build datasets with configurable hyperparameters, log
  results, and compare runs (including parameter sweeps run in parallel) instead of hand-copying
  a script per variant. See `experiments/README.md` for the full how-to.

## Setup

```bash
mamba create -n babysteerling python
pip install -r requirements.txt      # deps for the standalone numbered folders
pip install -e ".[atlas]"            # editable install of babysteerling, needed for experiments/
```

## Running a single experiment

```bash
cd experiments
python build_dataset.py   # first time only: downloads TinyStories and builds a concept dataset
python train.py
```

## License

Apache License 2.0 (see `LICENSE`). See `NOTICE` for attribution to Guide Labs' Steerling and
Atlas work, which this project independently implements at a much smaller scale.
