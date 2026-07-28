# babysteerling

A didactic, from scratch build of interpretable language models. It trains on a laptop in a few
minutes per step.

The numbered folders are a step by step build, one file each, meant to be read in order (see
`CONTRIBUTING.md`). `babysteerling/` and `experiments/` turn that build into a real package and a
runner around it.

- **`0_gpt_chars/`**: A bigram model and a tiny GPT, trained character by character on
  TinyShakespeare. No tokenizer, no interpretability, just next character prediction.
- **`1_gpt_tokens/`**: Same GPT, but with a BPE tokenizer, trained on TinyStories. Includes a
  couple of faster training script variants.
- **`2_atlas/`**: Before a model can use concepts, we need to define them. This folder builds a
  small concept labeled version of TinyStories: an LLM tags text chunks, we cluster the tags into
  a concept library, and assign concepts back to the data. Based on Guide Labs' Atlas pipeline
  (see NOTICE), scaled down to run on a laptop.
- **`3_steerling/`**: Single file implementations of the Steerling architecture: a concept
  bottleneck that splits the model's hidden state into known concepts, unknown concepts, and a
  residual, trained with extra losses so predictions can be traced back to concepts.
  `steerling.py` follows the reference architecture (Causal Diffusion backbone, teacher forcing).
  `gpt_steerling.py` uses a plain GPT backbone instead, simpler and faster to iterate on.
  `gpt_steerling_deephead.py` and `gpt_steerling_no_supervision.py` are small ablations (a deeper
  head; concept losses turned off).
- **`babysteerling/`**: The same architecture and Atlas pipeline as a real, installable package
  (`pip install -e .`). See NOTICE for attribution.
- **`experiments/`**: A Hydra + Weights & Biases runner for `babysteerling`. Use it to train
  models, build datasets, log results, and compare runs, including parallel sweeps. See
  `experiments/README.md` for the full how-to.

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
Atlas work, which this project implements independently at a much smaller scale.
