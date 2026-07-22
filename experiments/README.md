# experiments

The [Hydra](https://hydra.cc) + [Weights & Biases](https://wandb.ai) runner around the
`babysteerling` package: same architecture and dataset pipeline as the package provides (causal
transformer backbone + concept bottleneck + concept/reconstruction/independence losses +
classification head; the Atlas tag/cluster/assign/tokenize pipeline), driven by configs instead
of hardcoded constants, with W&B logging so runs (including parameter sweeps run in parallel) can
be compared without hand-copying a script per variant.

This folder itself owns no reusable logic -- `train.py` and `build_dataset.py` are thin adapters
that unpack a Hydra config into calls against the installed `babysteerling` package.

## Setup

From the repo root:
```bash
pip install -e ".[atlas]"   # editable install of babysteerling, incl. dataset-building deps
wandb login                 # one-time; skip if you plan to use mode=offline (see below)
```

## Building a dataset

`experiments/` builds and owns its own dataset -- it doesn't depend on `2_atlas/` or any other
numbered folder. Before training for the first time:
```bash
cd experiments
python build_dataset.py
```
This downloads a raw corpus and trains a tokenizer on it (both idempotent, skipped on later
runs), then runs the four Atlas stages (tag, cluster, assign, tokenize) in sequence, writing
everything to `atlas.output_dir` (default `./data`, i.e. `experiments/data/`). Every stage is
idempotent, so re-running after a partial failure, or after only changing a later-stage
hyperparameter, doesn't redo already-finished work. Override any hyperparameter the same way as
for training:
```bash
python build_dataset.py atlas.num_documents=2000 atlas.k=80
```
`train.py`'s default config already points `data.data_dir` at this same `./data`, so once the
build finishes, `python train.py` just works.

**Which corpus gets built vs. how it's processed are separate config concerns.**
`configs/corpus/tinystories.yaml` holds only what/where the data is: `sources` -- a *list* of
`{name, url, document_delimiter}` entries (each source's delimiter can differ) -- plus the
shared `tokenizer_vocab_size`/`boundary_token`. `configs/atlas/default.yaml` holds how it's
processed: sample size, which models to use, clustering/dedup thresholds, and the two LLM
tag/label prompt templates (these live in `atlas` rather than `corpus` since they're inputs to
the atlas pipeline's tagging/labeling stages, the same as `tagging_model` is -- not a property
of the raw data itself).

To build from a different corpus, copy `configs/corpus/tinystories.yaml` to a new file (e.g.
`configs/corpus/my_corpus.yaml`) and replace its `sources` entry. If the new corpus's domain
doesn't fit the default prompts' "children's story" framing, also copy
`configs/atlas/default.yaml` to a matching `configs/atlas/my_corpus.yaml` with adjusted
`tag_prompt_template`/`label_prompt_template`. Then: `python build_dataset.py corpus=my_corpus
atlas=my_corpus`.

**To train on the union of several corpora**, add more entries to `sources`:
```yaml
sources:
  - name: tinystories
    url: https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
    document_delimiter: "<|endoftext|>"
  - name: my_other_corpus
    url: https://example.com/my_other_corpus.txt
    document_delimiter: "\n\n\n"   # can differ per source -- whatever that file actually uses
```
Each source is downloaded and LLM-tagged independently (each split on its own
`document_delimiter`), but every source's tags are combined to build **one shared concept
library**, each source is then assigned concepts from that same library, and everything is
merged into one training set. That shared library is what makes this a real union rather than
just concatenating unrelated datasets -- a concept means the same thing regardless of which
source a chunk came from. The corpus-level `boundary_token` (default `<|endoftext|>`) is a
separate, single marker the pipeline inserts after every chunk once tokenized -- it doesn't need
to match any source's `document_delimiter`, so it stays one consistent value even when sources
use different delimiters.

## Running a single experiment

```bash
cd experiments
python train.py
```

This uses the defaults in `configs/config.yaml` (which pulls in `configs/data/tinystories.yaml`,
`configs/model/base.yaml`, `configs/training/default.yaml`, `configs/wandb/default.yaml`,
`configs/atlas/default.yaml`, `configs/corpus/tinystories.yaml`).

Override any hyperparameter from the command line without editing a file:
```bash
python train.py training.lr=1e-3 training.max_steps=2000
python train.py model=deep_head                          # swap in a whole config variant
python train.py model.unknown_rank=32 model.top_k_known=16
```

Each run prints train/val loss (broken into `lm`, `concept`, `rec`, `indep` components) every
`training.eval_interval` steps, saves a checkpoint to `./checkpoints/<wandb-run-name>.pt`, and
logs a short sample generation to W&B at the end.

## Adjusting configs / adding a new variant

Each config group is a folder under `configs/`: `data/`, `model/`, `training/`, `wandb/`,
`atlas/`, `corpus/`. To tweak a value permanently, edit the relevant YAML file directly. To add a
new architecture variant without touching `base.yaml`, copy the pattern used by
`configs/model/deep_head.yaml`:

```yaml
# configs/model/my_variant.yaml
defaults:
  - base       # inherit everything from base.yaml
  - _self_     # then apply the overrides below on top

unknown_ratio: 5
```

Then run it with `python train.py model=my_variant`. The same pattern works for `training/`,
`wandb/`, `atlas/`, or `corpus/` variants.

## Running experiments in parallel (sweeps)

Hydra's multirun mode (`-m`) runs every combination of the listed values as a separate run. Add
the `joblib` launcher to actually parallelize them on your machine instead of running
sequentially:

```bash
python train.py -m model=base,deep_head training.lr=1e-3,3e-4 \
    hydra/launcher=joblib hydra.launcher.n_jobs=4
```

This launches 4 runs (2 models × 2 learning rates), up to 4 of them in parallel worker
processes at once. Each gets its own W&B run.

To tie a sweep's runs together for comparison, add a shared group. This simpler example sweeps
just the model (2 runs) and groups them:

```bash
python train.py -m model=base,deep_head wandb.group=head_ablation hydra/launcher=joblib
```

The same `-m` mechanism works for `build_dataset.py` too, e.g. to build several differently
clustered datasets side by side: `python build_dataset.py -m atlas.k=80,150,300
atlas.output_dir=./data_k80,./data_k150,./data_k300`.

## Comparing results

Runs are logged to the W&B project named in `configs/wandb/default.yaml` (`project:
steerling-experiments` by default). Open the project page on [wandb.ai](https://wandb.ai) to:
- Compare loss curves (`train/*`, `val/*`) across runs on the same chart
- Filter/group by `group` or `tags` (set via `wandb.group=...` / `wandb.tags=[...]`) to isolate
  one sweep from the rest of your run history
- Use the **Parallel Coordinates** panel to see how a hyperparameter (e.g. `model.unknown_rank`)
  relates to final `val/total` loss across many runs at once
- Read each run's config panel to see every resolved hyperparameter, including ones set via CLI
  override

We rely on W&B's UI for this rather than building a separate local comparison tool, since it
already does run comparison well and a second tool would just duplicate it.

### Running without a W&B account

For a quick local test without logging in: `python train.py wandb.mode=offline` (writes run
data locally under `./wandb/`, sync later with `wandb sync` if desired), or `python train.py
wandb.mode=disabled` to skip W&B entirely.

## File overview

- `train.py` — thin Hydra + W&B adapter: unpacks `cfg` into calls against `babysteerling.nn`,
  `babysteerling.loss`, `babysteerling.training`, and `babysteerling.data.utils`, and wires them
  into a training loop
- `build_dataset.py` — thin Hydra adapter around `babysteerling.data.atlas`'s four pipeline
  stages
- `configs/` — Hydra config groups: `data/`, `model/`, `training/`, `wandb/`, `atlas/`

The reusable logic itself (model architecture, losses, training utilities, data loading, the
Atlas pipeline) lives in the installed `babysteerling` package, not in this folder -- see the
root `README.md` and `babysteerling/`'s module docstrings.
