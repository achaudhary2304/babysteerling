# experiments

The [Hydra](https://hydra.cc) + [Weights & Biases](https://wandb.ai) runner around the
`babysteerling` package. Same architecture and dataset pipeline as the package (causal
transformer backbone, concept bottleneck, concept/reconstruction/independence losses,
classification head, and the Atlas tag/cluster/assign/tokenize pipeline), but driven by configs
instead of hardcoded constants, with W&B logging so you can compare runs, including parallel
sweeps, without hand-copying a script per variant.

This folder owns no reusable logic. `train.py` and `build_dataset.py` just unpack a Hydra config
and call into the installed `babysteerling` package.

## Setup

From the repo root:
```bash
pip install -e ".[atlas]"   # editable install of babysteerling, incl. dataset-building deps
wandb login                 # one-time; skip if you plan to use mode=offline (see below)
```

## Building a dataset

`experiments/` builds and owns its own dataset. It doesn't depend on `2_atlas/` or any other
numbered folder. Before training for the first time:
```bash
cd experiments
python build_dataset.py
```
This downloads a raw corpus, trains a tokenizer on it, then runs the four Atlas stages (tag,
cluster, assign, tokenize) in order, writing everything to `atlas.output_dir` (default `./data`).
Every stage is idempotent: it's skipped if already done, so re-running after a partial failure,
or after changing a later stage's hyperparameter, doesn't redo finished work. Override
hyperparameters the same way as for training:
```bash
python build_dataset.py atlas.num_documents=2000 atlas.k=80
```
`train.py`'s default config already points `data.data_dir` at this same `./data`, so once the
build finishes, `python train.py` just works.

**Which corpus gets built and how it's processed are two separate configs.**
`configs/corpus/tinystories.yaml` says what/where the data is: `sources`, a list of
`{name, url, document_delimiter}` entries, plus the shared `tokenizer_vocab_size` and
`boundary_token`. `configs/atlas/default.yaml` says how it's processed: sample size, which models
to use, clustering/dedup thresholds, and the two LLM prompt templates for tagging and labeling.

To build from a different corpus, copy `configs/corpus/tinystories.yaml` to a new file (e.g.
`configs/corpus/my_corpus.yaml`) and replace its `sources` entry. If your corpus isn't children's
stories, also copy `configs/atlas/default.yaml` to `configs/atlas/my_corpus.yaml` and adjust
`tag_prompt_template`/`label_prompt_template`. Then run:
```bash
python build_dataset.py corpus=my_corpus atlas=my_corpus
```

**To train on several corpora at once**, add more entries to `sources`:
```yaml
sources:
  - name: tinystories
    url: https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
    document_delimiter: "<|endoftext|>"
  - name: my_other_corpus
    url: https://example.com/my_other_corpus.txt
    document_delimiter: "\n\n\n"   # can differ per source
```
Each source is downloaded and tagged independently, but all sources' tags are combined into one
shared concept library, and every source is assigned concepts from that same library. This is
what makes it a real union: a concept means the same thing no matter which source a chunk came
from, rather than just concatenating unrelated datasets. `boundary_token` (default
`<|endoftext|>`) is separate from `document_delimiter`: it's the single marker the pipeline
inserts after every chunk once tokenized, and it doesn't need to match any source's delimiter.

## Running a single experiment

```bash
cd experiments
python train.py
```

This uses the defaults in `configs/config.yaml`, which pull in `configs/data/tinystories.yaml`,
`configs/model/steerling_gpt.yaml`, `configs/training/default.yaml`, `configs/wandb/default.yaml`,
`configs/atlas/default.yaml`, and `configs/corpus/tinystories.yaml`.

Override any hyperparameter from the command line, no file editing needed:
```bash
python train.py training.lr=1e-3 training.max_steps=2000
python train.py model=deep_head                          # swap in a whole config variant
python train.py model.unknown_rank=32 model.top_k_known=16
```

Each run prints train/val loss (split into `lm`, `concept`, `rec`, `indep`) every
`training.eval_interval` steps, saves a checkpoint to `./checkpoints/<wandb-run-name>.pt`, and
logs a short sample generation to W&B at the end.

**Which model**: two independent choices -- the backbone (`model.backbone_type`: `causal`
next-token prediction, or `diffusion` masked-diffusion with block-causal attention, matching
`3_steerling/steerling.py`) and whether the concept bottleneck is there at all
(`model.interpretable`). One config group per combination:

| | with bottleneck | no bottleneck |
|---|---|---|
| causal | `model=steerling_gpt` (default) | `model=gpt` |
| diffusion | `model=steerling_diffusion` | `model=diffusion` |

The two no-bottleneck variants are plain language models sharing the same backbone, head and
training loop, so the difference in `val/lm` against their bottlenecked counterpart is what the
bottleneck costs. They report `concept`/`rec`/`indep` as `0.0` so the W&B panels line up. Pair them
with `steering.enabled=false`, since steering needs the bottleneck's concept embeddings.

The `[MASK]` token the diffusion objective needs is added to the tokenizer automatically. Nothing
else needs to change to switch backbones.

## Adjusting configs / adding a new variant

Each config group is a folder under `configs/`: `data/`, `model/`, `training/`, `wandb/`,
`atlas/`, `corpus/`. To change a value permanently, edit the YAML file directly. To add a new
architecture variant without touching `base.yaml`, copy the pattern in
`configs/model/deep_head.yaml`:

```yaml
# configs/model/my_variant.yaml
defaults:
  - base       # inherit everything from base.yaml
  - _self_     # then apply the overrides below on top

unknown_ratio: 5
```

Run it with `python train.py model=my_variant`. The same pattern works for `training/`,
`wandb/`, `atlas/`, or `corpus/` variants.

## Running experiments in parallel (sweeps)

Hydra's multirun mode (`-m`) runs every combination of the listed values as a separate run. Add
the `joblib` launcher to run them in parallel instead of one after another:

```bash
python train.py -m model=steerling_gpt,deep_head training.lr=1e-3,3e-4 \
    hydra/launcher=joblib hydra.launcher.n_jobs=4
```

This launches 4 runs (2 models x 2 learning rates), up to 4 at once. Each gets its own W&B run.

To group a sweep's runs together for comparison, add a shared group:

```bash
python train.py -m model=steerling_gpt,deep_head wandb.group=head_ablation hydra/launcher=joblib
```

The same `-m` mechanism works for `build_dataset.py`, e.g. to build several differently
clustered datasets side by side:
```bash
python build_dataset.py -m atlas.k=80,150,300 atlas.output_dir=./data_k80,./data_k150,./data_k300
```

## Comparing results

Runs are logged to the W&B project named in `configs/wandb/default.yaml`
(`steerling-experiments` by default). On the project page at [wandb.ai](https://wandb.ai) you
can:
- Compare loss curves (`train/*`, `val/*`) across runs on the same chart
- Filter or group by `group`/`tags` (set via `wandb.group=...`/`wandb.tags=[...]`) to isolate one
  sweep from the rest of your history
- Use the Parallel Coordinates panel to see how a hyperparameter (e.g. `model.unknown_rank`)
  relates to final `val/total` loss across many runs
- Read each run's config panel for every resolved hyperparameter, including CLI overrides

We use W&B's UI for this instead of building a separate comparison tool, since it already does
the job well.

### Running without a W&B account

For a quick local test: `python train.py wandb.mode=offline` (writes run data locally under
`./wandb/`, sync later with `wandb sync`), or `python train.py wandb.mode=disabled` to skip W&B
entirely.

## File overview

- `train.py`: thin Hydra + W&B adapter. Unpacks `cfg` into calls against `babysteerling.nn`,
  `babysteerling.loss`, `babysteerling.training`, and `babysteerling.data.utils`, and runs the
  training loop.
- `build_dataset.py`: thin Hydra adapter around `babysteerling.data.atlas`'s four pipeline
  stages.
- `configs/`: Hydra config groups: `data/`, `model/`, `training/`, `wandb/`, `atlas/`, `corpus/`.

The reusable logic (model architecture, losses, training utilities, data loading, the Atlas
pipeline) lives in the installed `babysteerling` package, not here. See the root `README.md` and
`babysteerling/`'s module docstrings.
