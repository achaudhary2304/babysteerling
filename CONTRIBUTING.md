# Contributing

Thanks for your interest in babysteerling. This is a small, didactic project, so the bar for
contributions is "does it stay simple and readable" more than "does it cover every edge case."

## Development setup

```bash
git clone https://github.com/pietrobarbiero/babysteerling.git
cd babysteerling
pip install -e ".[atlas]"   # editable install; the atlas extra pulls in the dataset-building deps
```

## Repository layout

- `babysteerling/` — the installable package: model (`nn.py`), losses (`loss.py`), training
  utilities (`training.py`), and the Atlas dataset pipeline (`data/`).
- `experiments/` — the Hydra + Weights & Biases runner around the package: configs, `train.py`,
  `build_dataset.py`. See `experiments/README.md`.
- `0_gpt_chars/`, `1_gpt_tokens/`, `2_atlas/`, `3_steerling/` — standalone, single-file didactic
  scripts documenting the build-up to `babysteerling`. These are intentionally self-contained and
  don't depend on the package; please don't refactor them to import from `babysteerling` (that
  would defeat the point of being readable top-to-bottom in isolation), and please don't make
  `babysteerling`/`experiments` depend on them either (see `experiments/README.md`'s "Building a
  dataset" section -- `build_dataset.py` fetches and builds everything it needs on its own).

## Making a change

1. Open an issue or a small PR describing what you're changing and why.
2. Keep changes minimal and consistent with the existing style: flat, commented, explicit shapes
   on tensor operations (see any file in `babysteerling/` for the convention), no abstraction
   beyond what the change actually needs.
3. Sanity-check your change runs end-to-end before opening a PR (there's no CI yet) -- e.g. a
   short `experiments/train.py` run with `training.max_steps` overridden to a small number.
4. Submit a PR against `main`.

By contributing, you agree your contributions are licensed under the project's Apache License,
Version 2.0 (see `LICENSE`).
