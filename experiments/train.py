"""Minimal Hydra + W&B training entry point.

Kept thin on purpose: every non-trivial piece of logic (data loading/batching, model
architecture, loss computation, training-loop utilities) lives in the installed `babysteerling`
package. This file only unpacks Hydra's `cfg` into plain calls against that package and wires
them into a training loop -- sample a batch, forward, compute loss, step, occasionally evaluate
and log -- so the actual experiment logic stays easy to read in one place and easy to swap via
config groups without touching this file at all.

Run a single experiment:      python train.py
Override any hyperparameter:  python train.py training.lr=1e-3 model=deep_head
Run a parallel sweep:         python train.py -m model=base,deep_head training.lr=1e-3,3e-4
(see README.md for the full explanation of config overrides and multirun)
"""
import os
import time

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from babysteerling.data.utils import load_dataset, load_tokenizer
from babysteerling.nn import build_model
from babysteerling.training import estimate_loss, get_lr, run_batch


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

    # W&B tracks the fully-resolved config, so every hyperparameter (including ones set via
    # CLI override or a swapped config group like model=deep_head) shows up in the run's config
    # panel and can be filtered/grouped on in the UI
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=cfg.wandb.group,
        tags=list(cfg.wandb.tags),
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    tok, vocab_size, decode = load_tokenizer(cfg.data.data_dir)
    tokens, doc_records, n_concepts = load_dataset(cfg.data.data_dir)
    n_train = int(0.9 * len(tokens))  # 90/10 train/val split
    doc_starts = [d['start'] for d in doc_records]  # sorted (documents are laid out in order), used for binary search
    print(f"Loaded {len(tokens)} tokens, {len(doc_records)} documents, {n_concepts} known concepts")

    # babysteerling.nn.build_model takes plain kwargs (no Hydra dependency), so this is the
    # adapter line that unpacks the `model` config group into that call
    model = build_model(
        vocab_size=vocab_size, n_concepts=n_concepts, block_size=cfg.data.block_size,
        n_embed=cfg.model.n_embed, num_heads=cfg.model.num_heads, num_kv_heads=cfg.model.num_kv_heads,
        n_layers=cfg.model.n_layers, dropout=cfg.model.dropout, unknown_ratio=cfg.model.unknown_ratio,
        p_epsilon=cfg.model.p_epsilon, unknown_rank=cfg.model.unknown_rank,
        top_k_known=cfg.model.top_k_known, top_k_unknown=cfg.model.top_k_unknown,
        head_type=cfg.model.head.type, tie_weights=cfg.model.head.tie_weights,
        head_mlp_hidden=cfg.model.head.mlp_hidden,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"{n_params:.3f} M params")
    wandb.run.summary['n_params_M'] = n_params

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)

    for step in range(cfg.training.max_steps + 1):
        current_lr = get_lr(step, cfg.training.lr, cfg.training.min_lr,
                             cfg.training.warmup_steps, cfg.training.max_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        train_loss, _ = run_batch(
            model, tokens, doc_records, doc_starts, n_train, n_concepts, 'train',
            cfg.data.block_size, cfg.data.batch_size, device,
            lambda_concept=cfg.training.lambda_concept, lambda_rec=cfg.training.lambda_rec,
            lambda_indep=cfg.training.lambda_indep,
        )

        optimizer.zero_grad(set_to_none=True)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
        optimizer.step()

        if step % cfg.training.eval_interval == 0:
            losses = estimate_loss(
                model, tokens, doc_records, doc_starts, n_train, n_concepts,
                cfg.data.block_size, cfg.data.batch_size, device, cfg.training.eval_iters,
                lambda_concept=cfg.training.lambda_concept, lambda_rec=cfg.training.lambda_rec,
                lambda_indep=cfg.training.lambda_indep,
            )
            print(f"step {step}, train loss {losses['train']['total']:.4f}, "
                  f"val loss {losses['val']['total']:.4f}, lr {current_lr:.6f}")
            # prefix keys so W&B groups train/*.total and val/*.total as separate lines on the
            # same chart, and each loss component gets its own comparable panel across runs
            wandb.log({
                'step': step,
                'lr': current_lr,
                **{f'train/{k}': v for k, v in losses['train'].items()},
                **{f'val/{k}': v for k, v in losses['val'].items()},
            })

    # each run gets its own checkpoint (named after the W&B run) rather than a
    # skip-if-exists pattern -- an experiment sweep is expected to produce many runs, not
    # resume a single canonical one
    run_name = wandb.run.name or f"run_{int(time.time())}"
    os.makedirs("./checkpoints", exist_ok=True)
    ckpt_path = os.path.join("./checkpoints", f"{run_name}.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    # sample a short generation and log it to W&B as text, so a run's qualitative output is
    # visible next to its loss curves without needing to reload the checkpoint separately
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    sample_ids = model.generate(
        idx, max_new_tokens=cfg.training.gen_max_new_tokens,
        temperature=cfg.training.gen_temperature, top_k=cfg.training.gen_top_k,
    )[0].tolist()
    sample_text = decode(sample_ids)
    print(sample_text)
    wandb.log({'sample': wandb.Html(f"<pre>{sample_text}</pre>")})

    wandb.finish()


if __name__ == "__main__":
    main()
