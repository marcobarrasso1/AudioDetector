"""
Training loop for the Bayes-by-Backprop network (BayesNN).

Mirrors train.py (MCDC) as closely as possible so the comparison is fair:
  - same train manifest, same collate, same class weights
  - same optimizer family / LR / cosine schedule
  - same validation protocol (dev set, every val_every epochs)

Differences forced by the Bayesian treatment:
  - loss is the ELBO:  mean-NLL + beta * KL,  beta = 1/N_train
  - KL warm-up: beta is linearly ramped over the first `kl_warmup` epochs,
    otherwise the KL term dominates early and the net collapses to the prior
  - weight_decay = 0  (the KL term to the N(0, sigma_p) prior IS the
    regularizer — L2 on top of it would double-penalize mu)
  - validation uses eval() — weight sampling stays active regardless of mode
"""

import argparse
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from bayes_model import BayesNN, total_kl
from utils import SpecDataset, AudioDataset, collate_trim_whisper, compute_eer


def run_epoch(model, loader, optimizer, device, is_train,
              class_weights, n_train_samples, beta_scale=1.0,
              scaler=None, max_batches=None):
    """Single epoch. Returns (avg_elbo, avg_nll, avg_kl, eer, accuracy)."""
    model.train() if is_train else model.eval()

    total_elbo = total_nll = 0.0
    total_kl_v = 0.0
    n          = 0
    correct    = 0
    all_scores = []
    all_labels = []

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for i, (x, y) in enumerate(tqdm(loader, desc="train" if is_train else "val ", leave=False)):
            if max_batches is not None and i >= max_batches:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=(scaler is not None)):
                logits = model(x)
                nll    = F.cross_entropy(logits, y, weight=class_weights)

            # KL in fp32, outside autocast — involves log/softplus on the
            # variational parameters and must not run in half precision
            kl   = total_kl(model)
            beta = beta_scale / n_train_samples
            loss = nll.float() + beta * kl

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()

            bs          = x.size(0)
            total_elbo += loss.item() * bs
            total_nll  += nll.item()  * bs
            total_kl_v += kl.item()   * bs
            n          += bs
            correct    += (logits.argmax(dim=1) == y).sum().item()

            prob_real = torch.softmax(logits.float(), dim=1)[:, 0].detach().cpu().numpy()
            target    = (y == 0).long().cpu().numpy()
            all_scores.append(prob_real)
            all_labels.append(target)

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    eer        = compute_eer(all_scores, all_labels)

    return (total_elbo / n, total_nll / n, total_kl_v / n,
            eer, correct / n)


def main(
    train_manifest : str   = "LA/features/train/manifest.csv",
    val_protocol   : str   = "LA/ASVspoof2019.LA.cm.dev.subset.txt",
    val_flac_root  : str   = "LA/ASVspoof2019_LA_dev/flac",
    batch_size     : int   = 64,
    lr             : float = 3e-4,
    epochs         : int   = 40,
    num_workers    : int   = 4,
    kl_warmup      : int   = 5,
    kl_scale       : float = 1.0,
    prior_sigma    : float = 1.0,
    val_every      : int   = 3,
    ckpt_prefix    : str   = "bayes",
    log_dir        : str   = "runs/bayes",
    use_amp        : bool  = True,
    resume         : str   = "auto",
    max_epochs_this_run : int = None,
    smoke          : bool  = False,
):
    # ---- device ----
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device  = torch.device("cpu")
        use_amp = False
    print(f"Device : {device}  |  AMP : {use_amp}", flush=True)

    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    writer = SummaryWriter(log_dir=log_dir)

    # ---- data ----
    df     = pd.read_csv(train_manifest)
    n_real = int((df["label"] == 0).sum())
    n_fake = int((df["label"] == 1).sum())
    n_train_samples = len(df)
    print(f"Train  — real: {n_real}  fake: {n_fake}  total: {n_train_samples}")

    train_ds     = SpecDataset(train_manifest)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_trim_whisper, num_workers=num_workers,
        pin_memory=True,
    )
    class_weights = torch.tensor([n_fake / n_real, 1.0]).to(device)

    val_ds = AudioDataset(protocol_file=val_protocol, flac_root=val_flac_root)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_trim_whisper, num_workers=num_workers,
        pin_memory=True,
    )
    print(f"Validation - {len(val_ds)} samples")

    # ---- model ----
    model   = BayesNN(prior_sigma=prior_sigma).to(device)
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params : {n_param / 1e6:.2f}M (mu+rho)")

    # weight_decay=0: the KL term is the regularizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    # ---- resume (jobs are capped at 2h — training runs as a chain) ----
    start_epoch  = 0
    best_val_eer = 1.0
    last_path    = f"{ckpt_prefix}_last.pt"
    if resume == "auto":
        resume = last_path if os.path.exists(last_path) else None
    if resume:
        ckpt = torch.load(resume, map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch  = ckpt["epoch"]
            best_val_eer = ckpt.get("best_val_eer", 1.0)
        else:
            model.load_state_dict(ckpt)   # plain state_dict
        print(f"Resumed from {resume} (epoch {start_epoch})")

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    for _ in range(start_epoch):
        scheduler.step()

    max_batches = 2 if smoke else None
    done_this_run = 0

    for epoch in range(start_epoch + 1, epochs + 1):
        # KL warm-up: 0 → kl_scale over the first kl_warmup epochs.
        # kl_scale < 1 is a tempered ("cold") posterior — down-weights the
        # prior relative to the likelihood (Wenzel et al., 2020)
        beta_scale = kl_scale * min(1.0, epoch / max(1, kl_warmup))

        elbo, nll, kl, train_eer, train_acc = run_epoch(
            model, train_loader, optimizer, device, is_train=True,
            class_weights=class_weights, n_train_samples=n_train_samples,
            beta_scale=beta_scale, scaler=scaler, max_batches=max_batches,
        )
        scheduler.step()

        writer.add_scalar("ELBO/train",     elbo,            epoch)
        writer.add_scalar("NLL/train",      nll,             epoch)
        writer.add_scalar("KL/train",       kl,              epoch)
        writer.add_scalar("beta_scale",     beta_scale,      epoch)
        writer.add_scalar("EER/train",      train_eer * 100, epoch)
        writer.add_scalar("Accuracy/train", train_acc * 100, epoch)
        writer.add_scalar("LR",             scheduler.get_last_lr()[0], epoch)

        log = (
            f"Epoch {epoch:02d}/{epochs} | "
            f"ELBO {elbo:.4f} | NLL {nll:.4f} | KL {kl:.0f} | "
            f"beta {beta_scale:.2f} | "
            f"EER {train_eer * 100:.2f}% | Acc {train_acc * 100:.2f}%"
        )

        if epoch % val_every == 0 or smoke:
            val_elbo, val_nll, _, val_eer, val_acc = run_epoch(
                model, val_loader, None, device, is_train=False,
                class_weights=class_weights, n_train_samples=n_train_samples,
                beta_scale=beta_scale, scaler=scaler, max_batches=max_batches,
            )
            writer.add_scalar("NLL/val",      val_nll,       epoch)
            writer.add_scalar("EER/val",      val_eer * 100, epoch)
            writer.add_scalar("Accuracy/val", val_acc * 100, epoch)
            writer.add_scalars("EER/train_vs_val", {
                "train": train_eer * 100,
                "val":   val_eer   * 100,
            }, epoch)

            log += (
                f" | val NLL {val_nll:.4f} | "
                f"val EER {val_eer * 100:.2f}% | val Acc {val_acc * 100:.2f}%"
            )

            torch.save(model.state_dict(), f"{ckpt_prefix}_epoch{epoch:02d}.pt")
            if val_eer < best_val_eer:
                best_val_eer = val_eer
                torch.save(model.state_dict(), f"{ckpt_prefix}_best.pt")
                log += "  <- best"

        # full training state — what the next chained job resumes from
        torch.save({
            "epoch":        epoch,
            "model":        model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "best_val_eer": best_val_eer,
        }, last_path)

        print(log, flush=True)

        if epoch == epochs:
            # marker for the self-chaining SLURM script: stop resubmitting
            open(f"{ckpt_prefix}.done", "w").close()

        done_this_run += 1
        if smoke:
            print("Smoke test finished OK")
            break
        if max_epochs_this_run is not None and done_this_run >= max_epochs_this_run:
            print(f"Reached {max_epochs_this_run} epochs this run — handing over to next job")
            break

    writer.close()
    print(f"\nBest val EER : {best_val_eer * 100:.2f}%")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train BayesNN (Bayes by Backprop)")
    p.add_argument("--train_manifest", default="LA/features/train/manifest.csv")
    p.add_argument("--val_protocol",   default="LA/ASVspoof2019.LA.cm.dev.subset.txt")
    p.add_argument("--val_flac_root",  default="LA/ASVspoof2019_LA_dev/flac")
    p.add_argument("--batch_size",     type=int,   default=64)
    p.add_argument("--lr",             type=float, default=3e-4)
    p.add_argument("--epochs",         type=int,   default=40)
    p.add_argument("--num_workers",    type=int,   default=4)
    p.add_argument("--kl_warmup",      type=int,   default=5)
    p.add_argument("--kl_scale",       type=float, default=1.0,
                   help="temper the KL term (0.1 = cold posterior)")
    p.add_argument("--prior_sigma",    type=float, default=1.0)
    p.add_argument("--val_every",      type=int,   default=3)
    p.add_argument("--ckpt_prefix",    default="bayes")
    p.add_argument("--log_dir",        default="runs/bayes")
    p.add_argument("--no_amp",         action="store_true")
    p.add_argument("--resume",         default="auto",
                   help="'auto' resumes from <ckpt_prefix>_last.pt if present; "
                        "'none' starts fresh; or a checkpoint path")
    p.add_argument("--max_epochs_this_run", type=int, default=None,
                   help="stop after N epochs (for chained 2h SLURM jobs)")
    p.add_argument("--smoke",          action="store_true",
                   help="run 2 batches of train+val and exit (sanity check)")
    args = p.parse_args()

    main(
        train_manifest = args.train_manifest,
        val_protocol   = args.val_protocol,
        val_flac_root  = args.val_flac_root,
        batch_size     = args.batch_size,
        lr             = args.lr,
        epochs         = args.epochs,
        num_workers    = args.num_workers,
        kl_warmup      = args.kl_warmup,
        kl_scale       = args.kl_scale,
        prior_sigma    = args.prior_sigma,
        val_every      = args.val_every,
        ckpt_prefix    = args.ckpt_prefix,
        log_dir        = args.log_dir,
        use_amp        = not args.no_amp,
        resume         = None if args.resume == "none" else args.resume,
        max_epochs_this_run = args.max_epochs_this_run,
        smoke          = args.smoke,
    )
