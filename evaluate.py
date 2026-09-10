"""
Unified evaluation for both models on an ASVspoof protocol.

For the chosen model (--model mcdc | bayes) it runs:
  1. a deterministic pass   (MCDC: dropout off / BayesNN: posterior-mean weights)
  2. a Monte Carlo pass     (MCDC: MC dropout   / BayesNN: weight sampling)

and reports EER, accuracy, ECE, NLL, Brier, mean BALD, BALD-vs-error AUROC,
AURC and EER@90% coverage. Per-sample results go to a CSV, scalars to a JSON —
analyze.py builds the comparison plots from those files.

Usage (on a GPU node):
  python evaluate.py --model mcdc  --ckpt mcdc_epoch36.pt
  python evaluate.py --model bayes --ckpt bayes_best.pt
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mcd_model import MCDC
from bayes_model import BayesNN
from utils import AudioDataset, collate_trim_whisper
import metrics


@torch.no_grad()
def run_deterministic(model, loader, device, is_bayes):
    """Single deterministic forward pass. Returns (scores, labels) as numpy."""
    model.eval()
    if is_bayes:
        model.set_sampling(False)    # posterior mean weights

    all_scores, all_labels = [], []
    for x, y in tqdm(loader, desc="det eval"):
        x = x.to(device, non_blocking=True)
        prob = torch.softmax(model(x).float(), dim=1)
        all_scores.append(prob[:, 0].cpu().numpy())
        all_labels.append((y == 0).long().numpy())

    if is_bayes:
        model.set_sampling(True)
    return np.concatenate(all_scores), np.concatenate(all_labels)


@torch.no_grad()
def run_mc(model, loader, device, n_samples):
    """MC pass via model.mc_predict. Returns dict of numpy arrays."""
    all_mean, all_var, all_bald, all_labels = [], [], [], []
    for x, y in tqdm(loader, desc=f"mc eval ({n_samples})"):
        x = x.to(device, non_blocking=True)
        mean_prob, variance, bald = model.mc_predict(x, n_samples=n_samples)
        all_mean.append(mean_prob.cpu().numpy())
        all_var.append(variance.cpu().numpy())
        all_bald.append(bald.cpu().numpy())
        all_labels.append((y == 0).long().numpy())

    return {
        "mean_probs": np.concatenate(all_mean),
        "variance":   np.concatenate(all_var),
        "bald":       np.concatenate(all_bald),
        "labels":     np.concatenate(all_labels),
    }


def main():
    p = argparse.ArgumentParser(description="Evaluate MCDC / BayesNN on ASVspoof")
    p.add_argument("--model",         required=True, choices=["mcdc", "bayes"])
    p.add_argument("--ckpt",          required=True)
    p.add_argument("--protocol_file", default="LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt")
    p.add_argument("--flac_root",     default="LA/ASVspoof2019_LA_eval/flac")
    p.add_argument("--batch_size",    type=int,   default=64)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--mc_samples",    type=int,   default=30)
    p.add_argument("--p_drop",        type=float, default=0.3)
    p.add_argument("--prior_sigma",   type=float, default=1.0)
    p.add_argument("--skip_det",      action="store_true")
    p.add_argument("--out_dir",       default="results")
    p.add_argument("--tag",           default=None,
                   help="output filename prefix (default: model name)")
    p.add_argument("--shard_idx",     type=int, default=0,
                   help="which contiguous shard of the protocol to evaluate")
    p.add_argument("--num_shards",    type=int, default=1,
                   help="split eval across N jobs (account MaxWall is 2h); "
                        "merge the shard CSVs with analyze.py")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device    : {device}")
    print(f"Model     : {args.model}")
    print(f"Checkpoint: {args.ckpt}")

    # ---- data ----
    ds = AudioDataset(protocol_file=args.protocol_file, flac_root=args.flac_root)

    shard_lo, shard_hi = 0, len(ds.samples)
    if args.num_shards > 1:
        bounds   = np.linspace(0, len(ds.samples), args.num_shards + 1).astype(int)
        shard_lo, shard_hi = bounds[args.shard_idx], bounds[args.shard_idx + 1]
        ds.samples = ds.samples[shard_lo:shard_hi]
        print(f"Shard     : {args.shard_idx + 1}/{args.num_shards} "
              f"(rows {shard_lo}:{shard_hi})")

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_trim_whisper, num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Samples   : {len(ds)}")

    # ---- model ----
    if args.model == "mcdc":
        model = MCDC(p_drop=args.p_drop).to(device)
    else:
        model = BayesNN(prior_sigma=args.prior_sigma).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    os.makedirs(args.out_dir, exist_ok=True)
    tag = args.tag or args.model
    if args.num_shards > 1:
        tag += f"_shard{args.shard_idx}"
    summary = {"model": args.model, "ckpt": args.ckpt,
               "protocol": args.protocol_file, "mc_samples": args.mc_samples}

    # ---- deterministic pass ----
    det_scores = None
    if not args.skip_det:
        det_scores, labels = run_deterministic(model, loader, device,
                                               is_bayes=(args.model == "bayes"))
        det = metrics.summarize(det_scores, labels)
        summary["det"] = det
        print("\n-- Deterministic (point estimate) --")
        for k, v in det.items():
            print(f"  {k:16s}: {v:.4f}")

    # ---- MC pass ----
    mc = run_mc(model, loader, device, n_samples=args.mc_samples)
    labels    = mc["labels"]
    mc_scores = mc["mean_probs"][:, 0]
    mc_sum    = metrics.summarize(mc_scores, labels, uncertainty=mc["bald"])
    summary["mc"] = mc_sum
    print(f"\n-- MC / posterior predictive ({args.mc_samples} samples) --")
    for k, v in mc_sum.items():
        print(f"  {k:16s}: {v:.4f}")

    # ---- per-sample CSV (protocol order == loader order, shuffle=False) ----
    df = pd.read_csv(
        args.protocol_file, sep=" ", header=None,
        names=["speaker", "utterance", "col3", "attack", "label_str"],
    )
    df = df.iloc[shard_lo:shard_hi].reset_index(drop=True)
    df["label"] = (df["label_str"] == "spoof").astype(int)   # 0=real, 1=fake
    if det_scores is not None:
        df["prob_real_det"] = det_scores
    df["prob_real_mc"] = mc_scores
    df["prob_fake_mc"] = mc["mean_probs"][:, 1]
    df["bald"]         = mc["bald"]
    df["variance"]     = mc["variance"][:, 0]
    df["pred_mc"]      = (mc_scores < 0.5).astype(int)
    df["correct_mc"]   = (df["pred_mc"] == df["label"]).astype(int)

    csv_path = os.path.join(args.out_dir, f"{tag}_eval.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nPer-sample CSV -> {csv_path}")

    json_path = os.path.join(args.out_dir, f"{tag}_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON   -> {json_path}")


if __name__ == "__main__":
    main()
