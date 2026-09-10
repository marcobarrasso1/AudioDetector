"""
Comparison & plots from per-sample evaluation CSVs.  No torch needed —
runs on any machine with numpy/pandas/matplotlib/sklearn.

Input CSVs (from evaluate.py, or the legacy eval_results.csv) must contain:
  label (0=real, 1=fake), prob_real_mc, bald
optional: prob_real_det, attack

Usage:
  python analyze.py --csv mcdc=results/mcdc_eval.csv bayes=results/bayes_eval.csv
  python analyze.py --csv mcdc=eval_results.csv          # single model works too

Outputs (in --out_dir, default results/):
  summary.csv / summary.md      — all scalar metrics, one row per model & mode
  reliability.png               — reliability diagrams (det vs MC, per model)
  bald_hist.png                 — BALD distribution, correct vs misclassified
  risk_coverage.png             — selective-prediction curves
  eer_per_attack.png            — EER broken down by attack type A07-A19
"""

import argparse
import glob as globlib
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import metrics

# ---------------------------------------------------------------------------
# Plot style (light surface, report-ready)
# ---------------------------------------------------------------------------

SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_2     = "#52514e"
MUTED     = "#898781"
GRID      = "#e1e0d9"
BASELINE  = "#c3c2b7"
SERIES    = {"mcdc": "#2a78d6", "bayes": "#1baf7a"}   # blue / aqua
FALLBACK  = ["#2a78d6", "#1baf7a", "#eda100", "#4a3aa7"]

plt.rcParams.update({
    "figure.facecolor":  SURFACE,
    "axes.facecolor":    SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family":       "sans-serif",
    "text.color":        INK,
    "axes.edgecolor":    BASELINE,
    "axes.labelcolor":   INK_2,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "axes.grid":         True,
    "grid.color":        GRID,
    "grid.linewidth":    0.8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.titlesize":    11,
    "axes.titlecolor":   INK,
    "font.size":         10,
    "legend.frameon":    False,
})

PRETTY = {"mcdc": "MC Dropout", "bayes": "Bayes by Backprop"}


def color_of(name, i):
    return SERIES.get(name, FALLBACK[i % len(FALLBACK)])


def label_of(name):
    return PRETTY.get(name, name)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_csvs(specs):
    """specs: list of 'name=path' (path may be a glob over shard CSVs).
    Returns {name: dataframe}."""
    out = {}
    for spec in specs:
        name, path = spec.split("=", 1)
        files = sorted(globlib.glob(path)) if any(c in path for c in "*?[") else [path]
        if not files:
            raise FileNotFoundError(f"no files match {path}")
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        # convention in metrics.py: labels 1=real. CSVs store 0=real,1=fake.
        df["y_real"] = (df["label"] == 0).astype(int)
        out[name] = df
        print(f"loaded {name:8s} {path}  ({len(files)} file(s), {len(df)} rows)")
    return out


def modes_of(df):
    """Which score columns exist -> list of (mode, column)."""
    modes = []
    if "prob_real_det" in df.columns:
        modes.append(("det", "prob_real_det"))
    modes.append(("mc", "prob_real_mc"))
    return modes


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def build_summary(data):
    rows = []
    for name, df in data.items():
        y = df["y_real"].to_numpy()
        for mode, col in modes_of(df):
            s   = df[col].to_numpy()
            unc = df["bald"].to_numpy() if mode == "mc" else None
            m   = metrics.summarize(s, y, uncertainty=unc)
            row = {"model": name, "mode": mode}
            row.update(m)
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_reliability(data, out):
    n = len(data)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.2), squeeze=False)
    for i, (name, df) in enumerate(data.items()):
        ax = axes[0][i]
        y  = df["y_real"].to_numpy()
        ax.plot([0.5, 1], [0.5, 1], "--", color=BASELINE, lw=1.2, zorder=1)
        for j, (mode, col) in enumerate(modes_of(df)):
            s = df[col].to_numpy()
            ece, bins = metrics.compute_ece(s, y)
            ok = ~np.isnan(bins["accuracy"])
            style = dict(color=color_of(name, i), lw=2)
            if mode == "det":
                style.update(alpha=0.45, linestyle=":")
            lbl = f"{'point estimate' if mode == 'det' else 'posterior predictive'}  (ECE {ece*100:.2f}%)"
            ax.plot(bins["centers"][ok], bins["accuracy"][ok],
                    marker="o", ms=4.5, label=lbl, zorder=3, **style)
        ax.set_xlim(0.5, 1.0); ax.set_ylim(0.4, 1.02)
        ax.set_xlabel("confidence")
        ax.set_ylabel("empirical accuracy" if i == 0 else "")
        ax.set_title(label_of(name))
        ax.legend(loc="lower right", fontsize=9)
    fig.suptitle("Reliability diagrams — perfectly calibrated = diagonal",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"wrote {out}")


def plot_bald_hist(data, out, min_bin_count=5):
    """min_bin_count: bins are trimmed once EITHER group's tail count drops
    below this — a couple of stray far-tail samples otherwise render as an
    isolated, unexplainable bar (e.g. 2 samples all landing 'correct')."""
    n = len(data)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.0), squeeze=False)
    for i, (name, df) in enumerate(data.items()):
        ax   = axes[0][i]
        y    = df["y_real"].to_numpy()
        s    = df["prob_real_mc"].to_numpy()
        bald = df["bald"].to_numpy()
        correct = ((s >= 0.5).astype(int) == y)

        # find a trimmed upper edge: walk fine-grained bins from the right,
        # drop the tail until both groups clear min_bin_count
        probe_bins = np.linspace(0, max(bald.max(), 1e-6), 60)
        cnt_c, _ = np.histogram(bald[correct], bins=probe_bins)
        cnt_w, _ = np.histogram(bald[~correct], bins=probe_bins)
        keep = np.where((cnt_c >= min_bin_count) & (cnt_w >= min_bin_count))[0]
        bald_max = probe_bins[keep[-1] + 1] if len(keep) else bald.max()

        bins = np.linspace(0, max(bald_max, 1e-6), 40)
        ax.hist(bald[correct], bins=bins, density=True, alpha=0.75,
                color=color_of(name, i), label=f"correct (n={correct.sum()})")
        ax.hist(bald[~correct], bins=bins, density=True, alpha=0.75,
                color="#e34948", label=f"misclassified (n={(~correct).sum()})")
        ax.set_yscale("log")
        ax.set_xlabel("BALD (epistemic uncertainty)")
        ax.set_ylabel("density (log)" if i == 0 else "")
        ax.set_title(label_of(name))
        ax.legend(fontsize=9)
    fig.suptitle("BALD separates the model's own mistakes", fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"wrote {out}")


def plot_bald_hist_eer_thresh(data, out, min_bin_count=5):
    """Same as plot_bald_hist, but 'correct' is defined at each model's own
    EER threshold (metrics.compute_eer) instead of the fixed 0.5 cutoff —
    the threshold each model would actually be operated at.

    min_bin_count: bins are trimmed once EITHER group's tail count drops
    below this — otherwise a thin, all-one-color tail (e.g. MCDC's few
    remaining high-BALD samples all landing 'correct' by chance) renders as
    a misleading zero-misclassified region."""
    n = len(data)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.0), squeeze=False)
    for i, (name, df) in enumerate(data.items()):
        ax   = axes[0][i]
        y    = df["y_real"].to_numpy()
        s    = df["prob_real_mc"].to_numpy()
        bald = df["bald"].to_numpy()
        _, thr = metrics.compute_eer(s, y)
        correct = ((s >= thr).astype(int) == y)

        probe_bins = np.linspace(0, max(bald.max(), 1e-6), 60)
        cnt_c, _ = np.histogram(bald[correct], bins=probe_bins)
        cnt_w, _ = np.histogram(bald[~correct], bins=probe_bins)
        keep = np.where((cnt_c >= min_bin_count) & (cnt_w >= min_bin_count))[0]
        bald_max = probe_bins[keep[-1] + 1] if len(keep) else bald.max()

        bins = np.linspace(0, max(bald_max, 1e-6), 40)
        ax.hist(bald[correct], bins=bins, density=True, alpha=0.75,
                color=color_of(name, i), label=f"correct (n={correct.sum()})")
        ax.hist(bald[~correct], bins=bins, density=True, alpha=0.75,
                color="#e34948", label=f"misclassified (n={(~correct).sum()})")
        ax.set_yscale("log")
        ax.set_xlabel("BALD (epistemic uncertainty)")
        ax.set_ylabel("density (log)" if i == 0 else "")
        ax.set_title(f"{label_of(name)}  (thr={thr:.2f})")
        ax.legend(fontsize=9)
    fig.suptitle("BALD separates the model's own mistakes — at each model's EER threshold",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"wrote {out}")


def plot_risk_coverage(data, out):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for i, (name, df) in enumerate(data.items()):
        y    = df["y_real"].to_numpy()
        s    = df["prob_real_mc"].to_numpy()
        bald = df["bald"].to_numpy()
        cov, risk, aurc = metrics.risk_coverage_curve(bald, s, y, n_points=200)
        ax.plot(cov * 100, risk * 100, lw=2, color=color_of(name, i),
                label=f"{label_of(name)}  (AURC {aurc*100:.2f}%)")
        # full-coverage error rate as reference point
        ax.plot([100], [risk[-1] * 100], "o", ms=5, color=color_of(name, i))
    ax.set_xlabel("coverage — % of samples kept (most-certain first)")
    ax.set_ylabel("error rate on kept samples (%)")
    ax.set_title("Risk-coverage: rejecting by BALD uncertainty")
    ax.set_xlim(0, 102)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"wrote {out}")


def plot_eer_per_attack(data, out):
    # attacks present in any csv (A07..A19 on the eval protocol)
    all_attacks = sorted(set().union(
        *[set(df["attack"].unique()) for df in data.values()
          if "attack" in df.columns]
    ) - {"-"})
    if not all_attacks:
        print("no attack column — skipping per-attack plot")
        return

    fig, ax = plt.subplots(figsize=(max(7, 0.6 * len(all_attacks) + 2), 4.2))
    width = 0.8 / len(data)
    xs    = np.arange(len(all_attacks))

    for i, (name, df) in enumerate(data.items()):
        if "attack" not in df.columns:
            continue
        eers = []
        bona = df[df["attack"] == "-"]          # bonafide rows
        for atk in all_attacks:
            sub = pd.concat([bona, df[df["attack"] == atk]])
            eer, _ = metrics.compute_eer(sub["prob_real_mc"].to_numpy(),
                                         sub["y_real"].to_numpy())
            eers.append(eer * 100)
        ax.bar(xs + i * width - 0.4 + width / 2, eers, width * 0.92,
               color=color_of(name, i), label=label_of(name), zorder=3)

    ax.set_xticks(xs)
    ax.set_xticklabels(all_attacks)
    ax.set_ylabel("EER (%)")
    ax.set_title("EER per attack type (bonafide vs single attack)")
    ax.legend(fontsize=9)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"wrote {out}")


def plot_calibration_compact(data, out):
    """Single-panel reliability diagram: posterior-predictive curves only,
    ECE in the legend. The gap to the diagonal IS the calibration error."""
    label_override = {"bayes": "Bayesian NN"}
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.plot([0.5, 1], [0.5, 1], "--", color=BASELINE, lw=1.2, zorder=1,
            label="perfect calibration")
    for i, (name, df) in enumerate(data.items()):
        y = df["y_real"].to_numpy()
        s = df["prob_real_mc"].to_numpy()
        ece, bins = metrics.compute_ece(s, y)
        ok = ~np.isnan(bins["accuracy"])
        lbl = label_override.get(name, label_of(name))
        ax.plot(bins["centers"][ok], bins["accuracy"][ok], marker="o", ms=5,
                lw=2, color=color_of(name, i),
                label=f"{lbl}  (ECE {ece*100:.2f}%)", zorder=3)
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0.4, 1.02)
    ax.set_xlabel("confidence")
    ax.set_ylabel("empirical accuracy")
    ax.set_title("Calibration: is a stated confidence of x% right x% of the time?\n"
                 "(above diagonal = underconfident, below = overconfident)")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"wrote {out}")


def plot_eer_vs_rejection(data, out, max_reject=20):
    """EER on the retained samples after rejecting the most-uncertain x%
    by BALD. Justifies the uncertainty in the units of the task metric."""
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    rejects = np.arange(0, max_reject + 1, 2.5)
    for i, (name, df) in enumerate(data.items()):
        y    = df["y_real"].to_numpy()
        s    = df["prob_real_mc"].to_numpy()
        bald = df["bald"].to_numpy()
        order = np.argsort(bald)            # most certain first
        eers = []
        for r in rejects:
            keep = order[: int(len(y) * (1 - r / 100))]
            eer, _ = metrics.compute_eer(s[keep], y[keep])
            eers.append(eer * 100)
        ax.plot(rejects, eers, marker="o", ms=4.5, lw=2,
                color=color_of(name, i), label=label_of(name))
        # annotate start and end
        ax.annotate(f"{eers[0]:.2f}%", (rejects[0], eers[0]),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=9, color=INK_2)
        ax.annotate(f"{eers[-1]:.2f}%", (rejects[-1], eers[-1]),
                    textcoords="offset points", xytext=(-4, 8),
                    fontsize=9, color=INK_2, ha="right")
    ax.set_xlabel("rejected fraction — most-uncertain samples by BALD (%)")
    ax.set_ylabel("EER on retained samples (%)")
    ax.set_title("Uncertainty pays in the task metric:\nEER after rejecting the least-trusted predictions")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"wrote {out}")


def plot_attack_scatter(data, out):
    """One point per attack type: EER (vs bonafide) against mean BALD on that
    attack's trials. Correlation = uncertainty anticipates difficulty."""
    if not any("attack" in df.columns for df in data.values()):
        print("no attack column — skipping attack scatter")
        return
    fig, axes = plt.subplots(1, len(data), figsize=(5.4 * len(data), 4.4),
                             squeeze=False)
    for i, (name, df) in enumerate(data.items()):
        ax = axes[0][i]
        bona    = df[df["attack"] == "-"]
        attacks = sorted(set(df["attack"].unique()) - {"-"})
        xs, ys = [], []
        for atk in attacks:
            grp = df[df["attack"] == atk]
            sub = pd.concat([bona, grp])
            eer, _ = metrics.compute_eer(sub["prob_real_mc"].to_numpy(),
                                         sub["y_real"].to_numpy())
            xs.append(grp["bald"].mean())
            ys.append(eer * 100)
            ax.annotate(atk, (xs[-1], ys[-1]), textcoords="offset points",
                        xytext=(5, 4), fontsize=8, color=INK_2)
        ax.scatter(xs, ys, s=42, color=color_of(name, i), zorder=3)
        rho = float(np.corrcoef(xs, ys)[0, 1])
        ax.set_xlabel("mean BALD on the attack's trials")
        ax.set_ylabel("EER vs bonafide (%)" if i == 0 else "")
        ax.set_title(f"{label_of(name)}  (r = {rho:.2f})")
    fig.suptitle("Does the model's uncertainty anticipate which attacks are hard?",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"wrote {out}")


# In the LA eval protocol, A16/A19 are the same algorithms as training
# attacks A04/A06 — the rest are genuinely novel spoofing methods.
SEEN_ATTACKS = {"A16", "A19"}


def plot_seen_unseen(data, out):
    """EER and BALD for known-algorithm vs novel-algorithm eval attacks.
    Tests whether epistemic uncertainty reacts to distribution shift."""
    if not any("attack" in df.columns for df in data.values()):
        print("no attack column — skipping seen/unseen plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    groups = ["known algorithms\n(A16, A19)", "novel algorithms\n(A07-A15, A17, A18)"]
    xs     = np.arange(len(groups))
    width  = 0.8 / len(data)

    for i, (name, df) in enumerate(data.items()):
        if "attack" not in df.columns:
            continue
        bona   = df[df["attack"] == "-"]
        seen   = df[df["attack"].isin(SEEN_ATTACKS)]
        unseen = df[(df["attack"] != "-") & ~df["attack"].isin(SEEN_ATTACKS)]

        # EER: bonafide vs each spoof group
        eers = []
        for grp in (seen, unseen):
            sub = pd.concat([bona, grp])
            eer, _ = metrics.compute_eer(sub["prob_real_mc"].to_numpy(),
                                         sub["y_real"].to_numpy())
            eers.append(eer * 100)
        off = i * width - 0.4 + width / 2
        ax1.bar(xs + off, eers, width * 0.92, color=color_of(name, i),
                label=label_of(name), zorder=3)

        # mean BALD on the spoofed trials of each group
        balds = [grp["bald"].mean() for grp in (seen, unseen)]
        ax2.bar(xs + off, balds, width * 0.92, color=color_of(name, i),
                label=label_of(name), zorder=3)

    for ax, ylab, title in (
        (ax1, "EER (%)", "Discrimination: EER vs bonafide"),
        (ax2, "mean BALD on spoofed trials", "Epistemic uncertainty (BALD)"),
    ):
        ax.set_xticks(xs)
        ax.set_xticklabels(groups)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
        ax.legend(fontsize=9, loc="upper left")
    fig.suptitle("Known vs novel attack algorithms — does the model know "
                 "when it's out of distribution?", fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Compare evaluation CSVs")
    p.add_argument("--csv", nargs="+", required=True,
                   metavar="name=path", help="e.g. mcdc=results/mcdc_eval.csv")
    p.add_argument("--out_dir", default="results")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data = load_csvs(args.csv)

    summary = build_summary(data)
    pct_cols = ["eer", "accuracy", "balanced_acc", "ece", "eer_at_90cov"]
    disp = summary.copy()
    for c in pct_cols:
        if c in disp.columns:
            disp[c] = disp[c] * 100
    print("\n=== SUMMARY (eer/accuracy/ece/eer@90cov in %) ===")
    print(disp.round(4).to_string(index=False))

    summary.to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
    def cell(v):
        if isinstance(v, float):
            return "-" if np.isnan(v) else f"{v:.4f}"
        return str(v)
    d = disp.round(4)
    md = ("| " + " | ".join(d.columns) + " |\n"
          + "|" + "---|" * len(d.columns) + "\n"
          + "\n".join("| " + " | ".join(cell(v) for v in r) + " |" for r in d.values))
    with open(os.path.join(args.out_dir, "summary.md"), "w") as f:
        f.write(md + "\n")
    print(f"\nwrote {args.out_dir}/summary.csv and summary.md")

    plot_reliability(data,   os.path.join(args.out_dir, "reliability.png"))
    plot_bald_hist(data,     os.path.join(args.out_dir, "bald_hist.png"))
    plot_bald_hist_eer_thresh(data, os.path.join(args.out_dir, "bald_hist_eer_thresh.png"))
    plot_risk_coverage(data, os.path.join(args.out_dir, "risk_coverage.png"))
    plot_eer_per_attack(data, os.path.join(args.out_dir, "eer_per_attack.png"))
    plot_seen_unseen(data,   os.path.join(args.out_dir, "seen_vs_unseen.png"))
    plot_eer_vs_rejection(data, os.path.join(args.out_dir, "eer_vs_rejection.png"))
    plot_attack_scatter(data,   os.path.join(args.out_dir, "eer_bald_scatter.png"))
    plot_calibration_compact(data, os.path.join(args.out_dir, "calibration.png"))


if __name__ == "__main__":
    main()
