"""
Probabilistic evaluation metrics — numpy only, no torch.

This module is shared by:
  - evaluate.py  (runs the models, then calls these on numpy arrays)
  - analyze.py   (recomputes everything from per-sample CSVs on any machine)

Conventions (same as the rest of the project):
  scores  : prob(real) in [0,1]  — higher = more likely bonafide
  labels  : 1 = real (bonafide), 0 = fake (spoof)   [positive class = real]
"""

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score


# ---------------------------------------------------------------------------
# EER
# ---------------------------------------------------------------------------

def compute_eer(scores: np.ndarray, labels: np.ndarray):
    """Equal Error Rate. Returns (eer, threshold)."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2), float(thresholds[idx])


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def compute_ece(scores: np.ndarray, labels: np.ndarray, n_bins: int = 15):
    """
    Expected Calibration Error (confidence-based, equal-width bins).

    For each sample the model's prediction is real if score >= 0.5,
    and its confidence is max(score, 1 - score).
    ECE = weighted average of |accuracy - confidence| over bins.

    Returns:
        ece       : float
        bin_stats : dict with per-bin arrays for the reliability diagram
                    (centers, confidence, accuracy, count)
    """
    preds      = (scores >= 0.5).astype(int)         # 1 = predicted real
    confidence = np.maximum(scores, 1.0 - scores)    # in [0.5, 1]
    correct    = (preds == labels).astype(float)

    edges = np.linspace(0.5, 1.0, n_bins + 1)
    ece   = 0.0
    n     = len(scores)

    centers, bin_conf, bin_acc, bin_count = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lo) & (confidence < hi if hi < 1.0 else confidence <= hi)
        c    = int(mask.sum())
        centers.append((lo + hi) / 2)
        bin_count.append(c)
        if c == 0:
            bin_conf.append(np.nan)
            bin_acc.append(np.nan)
            continue
        avg_conf = float(confidence[mask].mean())
        avg_acc  = float(correct[mask].mean())
        bin_conf.append(avg_conf)
        bin_acc.append(avg_acc)
        ece += (c / n) * abs(avg_acc - avg_conf)

    bin_stats = {
        "centers":    np.array(centers),
        "confidence": np.array(bin_conf),
        "accuracy":   np.array(bin_acc),
        "count":      np.array(bin_count),
    }
    return float(ece), bin_stats


def compute_nll(scores: np.ndarray, labels: np.ndarray, eps: float = 1e-12):
    """Mean negative log-likelihood of the true class (binary)."""
    p_true = np.where(labels == 1, scores, 1.0 - scores)
    return float(-np.log(np.clip(p_true, eps, 1.0)).mean())


def compute_brier(scores: np.ndarray, labels: np.ndarray):
    """Brier score: mean squared error between prob(real) and the label."""
    return float(np.mean((scores - labels) ** 2))


# ---------------------------------------------------------------------------
# Uncertainty diagnostics (BALD / predictive entropy)
# ---------------------------------------------------------------------------

def uncertainty_error_auroc(uncertainty: np.ndarray,
                            scores: np.ndarray,
                            labels: np.ndarray):
    """
    How well does the uncertainty rank errors?
    AUROC of `uncertainty` as a detector of misclassified samples.
    0.5 = useless, 1.0 = uncertainty perfectly separates right from wrong.
    """
    preds = (scores >= 0.5).astype(int)
    wrong = (preds != labels).astype(int)
    if wrong.sum() == 0 or wrong.sum() == len(wrong):
        return float("nan")
    return float(roc_auc_score(wrong, uncertainty))


def risk_coverage_curve(uncertainty: np.ndarray,
                        scores: np.ndarray,
                        labels: np.ndarray,
                        n_points: int = 100):
    """
    Selective prediction: reject the most-uncertain fraction of samples
    and measure the error rate on the rest.

    Returns:
        coverage : [n_points]  fraction of samples kept (ascending)
        risk     : [n_points]  error rate among kept samples
        aurc     : float       area under the risk-coverage curve (lower = better)
    """
    preds = (scores >= 0.5).astype(int)
    wrong = (preds != labels).astype(float)

    order  = np.argsort(uncertainty)      # most certain first
    wrong  = wrong[order]
    n      = len(wrong)

    cum_err  = np.cumsum(wrong)
    cov_full = np.arange(1, n + 1) / n
    risk_full = cum_err / np.arange(1, n + 1)

    idx      = np.linspace(0, n - 1, n_points).astype(int)
    coverage = cov_full[idx]
    risk     = risk_full[idx]
    aurc     = float(np.trapz(risk_full, cov_full))
    return coverage, risk, aurc


def eer_at_coverage(uncertainty: np.ndarray,
                    scores: np.ndarray,
                    labels: np.ndarray,
                    coverage: float = 0.9):
    """EER computed on the `coverage` fraction of least-uncertain samples."""
    n_keep = int(len(scores) * coverage)
    order  = np.argsort(uncertainty)[:n_keep]
    s, l   = scores[order], labels[order]
    if l.min() == l.max():          # only one class left — EER undefined
        return float("nan")
    eer, _ = compute_eer(s, l)
    return eer


# ---------------------------------------------------------------------------
# Full summary
# ---------------------------------------------------------------------------

def summarize(scores: np.ndarray,
              labels: np.ndarray,
              uncertainty: np.ndarray = None,
              n_bins: int = 15) -> dict:
    """Compute every scalar metric at once. Returns a flat dict."""
    eer, thr  = compute_eer(scores, labels)
    ece, _    = compute_ece(scores, labels, n_bins=n_bins)
    preds     = (scores >= 0.5).astype(int)

    # balanced accuracy: mean of per-class recalls — plain accuracy is
    # inflated by the ~1:9 spoof/bonafide imbalance of the eval protocol
    tpr = float((preds[labels == 1] == 1).mean())   # bonafide recall
    tnr = float((preds[labels == 0] == 0).mean())   # spoof recall

    out = {
        "eer":          eer,
        "threshold":    thr,
        "accuracy":     float((preds == labels).mean()),
        "balanced_acc": (tpr + tnr) / 2,
        "ece":          ece,
        "nll":          compute_nll(scores, labels),
        "brier":        compute_brier(scores, labels),
    }

    if uncertainty is not None:
        _, _, aurc = risk_coverage_curve(uncertainty, scores, labels)
        out.update({
            "mean_bald":        float(uncertainty.mean()),
            "bald_error_auroc": uncertainty_error_auroc(uncertainty, scores, labels),
            "aurc":             aurc,
            "eer_at_90cov":     eer_at_coverage(uncertainty, scores, labels, 0.90),
        })
    return out
