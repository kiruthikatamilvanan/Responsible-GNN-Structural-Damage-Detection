"""Torch-free metric definitions (Sec. 3.3.1 and 4.1 of the manuscript).

Kept separate from train_eval.py so that results can be reproduced from prediction records
(scripts/report_from_records.py) without a GPU or PyTorch installation.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score,
                             log_loss, precision_score, recall_score, roc_auc_score)


def ece_score(p, y, n_bins=15):
    """Expected calibration error on the predicted class (Guo et al., 2017)."""
    conf = np.maximum(p, 1 - p); pred = (p >= 0.5).astype(int); acc = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1); e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def selective_metrics(p, y, unc):
    """Risk-coverage curve from the uncertainty score `unc` (higher = more uncertain)."""
    order = np.argsort(unc); err = ((p >= 0.5).astype(int) != y).astype(float)[order]
    cov = np.arange(1, len(err) + 1) / len(err); risk = np.cumsum(err) / np.arange(1, len(err) + 1)
    aurc = float(np.trapezoid(risk, cov))
    acc_at = {f"acc@cov{c}": float(1 - risk[int(c * len(err)) - 1]) for c in (0.8, 0.9)}
    correct = 1 - ((p >= 0.5).astype(int) != y).astype(int)
    err_auroc = float(roc_auc_score(1 - correct, unc)) if 0 < correct.mean() < 1 else float("nan")
    return {"aurc": aurc, "error_detection_auroc": err_auroc, **acc_at}


def binary_metrics(p, y, tau=0.5):
    pred = (p >= tau).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"accuracy": float((pred == y).mean()), "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)), "f1": float(f1_score(y, pred, zero_division=0)),
            "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
            "auroc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
            "auprc": float(average_precision_score(y, p)), "nll": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])),
            "brier": float(brier_score_loss(y, p)), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn), "support": int(len(y))}


def multilabel_metrics(P, Y):
    out = {}
    for c, name in enumerate(["crack", "spalling", "corrosion", "efflorescence", "vegetation"]):
        out[f"f1_{name}"] = float(f1_score(Y[:, c], (P[:, c] >= 0.5).astype(int), zero_division=0))
    out["f1_micro_ml"] = float(f1_score(Y, (P >= 0.5).astype(int), average="micro", zero_division=0))
    out["f1_macro_ml"] = float(f1_score(Y, (P >= 0.5).astype(int), average="macro", zero_division=0))
    out["mAP_ml"] = float(np.mean([average_precision_score(Y[:, c], P[:, c]) for c in range(Y.shape[1]) if Y[:, c].any()]))
    return out


def priority_metrics(mu, sigma, y, severity, k=(10, 20)):
    """Inspection-priority score P_i = mu_i (1 + sigma_i) (Sec. 3.3.3) vs. ranking by mu alone.
    severity = defect pixel fraction of the patch (proxy for extent)."""
    out = {}
    for name, score in (("priority", mu * (1 + sigma)), ("prob_only", mu)):
        order = np.argsort(-score)
        for kk in k:
            out[f"{name}_precision@{kk}"] = float(y[order[:kk]].mean())
        out[f"{name}_spearman_severity"] = float(stats.spearmanr(score, severity).correlation)
    return out


