"""Reproduce the measured test-split numbers of the manuscript from per-seed prediction records.

Each record is a JSON file (results/records/seed_<k>.json) with at least
    y_true : list[int]   ground-truth binary label per test patch (1 = damaged)
    y_pred : list[int]   hard prediction per test patch
and, for the calibration / selective-prediction quantities (Table 9, Figure 7), also
    mu     : list[float] MC-dropout mean probability of the damaged class per test patch
    sigma  : list[float] MC-dropout standard deviation (optional)

Outputs (written to --out):
    table6_proposed.md      mean +- s.d. [95 % CI] of accuracy / precision / recall / F1 over seeds,
                            per-seed rows, per-seed and mean confusion counts
    confusion_matrix.png    confusion matrix of --cm-seed (default seed 0)
    calibration.md + reliability.png + risk_coverage.png   ONLY if `mu` is present in the records.
                            Hard 0/1 predictions are refused for these quantities: with hard labels
                            every confidence is 1.0 and ECE / AURC / risk-coverage are meaningless.

Usage
    python scripts/report_from_records.py --records results/records --out results
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from rgnn_shm.metrics import ece_score, selective_metrics


def mean_sd_ci(v):
    v = np.asarray(v, float)
    n = len(v)
    sd = v.std(ddof=1) if n > 1 else 0.0
    ci = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else 0.0
    return v.mean(), sd, ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="results/records")
    ap.add_argument("--out", default="results")
    ap.add_argument("--cm-seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.records, "seed_*.json")))
    if not files:
        raise SystemExit(f"no seed_*.json records found in {a.records}")
    rows, cms, probs = [], [], []
    for f in files:
        r = json.load(open(f))
        y, p = np.asarray(r["y_true"], int), np.asarray(r["y_pred"], int)
        tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
        rows.append({"seed": os.path.basename(f), "n": len(y), "acc": 100 * (y == p).mean(),
                     "prec": 100 * precision_score(y, p, zero_division=0), "rec": 100 * recall_score(y, p, zero_division=0),
                     "f1": 100 * f1_score(y, p, zero_division=0), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)})
        cms.append((tp, fp, fn, tn))
        if "mu" in r:
            probs.append((y, np.asarray(r["mu"], float)))
    n_test = rows[0]["n"]; pos = int(np.sum(np.asarray(json.load(open(files[0]))["y_true"]) == 1))
    lines = [f"# Test-split results reproduced from {len(rows)} seed records (n = {n_test}; {pos} damaged / {n_test - pos} no-defect)", "",
             "| Metric | Mean ± s.d. (%) | 95 % CI |", "|---|---|---|"]
    for key, name in (("acc", "Accuracy"), ("prec", "Precision"), ("rec", "Recall"), ("f1", "F1")):
        m, sd, ci = mean_sd_ci([r[key] for r in rows])
        lines.append(f"| {name} | {m:.2f} ± {sd:.2f} | [{m - ci:.2f}, {m + ci:.2f}] |")
    lines += ["", "| Seed | Accuracy | Precision | Recall | F1 | TP | FP | FN | TN |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['seed']} | {r['acc']:.2f} | {r['prec']:.2f} | {r['rec']:.2f} | {r['f1']:.2f} | {r['tp']} | {r['fp']} | {r['fn']} | {r['tn']} |")
    mc = np.mean(np.array(cms), 0)
    lines += ["", f"Mean confusion counts over seeds: TP = {mc[0]:.1f}, FP = {mc[1]:.1f}, FN = {mc[2]:.1f}, TN = {mc[3]:.1f}", "",
              "Sanity check: every per-seed accuracy must be an integer multiple of 100/n_test "
              f"({100 / n_test:.4f} %) — " + ("OK" if all(abs(r["acc"] * n_test / 100 - round(r["acc"] * n_test / 100)) < 1e-6 for r in rows) else "FAILED")]
    open(os.path.join(a.out, "table6_proposed.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---- confusion-matrix figure (Figure 5 of the manuscript)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tp, fp, fn, tn = cms[min(a.cm_seed, len(cms) - 1)]
    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=200)
    M = np.array([[tn, fp], [fn, tp]])
    ax.imshow(M, cmap="viridis")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(M[i, j]), ha="center", va="center", fontsize=16, color="black" if M[i, j] > M.max() / 2 else "white")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xticklabels(["No-defect", "Damaged"]); ax.set_yticklabels(["No-defect", "Damaged"])
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label"); ax.set_title(f"Confusion matrix (seed {a.cm_seed})")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "confusion_matrix.png")); plt.close(fig)

    # ---- calibration / selective prediction: only from probabilities
    if not probs:
        print("\nNo `mu` field in the records: ECE, AURC, reliability diagram, risk-coverage curve and AUROC are NOT computed. "
              "Save the MC-dropout mean probability per node (train_eval.evaluate stores it as `_p`).")
        return
    cal = ["# Calibration and selective prediction (from MC-dropout mean probabilities)", "", "| Seed | AUROC (%) | ECE (%) | AURC (%) | Acc @ 80 % cov. (%) | Acc @ 90 % cov. (%) | Error-detection AUROC (%) |", "|---|---|---|---|---|---|---|"]
    for k, (y, mu) in enumerate(probs):
        ent = -(mu * np.log(mu + 1e-8) + (1 - mu) * np.log(1 - mu + 1e-8))
        s = selective_metrics(mu, y, ent)
        cal.append(f"| {k} | {100 * roc_auc_score(y, mu):.2f} | {100 * ece_score(mu, y, 15):.2f} | {100 * s['aurc']:.2f} | {100 * s['acc@cov0.8']:.2f} | {100 * s['acc@cov0.9']:.2f} | {100 * s['error_detection_auroc']:.2f} |")
    open(os.path.join(a.out, "calibration.md"), "w").write("\n".join(cal) + "\n"); print("\n".join(cal))
    y, mu = probs[0]
    bins = np.linspace(0, 1, 16); conf = np.maximum(mu, 1 - mu); pred = (mu >= 0.5).astype(int); acc = (pred == y)
    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=200)
    xs, ys = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            xs.append(conf[m].mean()); ys.append(acc[m].mean())
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration"); ax.plot(xs, ys, "o-", label="Model")
    ax.set_xlabel("Mean predicted confidence"); ax.set_ylabel("Observed accuracy"); ax.set_title("Reliability diagram (15 bins, seed 0)"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "reliability.png")); plt.close(fig)
    ent = -(mu * np.log(mu + 1e-8) + (1 - mu) * np.log(1 - mu + 1e-8)); order = np.argsort(ent)
    err = (pred != y).astype(float)[order]; cov = np.arange(1, len(err) + 1) / len(err); risk = np.cumsum(err) / np.arange(1, len(err) + 1)
    fig, ax = plt.subplots(figsize=(5.2, 3.8), dpi=200); ax.plot(100 * cov, risk); ax.set_xlabel("Coverage (%)"); ax.set_ylabel("Risk (error rate)")
    ax.set_title("Risk–coverage curve (entropy rejection, seed 0)"); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(os.path.join(a.out, "risk_coverage.png")); plt.close(fig)


if __name__ == "__main__":
    main()
