"""Run every experiment of the revised manuscript and write results/tables.md + JSON records.

Usage
    python run_all.py --config config.yaml --out results            (full protocol, GPU)
    python run_all.py --stage index                                  (only build index.csv + stats)
    python run_all.py --stage tune | main | ablation | budget | views | hparam | corrupt | explain

Protocol (Sec. 4.1)
    * splits ....... original S2DS 563/87/93; index.csv is the split identifier
    * tuning ....... random search (20 configs) on VAL for EVERY method, same budget; test untouched
    * repeats ...... 5 seeds (model init, labelled-subset draw, augmentation); mean +- std, 95% CI
    * tests ........ paired Wilcoxon signed-rank over seeds vs. the strongest baseline, plus a
                     paired bootstrap on the test nodes for seed 0 (same test set, same predictions)
    * precision .... all metrics reported to two decimals (percent) in the tables
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy import stats

from rgnn_shm.data import build_index, load_index, select_labelled, summarise
from rgnn_shm.graph import build_adjacency, frozen_features, graph_stats, normalise
from rgnn_shm.models import DamageNet
from rgnn_shm.train_eval import corruption_suite, evaluate, inference_cost, train_model

BASELINES = {
    "CNN (ResNet-18)": dict(variant="cnn"),
    "CNN+LSTM (5 views)": dict(variant="cnn_lstm"),
    "GCN": dict(variant="gcn", gnn="gcn"),
    "GAT": dict(variant="gat", gnn="gat"),
    "Proposed (supervised only)": dict(variant="full", ssl=False, rob=False),
    "Proposed": dict(variant="full"),
}
ABLATIONS = {
    "Proposed": dict(variant="full"),
    "- graph propagation": dict(variant="no_graph"),
    "- multi-view aggregation (T=1)": dict(variant="no_views"),
    "- hierarchical attention": dict(variant="no_attention"),
    "- semi-supervised term (lambda=0)": dict(variant="full", ssl=False),
    "- robustness term (beta=0)": dict(variant="full", rob=False),
    "GCN -> GAT propagation": dict(variant="full", gnn="gat"),
    "shuffled view order (control)": dict(variant="full", order="shuffled"),
}
TUNE_SPACE = {"lr": [3e-4, 1e-3, 3e-3], "hidden": [64, 128, 256], "dropout": [0.1, 0.3, 0.5], "gnn_layers": [1, 2, 3],
              "lambda_u": [0.1, 0.3, 0.5, 0.7, 1.0], "beta": [0.0, 0.1, 0.2, 0.4], "conf_threshold": [0.8, 0.9, 0.95]}


# ---------------------------------------------------------------------------------------------
def mean_std(v):
    v = np.asarray(v, float); n = len(v)
    ci = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    return v.mean(), v.std(ddof=1) if n > 1 else 0.0, ci


def paired_tests(a, b):
    """a, b: per-seed metric lists (same seeds). Returns Wilcoxon and paired t p-values."""
    a, b = np.asarray(a), np.asarray(b)
    out = {"wilcoxon_p": float(stats.wilcoxon(a, b).pvalue) if len(a) > 1 and np.any(a != b) else float("nan"),
           "ttest_p": float(stats.ttest_rel(a, b).pvalue) if len(a) > 1 else float("nan")}
    return out


def paired_bootstrap(p1, p2, y, n=1000, seed=0):
    """P(method1 F1 <= method2 F1) under paired resampling of the SAME test nodes."""
    from sklearn.metrics import f1_score
    rng = np.random.RandomState(seed); y = np.asarray(y); p1, p2 = np.asarray(p1), np.asarray(p2); worse = 0
    for _ in range(n):
        b = rng.randint(0, len(y), len(y))
        worse += f1_score(y[b], p1[b] >= 0.5) <= f1_score(y[b], p2[b] >= 0.5)
    return worse / n


def fmt(v, ci=None):
    m, s, c = mean_std(v)
    return f"{100 * m:.2f} ± {100 * s:.2f}" + (f" [{100 * (m - c):.2f}, {100 * (m + c):.2f}]" if ci else "")


# ---------------------------------------------------------------------------------------------
def build_model(cfg, spec, device):
    m = cfg["model"]
    return DamageNet(variant=spec.get("variant", "full"), gnn=spec.get("gnn", m["gnn"]), gnn_layers=spec.get("gnn_layers", m["gnn_layers"]),
                     hidden=spec.get("hidden", m["hidden"]), dropout=spec.get("dropout", m["dropout"]), proj_dim=m["proj_dim"],
                     temporal_kernel=m["temporal_kernel"], heads=m["heads"], pretrained=m["pretrained"] == "imagenet").to(device)


def run_one(cfg, samples, A_hat, A_raw, spec, seed, device, out_dir, tag, label_fraction=None, T=None):
    """Train + evaluate one method for one seed. Writes the prediction record to disk."""
    cfg = json.loads(json.dumps(cfg))
    for k in ("lr",):
        if k in spec: cfg["train"][k] = spec[k]
    for k in ("lambda_u", "conf_threshold"):
        if k in spec: cfg["ssl"][k] = spec[k]
    if "beta" in spec: cfg["robust"]["beta"] = spec["beta"]
    if T: cfg["views"]["T"] = T
    if spec.get("variant") in ("cnn", "gcn", "gat", "no_views"): cfg["views"]["T"] = 1
    tr = [s for s in samples if s.split == "train"]
    lab_mask = select_labelled(tr, label_fraction or cfg["ssl"]["label_fraction"], seed)
    model = build_model(cfg, spec, device)
    model, hist = train_model(model, samples, A_hat, A_raw, lab_mask, cfg, seed, device, use_ssl=spec.get("ssl", True),
                              use_rob=spec.get("rob", True), views_order=spec.get("order", "fixed"))
    test = [i for i, s in enumerate(samples) if s.split == "test"]
    res = evaluate(model, samples, A_hat, A_raw, test, cfg, seed, device, mc=True, bootstrap=cfg["eval"]["bootstrap"],
                   views_order=spec.get("order", "fixed"))
    res["history"] = hist; res["labelled_nodes"] = int(lab_mask.sum()); res["unlabelled_nodes"] = int((~lab_mask).sum())
    res["cost"] = inference_cost(model, cfg, device)
    Path(out_dir, "records").mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir, "records", f"{tag}_seed{seed}.json"), "w") as f:
        json.dump(res, f)
    torch.save(model.state_dict(), Path(out_dir, "records", f"{tag}_seed{seed}.pt"))
    return model, res


def table(rows, header, caption):
    s = [f"**{caption}**", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    s += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(s) + "\n"


# ---------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml"); ap.add_argument("--out", default="results")
    ap.add_argument("--stage", default="all"); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(); cfg = yaml.safe_load(open(a.config)); out = Path(a.out); out.mkdir(exist_ok=True); md = []
    dev = a.device

    # ---- data index and statistics (Table 1, Sec. 4.2)
    idx_csv = out / "index.csv"
    samples = build_index(cfg["data"]["root"], idx_csv, cfg["data"]["min_defect_fraction"], cfg["data"]["defect_classes"]) if not idx_csv.exists() else load_index(idx_csv)
    (out / "dataset_stats.csv").write_text(summarise(samples))
    split = np.array([s.split for s in samples])
    if a.stage == "index":
        print(summarise(samples)); return

    # ---- graph (Sec. 3.1.2); frozen features cached
    fpath = out / "frozen_feats.npy"
    feats = np.load(fpath) if fpath.exists() else frozen_features(samples, cfg["data"]["image_size"], dev); np.save(fpath, feats)
    A_raw = build_adjacency(feats, split, cfg["graph"]["k"]); A_hat = normalise(A_raw)
    gs = graph_stats(A_raw, split); json.dump(gs, open(out / "graph_stats.json", "w"), indent=1)
    md.append("**Graph statistics (Sec. 4.2)**\n\n" + "\n".join(f"- {k}: {v}" for k, v in gs.items()) + "\n")
    seeds = cfg["eval"]["seeds"]; test = [i for i, s in enumerate(samples) if s.split == "test"]

    # ---- hyper-parameter search on VAL for every method (same budget)  [stage: tune]
    tuned = {}
    tpath = out / "tuned.json"
    if tpath.exists():
        tuned = json.load(open(tpath))
    elif a.stage in ("all", "tune"):
        rng = np.random.RandomState(0)
        for name, spec in BASELINES.items():
            best, best_f1 = None, -1
            for _ in range(20):
                trial = {k: rng.choice(v).item() for k, v in TUNE_SPACE.items()}
                if not spec.get("ssl", True): trial["lambda_u"] = 0.0
                if not spec.get("rob", True): trial["beta"] = 0.0
                c = json.loads(json.dumps(cfg)); c["train"]["epochs"] = 20
                tr = [s for s in samples if s.split == "train"]; va = [i for i, s in enumerate(samples) if s.split == "val"]
                c["train"]["lr"] = trial["lr"]; c["ssl"]["lambda_u"] = trial["lambda_u"]; c["ssl"]["conf_threshold"] = trial["conf_threshold"]; c["robust"]["beta"] = trial["beta"]
                if spec.get("variant") in ("cnn", "gcn", "gat"): c["views"]["T"] = 1
                m = build_model(c, {**spec, **trial}, dev)
                m, _ = train_model(m, samples, A_hat, A_raw, select_labelled(tr, c["ssl"]["label_fraction"], 0), c, 0, dev, log=lambda *x: None,
                                   use_ssl=spec.get("ssl", True), use_rob=spec.get("rob", True))
                f1 = evaluate(m, samples, A_hat, A_raw, va, c, 0, dev, mc=False, bootstrap=0)["f1"]
                if f1 > best_f1: best, best_f1 = trial, f1
            tuned[name] = {**spec, **best, "val_f1": best_f1}
        json.dump(tuned, open(tpath, "w"), indent=1)
    md.append(table([[n, json.dumps({k: v for k, v in t.items() if k in TUNE_SPACE}), f"{100 * t['val_f1']:.2f}"] for n, t in tuned.items()],
                    ["Method", "Selected hyper-parameters (val)", "Val F1 (%)"], "Table T. Hyper-parameters selected on the validation split (20-trial random search per method)"))

    # ---- main comparison (Table 6)  [stage: main]
    if a.stage in ("all", "main"):
        R = {}
        for name in BASELINES:
            spec = tuned.get(name, BASELINES[name])
            R[name] = [run_one(cfg, samples, A_hat, A_raw, spec, s, dev, out, f"main_{name}", )[1] for s in seeds]
        strongest = max((n for n in R if not n.startswith("Proposed")), key=lambda n: np.mean([r["f1"] for r in R[n]]))
        rows = []
        for n, rs in R.items():
            pt = paired_tests([r["f1"] for r in R["Proposed"]], [r["f1"] for r in rs]) if n != "Proposed" else {"wilcoxon_p": float("nan")}
            pb = paired_bootstrap(R["Proposed"][0]["_p"], rs[0]["_p"], rs[0]["_y"]) if n != "Proposed" else float("nan")
            rows.append([n, fmt([r["accuracy"] for r in rs], True), fmt([r["precision"] for r in rs]), fmt([r["recall"] for r in rs]),
                         fmt([r["f1"] for r in rs], True), fmt([r["auroc"] for r in rs], True), fmt([r["f1_macro_ml"] for r in rs]),
                         f"{pt['wilcoxon_p']:.3f}", f"{pb:.3f}", f"{rs[0]['cost']['encode_ms_per_node']:.1f}"])
        md.append(table(rows, ["Method", "Accuracy (%) ± s [95% CI]", "Precision (%)", "Recall (%)", "F1 (%) ± s [95% CI]", "AUROC (%) ± s [95% CI]",
                               "Multi-label macro-F1 (%)", "Wilcoxon p vs Proposed", "Paired-bootstrap p (F1)", "ms / node"],
                        f"Table 6. Test-set comparison (5 seeds). Strongest baseline: {strongest}. Support n={len(test)}; confusion counts in records/*.json"))
        rows = [[n, fmt([r["ece"] for r in rs]), fmt([r["nll"] for r in rs]) if False else f"{np.mean([r['nll'] for r in rs]):.3f}", f"{np.mean([r['brier'] for r in rs]):.3f}",
                 fmt([r["sel_aurc"] for r in rs]), fmt([r["sel_error_detection_auroc"] for r in rs]), fmt([r["sel_acc@cov0.8"] for r in rs])] for n, rs in R.items()]
        md.append(table(rows, ["Method", "ECE (%)", "NLL", "Brier", "AURC (%)", "Error-detection AUROC (%)", "Accuracy @ 80% coverage (%)"],
                        "Table 8. Calibration and selective prediction on the test split (MC dropout, M=20)"))
        rows = [[n, fmt([r["priority_precision@10"] for r in rs]), fmt([r["prob_only_precision@10"] for r in rs]),
                 f"{np.mean([r['priority_spearman_severity'] for r in rs]):.3f}", f"{np.mean([r['prob_only_spearman_severity'] for r in rs]):.3f}"] for n, rs in R.items()]
        md.append(table(rows, ["Method", "Priority score: precision@10 (%)", "Probability only: precision@10 (%)", "Priority: Spearman vs defect extent", "Prob. only: Spearman vs defect extent"],
                        "Table 10. Inspection-priority ranking quality (heuristic score vs. probability-only ranking)"))

    # ---- ablations (Table 7)  [stage: ablation]
    if a.stage in ("all", "ablation"):
        base = tuned.get("Proposed", BASELINES["Proposed"]); R = {}
        for name, spec in ABLATIONS.items():
            R[name] = [run_one(cfg, samples, A_hat, A_raw, {**base, **spec}, s, dev, out, f"abl_{name}")[1] for s in seeds]
        rows = [[n, fmt([r["accuracy"] for r in rs]), fmt([r["f1"] for r in rs]), fmt([r["auroc"] for r in rs]),
                 f"{paired_tests([r['f1'] for r in R['Proposed']], [r['f1'] for r in rs])['wilcoxon_p']:.3f}" if n != "Proposed" else "—"] for n, rs in R.items()]
        md.append(table(rows, ["Configuration", "Accuracy (%)", "F1 (%)", "AUROC (%)", "Wilcoxon p vs full"], "Table 7. Controlled ablation (5 seeds, 25% labels)"))

    # ---- annotation budgets (Table 9 / Fig.)  [stage: budget]
    if a.stage in ("all", "budget"):
        base = tuned.get("Proposed", BASELINES["Proposed"]); rows = []
        for frac in (0.10, 0.25, 0.50, 1.00):
            ssl = [run_one(cfg, samples, A_hat, A_raw, base, s, dev, out, f"budget_ssl_{frac}", label_fraction=frac)[1] for s in seeds]
            sup = [run_one(cfg, samples, A_hat, A_raw, {**base, "ssl": False}, s, dev, out, f"budget_sup_{frac}", label_fraction=frac)[1] for s in seeds]
            rows.append([f"{int(frac * 100)}% ({ssl[0]['labelled_nodes']} labelled / {ssl[0]['unlabelled_nodes']} unlabelled)", fmt([r["f1"] for r in sup]), fmt([r["f1"] for r in ssl]),
                         f"{paired_tests([r['f1'] for r in ssl], [r['f1'] for r in sup])['wilcoxon_p']:.3f}", f"{np.mean([h['pseudo_label_rate'] for r in ssl for h in r['history'][-5:]]):.2f}"])
        md.append(table(rows, ["Labelled fraction of train", "Supervised-only F1 (%)", "Semi-supervised F1 (%)", "Wilcoxon p", "Pseudo-label acceptance rate"],
                        "Table 9. Annotation budget study (same architecture with / without the unlabelled objective; 5 labelled-subset draws)"))

    # ---- number of views T (Sec. 4.3)  [stage: views]
    if a.stage in ("all", "views"):
        base = tuned.get("Proposed", BASELINES["Proposed"]); rows = []
        for T in (1, 3, 5, 7):
            rs = [run_one(cfg, samples, A_hat, A_raw, base, s, dev, out, f"views_T{T}", T=T)[1] for s in seeds]
            rows.append([str(T), fmt([r["f1"] for r in rs]), fmt([r["auroc"] for r in rs]), f"{rs[0]['cost']['encode_ms_per_node']:.1f}"])
        md.append(table(rows, ["T (views)", "F1 (%)", "AUROC (%)", "ms / node"], "Table 5. Effect of the number of views T (fixed order)"))

    # ---- lambda / beta / threshold sensitivity (Tables 3-4)  [stage: hparam]
    if a.stage in ("all", "hparam"):
        base = tuned.get("Proposed", BASELINES["Proposed"]); rows = []
        for lam in (0.0, 0.1, 0.3, 0.5, 0.7, 1.0):
            rs = [run_one(cfg, samples, A_hat, A_raw, {**base, "lambda_u": lam}, s, dev, out, f"lam_{lam}")[1] for s in seeds]
            rows.append([str(lam), fmt([r["accuracy"] for r in rs]), fmt([r["f1"] for r in rs]), fmt([r["ece"] for r in rs]), fmt([r["mean_mc_std"] for r in rs])])
        md.append(table(rows, ["lambda", "Accuracy (%)", "F1 (%)", "ECE (%)", "Mean MC std (%)"], "Table 3. Sensitivity to the unlabelled-loss weight lambda (val-selected value in bold in the manuscript)"))
        rows = []
        for beta in (0.0, 0.1, 0.2, 0.4, 0.8):
            m, r0 = run_one(cfg, samples, A_hat, A_raw, {**base, "beta": beta}, seeds[0], dev, out, f"beta_{beta}")
            c = corruption_suite(m, samples, A_hat, A_raw, test, cfg, seeds[0], dev)
            rows.append([str(beta), f"{100 * c['clean_accuracy']:.2f}", f"{100 * c['mean_corrupted_accuracy']:.2f}", f"{100 * c['robustness_score']:.2f}"] + [f"{100 * c[k]:.2f}" for k in c if k not in ("clean_accuracy", "mean_corrupted_accuracy", "robustness_score")])
        md.append(table(rows, ["beta", "Clean acc. (%)", "Mean corrupted acc. (%)", "Robustness score (%)"] + [k for k in c if k not in ("clean_accuracy", "mean_corrupted_accuracy", "robustness_score")],
                        "Table 4. Robustness weight beta: clean vs corrupted accuracy (identical test perturbations; seed 0)"))

    # ---- corruption suite for every method (Table 11)  [stage: corrupt]
    if a.stage in ("all", "corrupt"):
        rows = []
        for name in BASELINES:
            spec = tuned.get(name, BASELINES[name]); cs = []
            for s in seeds:
                p = out / "records" / f"main_{name}_seed{s}.pt"
                m = build_model(cfg if spec.get("variant") not in ("cnn", "gcn", "gat") else {**cfg, "views": {**cfg["views"], "T": 1}}, spec, dev)
                m.load_state_dict(torch.load(p, map_location=dev)); cs.append(corruption_suite(m, samples, A_hat, A_raw, test, cfg, s, dev))
            rows.append([name, fmt([c["clean_accuracy"] for c in cs]), fmt([c["mean_corrupted_accuracy"] for c in cs]), fmt([c["robustness_score"] for c in cs])])
        md.append(table(rows, ["Method", "Clean accuracy (%)", "Mean corrupted accuracy (%)", "Robustness score (%)"], "Table 11. Robustness to evaluation-time corruptions (all methods, identical perturbations, 5 seeds)"))

    (out / "tables.md").write_text("\n\n".join(md))
    print("\n\n".join(md))


if __name__ == "__main__":
    main()
