"""Training objective (Sec. 3.2) and evaluation protocol (Sec. 4.1) in one place.

Objective   L = L_sup + lambda(t) * L_unsup + beta * L_rob                    [Eq. 11-14, 20-22]
    L_sup   : cross-entropy on labelled nodes (binary head) + 0.5 * BCE (multi-label head)
    L_unsup : FixMatch-style (Sohn et al., 2020). For an unlabelled node, the prediction on the
              WEAK views (no gradient) gives p_w; if max p_w >= conf_threshold the arg-max is kept
              as a one-hot pseudo-label and cross-entropy is applied to the prediction on the
              STRONG views. Pseudo-labels are therefore recomputed at every iteration (online),
              only high-confidence ones are used, and lambda is ramped up linearly over
              `rampup_epochs`. The fraction of unlabelled nodes passing the filter is logged.
    L_rob   : ||p(x) - p(x + eps)||^2 with eps ~ N(0, sigma_train^2) added to the [0,1] image
              tensors of the weak views (training-time perturbation only; evaluation uses the
              corruption suite in data.corrupt with fixed, method-independent perturbations).

Graph training with a CNN front-end: embeddings of all N nodes are cached at the start of every
epoch (no gradient). For a mini-batch, the batch nodes are re-encoded WITH gradient and inserted
into the cached matrix, then full-graph propagation is run (historical-embedding scheme, cf.
GNNAutoScale, Fey et al. 2021). Message passing therefore always uses the full inductive
adjacency; only the staleness of non-batch embeddings is approximate.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score,
                             log_loss, precision_score, recall_score, roc_auc_score)
from torch.utils.data import DataLoader

from .data import IMAGENET_MEAN, IMAGENET_STD, S2DSNodes
from .metrics import binary_metrics, ece_score, multilabel_metrics, priority_metrics, selective_metrics

_MEAN = torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1)
_STD = torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1)


def add_gaussian_noise(x_norm: torch.Tensor, sigma: float) -> torch.Tensor:
    """Noise is defined on the [0,1] intensity scale, then re-normalised (Reviewer 4, comment 10)."""
    m, s = _MEAN.to(x_norm.device), _STD.to(x_norm.device)
    x = x_norm * s + m
    x = (x + sigma * torch.randn_like(x)).clamp(0, 1)
    return (x - m) / s


# =============================================================================================
# Training
# =============================================================================================
@torch.no_grad()
def encode_all(model, loader, device):
    model.eval()
    Z = []
    for b in loader:
        Z.append(model.encode(b["x"].to(device)).cpu())
    return torch.cat(Z)


def train_model(model, samples, A_hat, A_raw, labelled_mask, cfg, seed, device, log=print,
                use_ssl=True, use_rob=True, views_order="fixed"):
    """Returns the best model (selected on validation F1) and the training log."""
    torch.manual_seed(seed); np.random.seed(seed)
    tr = [i for i, s in enumerate(samples) if s.split == "train"]
    va = [i for i, s in enumerate(samples) if s.split == "val"]
    lab = [i for i in tr if labelled_mask[tr.index(i)]]
    unl = [i for i in tr if not labelled_mask[tr.index(i)]]
    T, size = cfg["views"]["T"], cfg["data"]["image_size"]
    y_bin = torch.tensor([s.y_bin for s in samples])
    y_ml = torch.tensor(np.stack([s.y_ml for s in samples])).float()
    full_ds = S2DSNodes(samples, T, size, "eval", seed, views_order)
    full_loader = DataLoader(full_ds, batch_size=32, num_workers=4)
    weak_ds = S2DSNodes(samples, T, size, "weak", seed, views_order)
    strong_ds = S2DSNodes(samples, T, size, "strong", seed + 1, views_order)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    B = cfg["train"]["batch_nodes"]
    best, best_f1, bad, history = None, -1, 0, []
    A_hat, A_raw = A_hat.to(device), (torch.from_numpy(A_raw).to(device) if A_raw is not None else None)
    for epoch in range(cfg["train"]["epochs"]):
        Zc = encode_all(model, full_loader, device).to(device)        # cached embeddings (no grad)
        model.train()
        rng = np.random.RandomState(seed * 1000 + epoch)
        lam = cfg["ssl"]["lambda_u"] * min(1.0, epoch / max(1, cfg["ssl"]["rampup_epochs"])) if use_ssl else 0.0
        beta = cfg["robust"]["beta"] if use_rob else 0.0
        order = rng.permutation(lab)
        n_mask, n_unl, tot = 0, 0, 0.0
        for bstart in range(0, len(order), B):
            bl = order[bstart:bstart + B].tolist()
            bu = rng.choice(unl, min(B, len(unl)), replace=False).tolist() if (use_ssl and unl) else []
            nodes = bl + bu
            xw = torch.stack([weak_ds[i]["x"] for i in nodes]).to(device)      # (n,T,3,H,W)
            zw = model.encode(xw)
            Z = Zc.clone(); Z[nodes] = zw                                        # insert with gradient
            logit_b, logit_ml = model.heads(model.propagate(Z, A_hat, A_raw))
            # ---- supervised (Eq. 11)
            loss = F.cross_entropy(logit_b[bl], y_bin[bl].to(device)) + 0.5 * F.binary_cross_entropy_with_logits(logit_ml[bl], y_ml[bl].to(device))
            # ---- FixMatch-style unsupervised (Eq. 12-13)
            if bu and lam > 0:
                with torch.no_grad():
                    pw = torch.softmax(logit_b[bu], -1)
                    conf, pseudo = pw.max(-1)
                    mask = (conf >= cfg["ssl"]["conf_threshold"]).float()
                xs = torch.stack([strong_ds[i]["x"] for i in bu]).to(device)
                Zs = Zc.clone(); Zs[bl] = zw[:len(bl)].detach(); Zs[bu] = model.encode(xs)
                logit_s, _ = model.heads(model.propagate(Zs, A_hat, A_raw))
                l_u = (F.cross_entropy(logit_s[bu], pseudo, reduction="none") * mask).mean()
                loss = loss + lam * l_u
                n_mask += int(mask.sum()); n_unl += len(bu)
            # ---- robustness (Eq. 20-21)
            if beta > 0:
                zn = model.encode(add_gaussian_noise(xw, cfg["robust"]["sigma_train"]))
                Zn = Zc.clone(); Zn[nodes] = zn
                logit_n, _ = model.heads(model.propagate(Zn, A_hat, A_raw))
                l_r = ((torch.softmax(logit_n[nodes], -1) - torch.softmax(logit_b[nodes], -1)) ** 2).sum(-1).mean()
                loss = loss + beta * l_r
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss)
        # ---- validation (model selection uses VAL only)
        res = evaluate(model, samples, A_hat, A_raw, va, cfg, seed, device, mc=False, bootstrap=0)
        f1 = res["f1"]
        history.append({"epoch": epoch, "loss": tot, "val_f1": f1, "val_acc": res["accuracy"], "pseudo_label_rate": n_mask / max(1, n_unl)})
        log(f"epoch {epoch:02d} loss {tot:.3f} val_f1 {f1:.3f} val_acc {res['accuracy']:.3f} pl_rate {n_mask / max(1, n_unl):.2f}")
        if f1 > best_f1:
            best_f1, best, bad = f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg["train"]["patience"]:
                break
    model.load_state_dict(best)
    return model, history


# =============================================================================================
# Evaluation
# =============================================================================================
@torch.no_grad()
def predict(model, samples, A_hat, A_raw, cfg, seed, device, mc=True, corruption=None, views_order="fixed", T=None):
    """Probabilities for ALL nodes. With mc=True returns (mu, sigma, p_ml) from M stochastic passes
    of dropout + GNN on the deterministic embeddings (backbone in eval mode)."""
    T = T or cfg["views"]["T"]
    ds = S2DSNodes(samples, T, cfg["data"]["image_size"], "corrupt" if corruption else "eval", seed, views_order, corruption)
    Z = encode_all(model, DataLoader(ds, batch_size=32, num_workers=4), device).to(device)
    M = cfg["uncertainty"]["mc_samples"] if mc else 1
    model.eval()
    if mc:
        for mod in model.modules():
            if isinstance(mod, torch.nn.Dropout):
                mod.train()
    P, PML = [], []
    for _ in range(M):
        lb, lml = model.heads(model.propagate(Z, A_hat, A_raw))
        P.append(torch.softmax(lb, -1)[:, 1].cpu().numpy()); PML.append(torch.sigmoid(lml).cpu().numpy())
    model.eval()
    P = np.stack(P)
    return P.mean(0), P.std(0), np.stack(PML).mean(0)


def evaluate(model, samples, A_hat, A_raw, idx, cfg, seed, device, mc=True, bootstrap=1000, corruption=None, views_order="fixed", T=None):
    mu, sigma, pml = predict(model, samples, A_hat, A_raw, cfg, seed, device, mc, corruption, views_order, T)
    idx = np.asarray(idx)
    y = np.array([samples[i].y_bin for i in idx]); Y = np.stack([samples[i].y_ml for i in idx])
    sev = np.array([samples[i].pixel_fractions[1:6].sum() for i in idx])
    p, s = mu[idx], sigma[idx]
    res = binary_metrics(p, y, cfg["uncertainty"]["tau"])
    res["ece"] = ece_score(p, y, cfg["eval"]["ece_bins"])
    res.update(multilabel_metrics(pml[idx], Y))
    if mc:
        ent = -(p * np.log(p + 1e-8) + (1 - p) * np.log(1 - p + 1e-8))
        res.update({f"sel_{k}": v for k, v in selective_metrics(p, y, ent).items()})
        res["mean_mc_std"] = float(s.mean())
        res.update(priority_metrics(p, s, y, sev))
    if bootstrap:
        rng = np.random.RandomState(seed)
        boots = {"accuracy": [], "f1": [], "auroc": []}
        for _ in range(bootstrap):
            b = rng.randint(0, len(y), len(y))
            if len(set(y[b])) < 2:
                continue
            m = binary_metrics(p[b], y[b], cfg["uncertainty"]["tau"])
            for k in boots:
                boots[k].append(m[k])
        for k, v in boots.items():
            res[f"{k}_ci95"] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    res["_p"], res["_y"] = p.tolist(), y.tolist()      # prediction records (Reviewer 4, comment 1)
    return res


def corruption_suite(model, samples, A_hat, A_raw, idx, cfg, seed, device):
    """Clean vs corrupted accuracy. Robustness score := mean corrupted accuracy / clean accuracy."""
    clean = evaluate(model, samples, A_hat, A_raw, idx, cfg, seed, device, mc=False, bootstrap=0)["accuracy"]
    out = {"clean_accuracy": clean}
    accs = []
    for kind, levels in cfg["eval"]["corruptions"].items():
        for lv in levels:
            a = evaluate(model, samples, A_hat, A_raw, idx, cfg, seed, device, mc=False, bootstrap=0, corruption=(kind, lv))["accuracy"]
            out[f"{kind}_{lv}"] = a; accs.append(a)
    out["mean_corrupted_accuracy"] = float(np.mean(accs))
    out["robustness_score"] = float(np.mean(accs) / max(clean, 1e-8))
    return out


def inference_cost(model, cfg, device, n=20):
    """Measured inference time per node with and without the M MC passes (Reviewer 4, comment 12)."""
    x = torch.randn(1, cfg["views"]["T"], 3, cfg["data"]["image_size"], cfg["data"]["image_size"]).to(device)
    model.eval()
    with torch.no_grad():
        model.encode(x); t0 = time.time()
        for _ in range(n):
            model.encode(x)
        t_enc = (time.time() - t0) / n
    return {"encode_ms_per_node": 1000 * t_enc, "mc_overhead_factor_gnn_only": cfg["uncertainty"]["mc_samples"],
            "params_M": sum(p.numel() for p in model.parameters()) / 1e6}
