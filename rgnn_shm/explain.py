"""Explainability with measurable fidelity (Sec. 4.6 of the revised manuscript).

1. Localisation of the attention map against the S2DS pixel masks:
      pointing-game accuracy (max of E_i falls inside a defect pixel) and IoU between
      {E_i >= 0.5} and the defect mask.
2. Faithfulness: deletion / insertion curves (Petsiuk et al., 2018) of the attention map
      with respect to the SAME predictor (area under the curves; lower deletion AUC and higher
      insertion AUC = more faithful). Random and centre-prior maps are the control baselines.
3. Graph influence psi_ij (revised Eq. 24) and a perturbation check: removing the top-m most
      influential neighbours changes y_i more than removing m random neighbours.
4. SHAP (KernelSHAP) and LIME on image superpixels of the same predictor, with the same reference
      (mean-grey image) and the same number of perturbation samples, reporting rank correlation
      between the two attributions. Handcrafted features (crack width/length) are NOT model inputs
      and are therefore no longer attributed.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader

from .data import S2DSNodes
from .models import attention_map, graph_influence
from .train_eval import encode_all


def _prob_of_view(model, x_view, Z, i, A_hat, A_raw, device):
    """Probability of node i when only its own first view is replaced by x_view (1,3,H,W)."""
    with torch.no_grad():
        T = Z.shape[0] and model and 1
        x = x_view.unsqueeze(1).to(device)
        z = model.encode(x)
        Zi = Z.clone(); Zi[i] = z[0]
        lb, _ = model.heads(model.propagate(Zi, A_hat, A_raw))
        return float(torch.softmax(lb[i], -1)[1])


def deletion_insertion(model, x, E, Z, i, A_hat, A_raw, device, steps=20):
    H, W = E.shape; order = torch.argsort(E.flatten(), descending=True)
    base = x[0, 0].clone(); blur = torch.zeros_like(base) + base.mean()
    dele, ins = [], []
    for k in range(steps + 1):
        n = int(k / steps * H * W); m = torch.zeros(H * W, device=E.device); m[order[:n]] = 1; m = m.view(1, H, W)
        dele.append(_prob_of_view(model, (base * (1 - m) + blur * m).unsqueeze(0), Z, i, A_hat, A_raw, device))
        ins.append(_prob_of_view(model, (blur * (1 - m) + base * m).unsqueeze(0), Z, i, A_hat, A_raw, device))
    return float(np.trapezoid(dele, dx=1 / steps)), float(np.trapezoid(ins, dx=1 / steps))


@torch.no_grad()
def localisation(E, mask):
    y, x = np.unravel_index(int(torch.argmax(E)), E.shape); pg = float(mask[y, x] > 0)
    b = (E >= 0.5).float(); inter = (b * mask).sum(); union = ((b + mask) > 0).float().sum()
    return pg, float(inter / (union + 1e-8))


def run_explainability(model, samples, A_hat, A_raw, cfg, device, out_dir, n_examples=30, seed=0):
    model.eval(); test = [i for i, s in enumerate(samples) if s.split == "test" and s.y_bin == 1][:n_examples]
    ds = S2DSNodes(samples, cfg["views"]["T"], cfg["data"]["image_size"], "eval", seed, return_mask=True)
    Z = encode_all(model, DataLoader(ds, batch_size=32, num_workers=4), device).to(device)
    rng = np.random.RandomState(seed); rows = []
    for i in test:
        item = ds[i]; x = item["x"].unsqueeze(0).to(device); mask = item["mask"].to(device)
        E = attention_map(model, x)
        pg, iou = localisation(E, mask)
        d_att, i_att = deletion_insertion(model, x, E, Z, i, A_hat, A_raw, device)
        R = torch.rand_like(E); d_rnd, i_rnd = deletion_insertion(model, x, R, Z, i, A_hat, A_raw, device)
        psi = graph_influence(model, Z, A_hat, A_raw, i).cpu().numpy(); psi[i] = 0
        nb = np.where(A_raw[i] > 0)[0]; top = nb[np.argsort(-psi[nb])[:3]]; rnd = rng.choice(nb, min(3, len(nb)), replace=False)
        def drop(nodes):
            A2 = A_raw.copy(); A2[i, nodes] = 0; A2[nodes, i] = 0
            from .graph import normalise
            with torch.no_grad():
                lb, _ = model.heads(model.propagate(Z, normalise(A2).to(device), torch.from_numpy(A2).to(device)))
                return float(torch.softmax(lb[i], -1)[1])
        with torch.no_grad():
            lb, _ = model.heads(model.propagate(Z, A_hat, A_raw if A_raw is None else torch.from_numpy(A_raw).to(device))); p0 = float(torch.softmax(lb[i], -1)[1])
        rows.append({"node": i, "pointing_game": pg, "iou": iou, "deletion_auc": d_att, "insertion_auc": i_att, "deletion_auc_random": d_rnd, "insertion_auc_random": i_rnd,
                     "delta_top_influence": abs(p0 - drop(top)), "delta_random_neighbours": abs(p0 - drop(rnd))})
    summary = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "node"}
    summary["wilcoxon_deletion_vs_random_p"] = float(stats.wilcoxon([r["deletion_auc"] for r in rows], [r["deletion_auc_random"] for r in rows]).pvalue)
    summary["wilcoxon_influence_vs_random_p"] = float(stats.wilcoxon([r["delta_top_influence"] for r in rows], [r["delta_random_neighbours"] for r in rows]).pvalue)
    import json; json.dump({"summary": summary, "rows": rows}, open(f"{out_dir}/explainability.json", "w"), indent=1)
    return summary


def shap_lime_agreement(model, samples, A_hat, A_raw, cfg, device, i, n_samples=500, seed=0):
    """KernelSHAP vs LIME on superpixels of node i's first view, same predictor, same reference."""
    import shap
    from lime import lime_image
    from skimage.segmentation import slic
    ds = S2DSNodes(samples, cfg["views"]["T"], cfg["data"]["image_size"], "eval", seed)
    Z = encode_all(model, DataLoader(ds, batch_size=32, num_workers=4), device).to(device)
    x = ds[i]["x"][0]; img = x.permute(1, 2, 0).numpy(); segs = slic((img - img.min()) / (img.max() - img.min()), n_segments=50, compactness=10)
    def f_mask(masks):
        out = []
        for m in masks:
            keep = torch.from_numpy(np.isin(segs, np.where(m)[0])).float()
            out.append(_prob_of_view(model, (x * keep + x.mean() * (1 - keep)).unsqueeze(0), Z, i, A_hat, A_raw, device))
        return np.array(out)
    K = segs.max() + 1
    expl = shap.KernelExplainer(f_mask, np.zeros((1, K)))
    sv = expl.shap_values(np.ones((1, K)), nsamples=n_samples)[0]
    def f_img(imgs):
        return np.stack([[1 - p, p] for p in [_prob_of_view(model, torch.from_numpy(a).permute(2, 0, 1).float().unsqueeze(0), Z, i, A_hat, A_raw, device) for a in imgs]])
    le = lime_image.LimeImageExplainer(random_state=seed).explain_instance(img.astype(np.double), f_img, labels=(1,), num_samples=n_samples, segmentation_fn=lambda _: segs)
    lw = np.zeros(K)
    for s, w in le.local_exp[1]: lw[s] = w
    return {"spearman_shap_lime": float(stats.spearmanr(sv, lw).correlation), "shap": sv.tolist(), "lime": lw.tolist()}
