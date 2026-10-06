"""Graph construction (Sec. 3.1.2 of the revised manuscript).

Connectivity rule (explicit, reproducible):
    1. Each patch is a node.
    2. A frozen ImageNet ResNet-18 produces a 512-d global-average-pooled descriptor f_i per patch
       (computed once, BEFORE any training, with no labels involved).
    3. Edge (i,j) exists iff j is among the k nearest neighbours of i under cosine similarity
       (or vice versa; the graph is symmetrised). k = 8 by default.
    4. Inductive protocol: neighbour search for TRAIN nodes is restricted to TRAIN nodes;
       VAL and TEST nodes attach to their k nearest TRAIN nodes only. No val-val, val-test or
       test-test edges exist, so no test feature ever participates in message passing for
       another test node, and test nodes never influence training.
    5. Optional spatial graph: if a metadata CSV with (source_image, row, col) exists for each
       patch, 4-neighbourhood grid edges are added between patches of the same source image.
       This is the only variant that reflects measured spatial adjacency; S2DS does not ship
       structural-component identifiers, so NO variant encodes physical structural connectivity.

Normalisation: A_hat = D~^{-1/2} (A + I) D~^{-1/2}  (Kipf & Welling, 2017).
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


def knn_edges(feats: np.ndarray, k: int, query_idx: np.ndarray, pool_idx: np.ndarray) -> list[tuple[int, int]]:
    """For every node in query_idx return edges to its k nearest nodes inside pool_idx (cosine)."""
    f = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(pool_idx)), metric="cosine").fit(f[pool_idx])
    _, ind = nn.kneighbors(f[query_idx])
    edges = []
    for qi, row in zip(query_idx, ind):
        for r in row:
            j = pool_idx[r]
            if j != qi:
                edges.append((int(qi), int(j)))
    return edges


def build_adjacency(feats: np.ndarray, split: np.ndarray, k: int, spatial: list[tuple[int, int]] | None = None) -> np.ndarray:
    """Dense symmetric 0/1 adjacency (N,N) following the inductive rule above."""
    N = len(feats)
    train = np.where(split == "train")[0]
    A = np.zeros((N, N), dtype=np.float32)
    edges = knn_edges(feats, k, train, train)
    for name in ("val", "test"):
        q = np.where(split == name)[0]
        if len(q):
            edges += knn_edges(feats, k, q, train)
    if spatial:
        edges += spatial
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    np.fill_diagonal(A, 0.0)
    return A


def normalise(A: np.ndarray) -> torch.Tensor:
    A_tilde = A + np.eye(len(A), dtype=np.float32)
    d = A_tilde.sum(1)
    D = np.diag(1.0 / np.sqrt(d))
    return torch.from_numpy(D @ A_tilde @ D)


def graph_stats(A: np.ndarray, split: np.ndarray) -> dict:
    """Numbers reported in Sec. 4.2 (graph-size distribution, Reviewer 4 comment 3)."""
    deg = A.sum(1)
    out = {"nodes": int(len(A)), "edges": int(A.sum() / 2), "mean_degree": float(deg.mean()),
           "min_degree": int(deg.min()), "max_degree": int(deg.max())}
    for s in ("train", "val", "test"):
        idx = split == s
        out[f"{s}_nodes"] = int(idx.sum())
        out[f"{s}_mean_degree"] = float(deg[idx].mean()) if idx.any() else 0.0
    # verify inductive constraint
    te = np.where(split == "test")[0]
    va = np.where(split == "val")[0]
    assert A[np.ix_(te, te)].sum() == 0 and A[np.ix_(va, va)].sum() == 0 and A[np.ix_(te, va)].sum() == 0, "non-inductive edges found"
    return out


@torch.no_grad()
def frozen_features(samples, size: int, device: str = "cuda") -> np.ndarray:
    """512-d frozen ImageNet ResNet-18 GAP descriptors used ONLY for graph construction."""
    import torchvision
    from PIL import Image
    from .data import _to_tensor

    net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity()
    net = net.to(device).eval()
    feats = []
    for s in samples:
        img = Image.open(s.image_path).convert("RGB").resize((size, size), Image.BILINEAR)
        feats.append(net(_to_tensor(img)[None].to(device)).cpu().numpy()[0])
    return np.stack(feats)
