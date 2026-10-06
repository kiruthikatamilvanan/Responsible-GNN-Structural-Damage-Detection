"""Torch-free smoke tests: graph rule (inductive constraint) and metric definitions."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rgnn_shm.metrics import binary_metrics, ece_score, priority_metrics, selective_metrics  # noqa: E402


def test_inductive_graph():
    src = open(os.path.join(os.path.dirname(__file__), "..", "rgnn_shm", "graph.py")).read()
    src = src.replace("import torch\n", "").split("def normalise")[0]
    ns = {}
    exec(src, ns)
    rng = np.random.RandomState(0)
    feats = rng.randn(60, 8)
    split = np.array(["train"] * 40 + ["val"] * 8 + ["test"] * 12)
    A = ns["build_adjacency"](feats, split, 4)
    te, va = np.where(split == "test")[0], np.where(split == "val")[0]
    assert A[np.ix_(te, te)].sum() == 0 and A[np.ix_(te, va)].sum() == 0 and A[np.ix_(va, va)].sum() == 0
    assert (A == A.T).all() and np.diag(A).sum() == 0


def test_metrics_match_manuscript_seed0():
    # seed-0 record of the paper: TP 59, FP 1, FN 6, TN 27 on 93 patches
    y = np.array([1] * 65 + [0] * 28)
    p = np.array([1] * 59 + [0] * 6 + [0] * 27 + [1] * 1).astype(float)
    m = binary_metrics(p, y)
    assert abs(m["accuracy"] * 100 - 92.4731) < 1e-3
    assert abs(m["precision"] * 100 - 98.3333) < 1e-3
    assert abs(m["recall"] * 100 - 90.7692) < 1e-3
    assert abs(m["f1"] * 100 - 94.4000) < 1e-3
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (59, 1, 6, 27)


def test_selective_and_priority_run():
    rng = np.random.RandomState(1)
    y = rng.randint(0, 2, 93)
    mu = np.clip(y * 0.6 + rng.rand(93) * 0.5, 0, 1)
    ent = -(mu * np.log(mu + 1e-8) + (1 - mu) * np.log(1 - mu + 1e-8))
    s = selective_metrics(mu, y, ent)
    assert 0 <= s["aurc"] <= 1 and 0 <= ece_score(mu, y) <= 1
    pr = priority_metrics(mu, rng.rand(93) * 0.1, y, rng.rand(93))
    assert 0 <= pr["priority_precision@10"] <= 1
