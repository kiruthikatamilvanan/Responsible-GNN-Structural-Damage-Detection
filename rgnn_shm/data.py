"""S2DS loading, label derivation and view construction.

Every design decision that reviewers asked about is made explicit here:

* Dataset version / subset ...... the public S2DS release (Benz & Rodehorst, 2022);
                                  743 patches, original 563/87/93 partition retained.
* Prediction target .............. PATCH level (one label vector per 1024x1024 patch).
      - binary  y_bin  = 1 if any defect class (1..5) covers >= min_defect_fraction of pixels
      - multi   y_ml   = 5-dim indicator over {crack, spalling, corrosion, efflorescence, vegetation}
      - class 6 (control point) is ignored; background alone => y_bin = 0 (negative / "no-defect")
* Views (T) ...................... S2DS contains NO repeated inspections. The T views fed to the
                                  "temporal" branch are deterministic crops/scales of the same
                                  patch (see build_views). We therefore call this branch
                                  "multi-view aggregation" and test a shuffled-order control.
"""
from __future__ import annotations

import csv
import io
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
CLASS_NAMES = {1: "crack", 2: "spalling", 3: "corrosion", 4: "efflorescence", 5: "vegetation", 6: "control_point"}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class Sample:
    image_path: str
    mask_path: str
    split: str
    source_id: str          # file stem; used to make sure no patch of one source leaks across splits
    y_bin: int
    y_ml: np.ndarray        # (5,)
    pixel_fractions: np.ndarray  # (7,) fraction of pixels per class id 0..6


# --------------------------------------------------------------------------------------
# Indexing
# --------------------------------------------------------------------------------------
def _find_mask(img: Path, root: Path) -> Path | None:
    """Locate the mask for an image. Adapt the candidate list to the local S2DS layout."""
    cands = [
        img.with_name(img.stem + "_lab" + img.suffix),
        img.with_name(img.stem + "_label.png"),
        img.parent.parent / "labels" / img.name,
        img.parent.parent / "masks" / img.name,
        Path(str(img).replace("/images/", "/labels/")),
        Path(str(img).replace("/img/", "/lab/")),
    ]
    for c in cands:
        if c.exists() and c != img:
            return c
    return None


def build_index(root: str, out_csv: str, min_defect_fraction: float, defect_classes: Sequence[int]) -> List[Sample]:
    """Walk the S2DS folder, derive patch-level labels from the pixel masks and write index.csv.

    The CSV is the reproducible split identifier requested by Reviewer 4 (comment 5)."""
    root = Path(root)
    samples: List[Sample] = []
    for split in ("train", "val", "test"):
        split_dir = root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Expected S2DS split folder {split_dir}; the original 563/87/93 partition must be kept.")
        for img in sorted(p for p in split_dir.rglob("*") if p.suffix.lower() in IMG_EXT and "lab" not in p.stem.lower()):
            mask = _find_mask(img, root)
            if mask is None:
                continue
            m = np.array(Image.open(mask))
            if m.ndim == 3:               # colour-coded masks: map by first channel (adapt if needed)
                m = m[..., 0]
            frac = np.array([(m == c).mean() for c in range(7)], dtype=np.float64)
            y_ml = (frac[list(defect_classes)] >= min_defect_fraction).astype(np.int64)
            samples.append(Sample(str(img), str(mask), split, img.stem, int(y_ml.any()), y_ml, frac))
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "mask", "split", "source_id", "y_bin"] + [f"y_{CLASS_NAMES[c]}" for c in defect_classes] + [f"frac_{c}" for c in range(7)])
        for s in samples:
            w.writerow([s.image_path, s.mask_path, s.split, s.source_id, s.y_bin] + s.y_ml.tolist() + [f"{v:.6f}" for v in s.pixel_fractions])
    # leakage check: a source id must occur in exactly one split
    seen = {}
    for s in samples:
        assert seen.setdefault(s.source_id, s.split) == s.split, f"source {s.source_id} appears in two splits"
    return samples


def load_index(csv_path: str) -> List[Sample]:
    out = []
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            y_ml = np.array([int(v) for k, v in row.items() if k.startswith("y_") and k != "y_bin"])
            frac = np.array([float(row[f"frac_{c}"]) for c in range(7)])
            out.append(Sample(row["image"], row["mask"], row["split"], row["source_id"], int(row["y_bin"]), y_ml, frac))
    return out


def summarise(samples: List[Sample]) -> str:
    """Counts reported in Table 1 of the revised manuscript (dataset statistics)."""
    lines = ["split,n,n_defect,n_no_defect," + ",".join(CLASS_NAMES[c] for c in range(1, 6))]
    for split in ("train", "val", "test"):
        ss = [s for s in samples if s.split == split]
        ml = np.stack([s.y_ml for s in ss]) if ss else np.zeros((0, 5))
        lines.append(f"{split},{len(ss)},{sum(s.y_bin for s in ss)},{sum(1 - s.y_bin for s in ss)}," + ",".join(str(int(v)) for v in ml.sum(0)))
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Labelled-subset selection (annotation budgets)
# --------------------------------------------------------------------------------------
def select_labelled(train_samples: List[Sample], fraction: float, seed: int) -> np.ndarray:
    """Stratified random labelled subset of the TRAIN split. Returns boolean mask (n_train,).
    Repeated with several seeds in run_all.py (Reviewer 4, comment 8)."""
    rng = np.random.RandomState(seed)
    y = np.array([s.y_bin for s in train_samples])
    mask = np.zeros(len(train_samples), dtype=bool)
    for c in (0, 1):
        idx = np.where(y == c)[0]
        n = max(1, int(round(fraction * len(idx))))
        mask[rng.choice(idx, n, replace=False)] = True
    return mask


# --------------------------------------------------------------------------------------
# Views and augmentation
# --------------------------------------------------------------------------------------
def _to_tensor(img: Image.Image) -> torch.Tensor:
    a = np.asarray(img, dtype=np.float32) / 255.0
    a = (a - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(a.transpose(2, 0, 1).copy())


def build_views(img: Image.Image, T: int, size: int, order: str = "fixed", rng: random.Random | None = None) -> torch.Tensor:
    """Deterministic multi-view construction. Returns (T, 3, size, size).

    View schedule (T=5): full patch, 4 scale/offset crops at 75 % (top-left, top-right, bottom-left,
    bottom-right). For T<5 the first T views are used; for T>5 additional 50 % centred crops at
    rotating offsets are appended. order='shuffled' permutes the views (temporal-order control)."""
    W, H = img.size
    schedule = [(0.0, 0.0, 1.0), (0.0, 0.0, 0.75), (0.25, 0.0, 0.75), (0.0, 0.25, 0.75), (0.25, 0.25, 0.75)]
    extra = [(0.25, 0.25, 0.5), (0.0, 0.0, 0.5), (0.5, 0.5, 0.5), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5)]
    sched = (schedule + extra)[:T]
    views = []
    for ox, oy, sc in sched:
        box = (int(ox * W), int(oy * H), int(ox * W + sc * W), int(oy * H + sc * H))
        views.append(_to_tensor(img.crop(box).resize((size, size), Image.BILINEAR)))
    if order == "shuffled":
        (rng or random).shuffle(views)
    return torch.stack(views)


def weak_aug(img: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    return img


def strong_aug(img: Image.Image, rng: random.Random) -> Image.Image:
    """Strong augmentation for the FixMatch-style consistency branch (Sec. 3.2.3)."""
    img = weak_aug(img, rng)
    img = img.rotate(rng.choice([0, 90, 180, 270]))
    if rng.random() < 0.7:
        from PIL import ImageEnhance
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.6, 1.4))
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.6, 1.4))
    if rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 1.5)))
    return img


# --------------------------------------------------------------------------------------
# Evaluation-time corruptions (identical for every method; Reviewer 4, comment 10)
# --------------------------------------------------------------------------------------
def corrupt(img: Image.Image, kind: str, level: float, rng: np.random.RandomState) -> Image.Image:
    a = np.asarray(img, dtype=np.float32) / 255.0
    if kind == "gaussian_noise":          # std on [0,1] scale, added before ImageNet normalisation
        a = a + rng.normal(0, level, a.shape)
    elif kind == "gaussian_blur":
        return img.filter(ImageFilter.GaussianBlur(level))
    elif kind == "jpeg":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(level))
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    elif kind == "brightness":
        a = a + level
    else:
        raise ValueError(kind)
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))


class S2DSNodes(Dataset):
    """One item = one node (patch) with its T views. mode in {'eval','weak','strong','corrupt'}."""

    def __init__(self, samples: List[Sample], T: int, size: int, mode: str = "eval", seed: int = 0,
                 order: str = "fixed", corruption: tuple | None = None, return_mask: bool = False):
        self.s, self.T, self.size, self.mode, self.order = samples, T, size, mode, order
        self.rng = random.Random(seed)
        self.nrng = np.random.RandomState(seed)
        self.corruption = corruption
        self.return_mask = return_mask

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        s = self.s[i]
        img = Image.open(s.image_path).convert("RGB")
        if self.mode == "weak":
            img = weak_aug(img, self.rng)
        elif self.mode == "strong":
            img = strong_aug(img, self.rng)
        elif self.mode == "corrupt" and self.corruption is not None:
            img = corrupt(img, self.corruption[0], self.corruption[1], self.nrng)
        x = build_views(img, self.T, self.size, self.order, self.rng)
        item = {"x": x, "y_bin": torch.tensor(s.y_bin), "y_ml": torch.from_numpy(s.y_ml).float(), "idx": torch.tensor(i)}
        if self.return_mask:
            m = np.array(Image.open(s.mask_path).resize((self.size, self.size), Image.NEAREST))
            if m.ndim == 3:
                m = m[..., 0]
            item["mask"] = torch.from_numpy(((m >= 1) & (m <= 5)).astype(np.float32))
        return item
