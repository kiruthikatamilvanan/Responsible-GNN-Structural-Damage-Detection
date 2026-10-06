"""Architectures (Sec. 3.1-3.2 of the revised manuscript) with explicit tensor dimensions.

Proposed model  (variant='full')
    x        : (B, T, 3, 224, 224)
    backbone : ResNet-18 (ImageNet init). Multi-scale maps from layer2 (128ch, 28x28),
               layer3 (256ch, 14x14), layer4 (512ch, 7x7). Each is projected by a 1x1 conv to
               d=256 and global-average-pooled  ->  per view, per scale: (B, T, S=3, 256)
    view agg : 1-D temporal convolution over the T axis (kernel 3, 256->256, padding 1) + mean
               over T  ->  (B, S, 256)                                       [Eq. 5]
    attention: query q = mean over scales (B,256); keys/values = per-scale vectors; scaled dot
               product -> alpha (B,S); z = sum_s alpha_s v_s -> (B,256)        [Eq. 6-7]
    GNN      : H0 = z ; L layers of GCN (or GAT) with hidden 128                 [Eq. 8-10]
    heads    : binary softmax (2) and multi-label sigmoid (5)                   [Eq. 15]

Baselines (same backbone, init, input resolution, augmentation and tuning budget):
    'cnn'      : single view, layer4 GAP -> MLP.                     (no views, no graph)
    'cnn_lstm' : T views, layer4 GAP -> LSTM(256) -> MLP.            (views, no graph)
    'gcn'      : single view, layer4 GAP -> GCN.                     (graph, no views/attention)
    'gat'      : single view, layer4 GAP -> GAT.                     (graph, no views/attention)
Ablations of the proposed model: no_graph, no_views (T=1), no_attention (mean over scales),
    gnn='gat' swap; the SSL / robustness terms are switched off in train_eval.py, not here.

Dense adjacency is used (N=743), so no torch_geometric dependency is needed.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class MultiScaleBackbone(nn.Module):
    def __init__(self, proj_dim=256, pretrained=True, scales=("layer2", "layer3", "layer4")):
        super().__init__()
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        r = torchvision.models.resnet18(weights=w)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1)
        self.layer2, self.layer3, self.layer4 = r.layer2, r.layer3, r.layer4
        self.scales = scales
        ch = {"layer2": 128, "layer3": 256, "layer4": 512}
        self.proj = nn.ModuleDict({s: nn.Conv2d(ch[s], proj_dim, 1) for s in scales})

    def forward(self, x):  # x: (N,3,H,W) -> list of (N,d,h,w) feature maps, one per scale
        x = self.stem(x)
        maps = {"layer2": self.layer2(x)}
        maps["layer3"] = self.layer3(maps["layer2"])
        maps["layer4"] = self.layer4(maps["layer3"])
        return [self.proj[s](maps[s]) for s in self.scales]


class HierarchicalAttention(nn.Module):
    """Scale attention, Eq. (6)-(7). Returns attended vector and the weights (used in Eq. 23)."""

    def __init__(self, d):
        super().__init__()
        self.q, self.k, self.v = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.scale = math.sqrt(d)

    def forward(self, Z):  # Z: (B,S,d)
        q = self.q(Z.mean(1))                      # (B,d)
        K, V = self.k(Z), self.v(Z)                # (B,S,d)
        alpha = torch.softmax((K @ q.unsqueeze(-1)).squeeze(-1) / self.scale, dim=1)  # (B,S)
        return (alpha.unsqueeze(-1) * V).sum(1), alpha


class GCNLayer(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.W = nn.Linear(i, o, bias=False)
        self.b = nn.Parameter(torch.zeros(o))

    def forward(self, H, A_hat):  # Eq. (9): H' = sigma(A_hat H W)
        return A_hat @ self.W(H) + self.b


class GATLayer(nn.Module):
    def __init__(self, i, o, heads=4, dropout=0.3):
        super().__init__()
        self.heads, self.o = heads, o
        self.W = nn.Linear(i, o * heads, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, o)); self.a_dst = nn.Parameter(torch.empty(heads, o))
        nn.init.xavier_uniform_(self.a_src); nn.init.xavier_uniform_(self.a_dst)
        self.drop = nn.Dropout(dropout)

    def forward(self, H, A):  # A: 0/1 adjacency with self loops added here
        N = H.shape[0]
        h = self.W(H).view(N, self.heads, self.o)                       # (N,h,o)
        e = (h * self.a_src).sum(-1).unsqueeze(1) + (h * self.a_dst).sum(-1).unsqueeze(0)  # (N,N,h)
        e = F.leaky_relu(e, 0.2)
        mask = (A + torch.eye(N, device=A.device)) > 0
        e = e.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        att = self.drop(torch.softmax(e, dim=1))                        # (N,N,h)
        out = torch.einsum("ijh,jho->iho", att, h)                      # (N,h,o)
        return out.mean(1)


class DamageNet(nn.Module):
    """Proposed model and all baselines/ablations, selected by `variant`."""

    def __init__(self, variant="full", gnn="gcn", gnn_layers=2, hidden=128, dropout=0.3, proj_dim=256,
                 temporal_kernel=3, heads=4, pretrained=True, n_ml=5):
        super().__init__()
        self.variant, self.gnn_type = variant, gnn
        self.use_views = variant in ("full", "cnn_lstm", "no_graph", "no_attention")
        self.use_graph = variant in ("full", "gcn", "gat", "no_views", "no_attention")
        self.use_attention = variant in ("full", "no_graph", "no_views")
        self.use_multiscale = variant in ("full", "no_graph", "no_views", "no_attention")
        scales = ("layer2", "layer3", "layer4") if self.use_multiscale else ("layer4",)
        self.backbone = MultiScaleBackbone(proj_dim, pretrained, scales)
        d = proj_dim
        if variant == "cnn_lstm":
            self.lstm = nn.LSTM(d, d, batch_first=True)
        elif self.use_views:
            self.tconv = nn.Conv1d(d, d, temporal_kernel, padding=temporal_kernel // 2)   # Eq. (5)
        self.att = HierarchicalAttention(d) if self.use_attention else None
        self.drop = nn.Dropout(dropout)
        if self.use_graph:
            dims = [d] + [hidden] * gnn_layers
            if gnn == "gat":
                self.gnn = nn.ModuleList([GATLayer(dims[i], dims[i + 1], heads, dropout) for i in range(gnn_layers)])
            else:
                self.gnn = nn.ModuleList([GCNLayer(dims[i], dims[i + 1]) for i in range(gnn_layers)])
            out_dim = hidden
        else:
            self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout))
            out_dim = hidden
        self.head_bin = nn.Linear(out_dim, 2)       # Eq. (15)
        self.head_ml = nn.Linear(out_dim, n_ml)

    # ---- stage 1: per-node attended embedding z_i (Eq. 3-7) ---------------------------------
    def encode(self, x, return_alpha=False):
        B, T = x.shape[:2]
        maps = self.backbone(x.flatten(0, 1))                         # list of (B*T, d, h, w)
        Z = torch.stack([m.mean((2, 3)) for m in maps], 1)            # (B*T, S, d)
        Z = Z.view(B, T, Z.shape[1], Z.shape[2])                      # (B, T, S, d)
        if self.variant == "cnn_lstm":
            z, _ = self.lstm(Z[:, :, 0])                              # (B,T,d)
            z = z[:, -1]
            return (z, None) if return_alpha else z
        if self.use_views and T > 1:
            S = Z.shape[2]
            Zt = Z.permute(0, 2, 3, 1).reshape(B * S, Z.shape[3], T)  # (B*S, d, T)
            Z = F.relu(self.tconv(Zt)).mean(-1).view(B, S, -1)        # (B,S,d)  Eq. (5)
        else:
            Z = Z.mean(1)                                             # (B,S,d)
        if self.att is not None:
            z, alpha = self.att(Z)                                    # Eq. (6)-(7)
        else:
            z, alpha = Z.mean(1), None
        return (self.drop(z), alpha) if return_alpha else self.drop(z)

    # ---- stage 2: graph propagation + heads (Eq. 8-10, 15) ---------------------------------
    def propagate(self, H0, A_hat=None, A_raw=None):
        H = H0
        if self.use_graph:
            for l, layer in enumerate(self.gnn):
                H = layer(H, A_hat if self.gnn_type == "gcn" else A_raw)
                H = self.drop(F.relu(H)) if l < len(self.gnn) - 1 else F.relu(H)
        else:
            H = self.mlp(H)
        return H

    def heads(self, H):
        return self.head_bin(H), self.head_ml(H)

    def forward(self, x, A_hat=None, A_raw=None):
        return self.heads(self.propagate(self.encode(x), A_hat, A_raw))


# ---------------------------------------------------------------------------------------------
# Attention-based explanation map  (Eq. 23)  and graph influence  (revised Eq. 24)
# ---------------------------------------------------------------------------------------------
def attention_map(model: DamageNet, x: torch.Tensor):
    """Spatial explanation E_i = sum_s alpha_s * ||P_s(F_s)||  upsampled to input size.
    x: (1,T,3,H,W). Uses the first view. Returns (H,W) map in [0,1]."""
    model.eval()
    maps = model.backbone(x[:, 0])
    Z = torch.stack([m.mean((2, 3)) for m in maps], 1)
    _, alpha = model.att(Z) if model.att is not None else (None, torch.full((1, len(maps)), 1 / len(maps), device=x.device))
    H, W = x.shape[-2:]
    E = 0
    for s, m in enumerate(maps):
        e = m.norm(dim=1, keepdim=True)                                # (1,1,h,w)
        E = E + alpha[0, s] * F.interpolate(e, (H, W), mode="bilinear", align_corners=False)
    E = E[0, 0]
    return (E - E.min()) / (E.max() - E.min() + 1e-8)


def graph_influence(model: DamageNet, H0: torch.Tensor, A_hat, A_raw, i: int, cls: int = 1):
    """psi_ij = d y_i / d H0_j  (revised Eq. 24). Differentiating w.r.t. the INPUT node embedding
    H0_j is non-zero for neighbours within L hops; differentiating w.r.t. the final embedding H*_j
    (original Eq. 24) is identically zero because the readout is node-wise."""
    H0 = H0.detach().clone().requires_grad_(True)
    logits, _ = model.heads(model.propagate(H0, A_hat, A_raw))
    y_i = torch.softmax(logits[i], -1)[cls]
    g, = torch.autograd.grad(y_i, H0)
    return g.norm(dim=1)                                               # (N,) influence of every j on i
