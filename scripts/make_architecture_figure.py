import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(17, 9), dpi=200)
ax.set_xlim(0, 17); ax.set_ylim(0, 9); ax.axis('off')

def box(x, y, w, h, title, sub='', fc='#E8EEF6', ec='#2F4F6F', fs=9.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.08', fc=fc, ec=ec, lw=1.2))
    if sub:
        ax.text(x + w/2, y + h*0.70, title, ha='center', va='center', fontsize=fs, weight='bold')
        ax.text(x + w/2, y + h*0.32, sub, ha='center', va='center', fontsize=fs-1.5, color='#333', linespacing=1.25)
    else:
        ax.text(x + w/2, y + h/2, title, ha='center', va='center', fontsize=fs, weight='bold')

def arrow(x1, y1, x2, y2, label='', color='#222', lw=1.3, dx=0.0, dy=0.1):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=12, color=color, lw=lw))
    if label:
        ax.text((x1+x2)/2+dx, (y1+y2)/2+dy, label, ha='center', va='bottom', fontsize=7.8, color='#444', style='italic')

def line(x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-', color='#222', lw=1.3))

def frame(x, y, w, h, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.1', fc='#F6F8FB', ec='#9AA7B5', lw=1, ls='--'))
    ax.text(x + w/2, y + h - 0.22, title, ha='center', va='center', fontsize=9.2, weight='bold', color='#2F4F6F')

frame(0.2, 2.3, 5.6, 6.3, 'Block A — Input and feature learning (Sec. 3.1)')
box(0.45, 6.9, 2.4, 1.05, 'S2DS patch  x_i', 'T = 5 deterministic views\n(T, 3, 224, 224)')
box(3.15, 6.9, 2.4, 1.05, 'Visual-similarity graph', 'k-NN cosine, k = 8, inductive\nA ∈ {0,1}^(N×N)', fc='#FFF4E5', ec='#B36B00')
box(0.45, 5.25, 2.4, 1.2, 'ResNet-18 backbone', 'ImageNet init, shared over views\nlayer2 / layer3 / layer4')
box(3.15, 5.25, 2.4, 1.2, 'Multi-scale projection', '1×1 conv → d = 256, GAP\n(T, S = 3, 256)')
box(0.45, 3.6, 2.4, 1.2, 'View aggregation (Eq. 5)', '1-D conv over T, kernel 3, + mean\n(S, 256)')
box(3.15, 3.6, 2.4, 1.2, 'Scale attention (Eq. 6–7)', 'α ∈ R^S, z_i ∈ R^256\n→ explanation map E_i (Eq. 23)')
box(1.1, 2.55, 3.8, 0.65, 'Node embedding  H_i^(0) = z_i ∈ R^256', fc='#DCE9D5', ec='#3A6B35', fs=8.6)
arrow(1.65, 6.9, 1.65, 6.47)
arrow(2.85, 5.85, 3.15, 5.85)
ax.text(3.0, 6.0, 'F_i^t', ha='center', fontsize=7.8, color='#444', style='italic')
arrow(4.35, 5.25, 4.35, 4.82)
arrow(3.15, 4.2, 2.85, 4.2)
arrow(1.65, 3.6, 2.6, 3.22)
arrow(4.35, 3.6, 3.4, 3.22)

frame(6.0, 2.3, 5.1, 6.3, 'Block B — Graph-aware semi-supervised detection (Sec. 3.2)')
box(6.25, 6.85, 4.6, 1.15, 'GCN propagation (Eq. 8–10)', 'Â = D̃^(−½)(A+I)D̃^(−½), L = 2 layers, hidden 128\nmini-batch training with historical embeddings')
box(6.25, 5.2, 4.6, 1.2, 'Readout heads (Eq. 11–12)', 'binary softmax (damaged / no defect), threshold τ\nauxiliary 5-way sigmoid (crack, spalling, corrosion, effl., veg.)')
box(6.25, 3.65, 2.2, 1.15, 'L_sup (Eq. 13)', 'CE on labelled nodes V_L\n(25 % of train by default)', fc='#FBE9E7', ec='#A23B2A')
box(8.65, 3.65, 2.2, 1.15, 'L_unsup (Eq. 14–15)', 'FixMatch-style: weak-view pseudo-\nlabel, conf ≥ 0.95, strong view', fc='#FBE9E7', ec='#A23B2A')
box(6.25, 2.55, 4.6, 0.75, 'L = L_sup + λ L_unsup + β L_rob', 'Eq. 16–19;  λ = 0.5, β = 0.2;  L_rob = ‖p(x) − p(x+ε)‖², ε ~ N(0, 0.05²)', fc='#FBE9E7', ec='#A23B2A', fs=8.6)
line(4.9, 2.87, 5.9, 2.87); line(5.9, 2.87, 5.9, 7.2); arrow(5.9, 7.2, 6.25, 7.2)
ax.text(5.72, 7.0, 'H^(0)', ha='right', fontsize=7.8, color='#444', style='italic')
arrow(5.55, 7.65, 6.25, 7.65); ax.text(5.9, 7.72, 'A', ha='center', fontsize=7.8, color='#444', style='italic')
arrow(8.55, 6.85, 8.55, 6.4); ax.text(8.7, 6.55, 'H_i^* ∈ R^128', ha='left', fontsize=7.8, color='#444', style='italic')
arrow(7.35, 5.2, 7.35, 4.8); arrow(9.75, 5.2, 9.75, 4.8)
arrow(7.35, 3.65, 7.35, 3.3); arrow(9.75, 3.65, 9.75, 3.3)

frame(11.3, 2.3, 5.5, 6.3, 'Block C — Responsible-AI layer (Sec. 3.3)')
box(11.55, 6.85, 5.0, 1.15, 'Uncertainty (Eq. 20–22)', 'MC dropout, M = 20 passes → μ_i, σ_i\nECE, selective prediction, error-detection AUROC', fc='#E6F0E6', ec='#3A6B35')
box(11.55, 5.2, 5.0, 1.2, 'Explainability (Eq. 23–24)', 'attention map E_i: pointing game, IoU, deletion/insertion\ngraph influence ψ_ij = ∂y_i / ∂H_j^(0) + perturbation test', fc='#E6F0E6', ec='#3A6B35')
box(11.55, 3.65, 5.0, 1.15, 'Inspection-priority score (Eq. 25–27)', 'P_i ∝ μ_i (1 + σ_i) c_i — a ranking heuristic,\nnot an engineering risk estimate; validated by precision@k', fc='#E6F0E6', ec='#3A6B35')
box(11.55, 2.55, 5.0, 0.75, 'Outputs  d̂_i, μ_i, σ_i, E_i, ψ_ij, P_i', 'to the human-in-the-loop SHM decision support', fc='#DCE9D5', ec='#3A6B35', fs=8.8)
arrow(10.85, 5.7, 11.55, 5.7); ax.text(11.2, 5.76, 'ŷ_i', ha='center', fontsize=7.8, color='#444', style='italic')
line(10.85, 6.05, 11.2, 6.05); line(11.2, 6.05, 11.2, 7.4); arrow(11.2, 7.4, 11.55, 7.4)
arrow(14.05, 6.85, 14.05, 6.4); arrow(14.05, 5.2, 14.05, 4.8); arrow(14.05, 3.65, 14.05, 3.3)

ax.text(0.25, 1.75, 'Adopted components: ResNet-18, GCN, MC dropout, FixMatch-style pseudo-labelling, SHAP/LIME.   Proposed integration: multi-view aggregation + scale attention feeding an', fontsize=8.4, color='#333')
ax.text(0.25, 1.42, 'inductive visual-similarity graph, trained jointly with the unlabelled-consistency and robustness terms, and coupled to uncertainty-weighted inspection prioritisation.', fontsize=8.4, color='#333')
ax.text(0.25, 1.09, 'Grey dashed frames = the three functional blocks of Section 3.  Orange = graph input;  red = training objective;  green = Responsible-AI outputs.', fontsize=8.4, color='#333')
ax.text(0.25, 0.6, 'S2DS contains no repeated inspections: the T views are deterministic crops/scales of one patch, and the graph encodes visual similarity rather than measured structural connectivity (Sec. 4.2).', fontsize=8.4, color='#8B0000', style='italic')
plt.savefig('docs/figures/architecture.png', bbox_inches='tight', facecolor='white')
