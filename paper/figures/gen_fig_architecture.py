import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

# ── Nature-quality muted palette ──────────────────────────────────────
BOX_FILL    = '#F7F7F7'   # very light warm gray
BOX_EDGE    = '#4A4A4A'   # charcoal
DKO_FILL    = '#D0D8E8'   # very pale slate (Path A accent)
XGB_FILL    = '#E8E0D0'   # very pale warm  (Path B accent)
ARROW_CLR   = '#666666'   # medium gray
TEXT_CLR     = '#1A1A1A'   # near-black
SUB_CLR     = '#555555'   # dark gray
WHITE        = '#FFFFFF'

fig, ax = plt.subplots(figsize=(15, 8))
ax.set_xlim(0, 15)
ax.set_ylim(0, 8)
ax.axis('off')
fig.patch.set_facecolor(WHITE)


def draw_box(ax, xy, w, h, text, facecolor=BOX_FILL, edgecolor=BOX_EDGE,
             fontsize=16, fontweight='normal', text_color=TEXT_CLR, lw=1.0,
             style='round,pad=0.12', zorder=2, extra_text=None, extra_fs=13):
    """Draw a rounded box with optional subtitle."""
    box = FancyBboxPatch(xy, w, h, boxstyle=style,
                         facecolor=facecolor, edgecolor=edgecolor,
                         lw=lw, zorder=zorder)
    ax.add_patch(box)
    cx, cy = xy[0] + w / 2, xy[1] + h / 2
    if extra_text:
        ax.text(cx, cy + 0.20, text, ha='center', va='center',
                fontsize=fontsize, fontweight=fontweight,
                color=text_color, zorder=zorder + 1)
        ax.text(cx, cy - 0.25, extra_text, ha='center', va='center',
                fontsize=extra_fs, color=SUB_CLR, style='italic',
                zorder=zorder + 1)
    else:
        ax.text(cx, cy, text, ha='center', va='center',
                fontsize=fontsize, fontweight=fontweight,
                color=text_color, zorder=zorder + 1)


def arrow(ax, x1, y1, x2, y2, color=ARROW_CLR, lw=1.4):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw),
                zorder=1)


# ── Title ─────────────────────────────────────────────────────────────
ax.text(7.0, 7.65, 'Distribution Kernel Operator (DKO) Pipeline',
        ha='center', va='center', fontsize=18, fontweight='bold',
        color=BOX_EDGE)

# ── Path A: Neural DKO ────────────────────────────────────────────────
ax.text(7.0, 7.05, 'Path A: DKO (Neural)', ha='center', fontsize=15,
        fontweight='bold', color=TEXT_CLR, style='italic')
ax.plot([0.3, 14.7], [6.85, 6.85], color='#CCCCCC', lw=0.8, ls='--')

ROW_A = 5.70
bh = 1.10

# SMILES
draw_box(ax, (0.2, ROW_A - bh/2), 1.6, bh, 'SMILES', fontweight='bold')
arrow(ax, 1.8, ROW_A, 2.25, ROW_A)

# ETKDG
draw_box(ax, (2.25, ROW_A - bh/2), 2.1, bh, 'RDKit ETKDG',
         extra_text='conformer gen.')
arrow(ax, 4.35, ROW_A, 4.80, ROW_A)

# Conformers
draw_box(ax, (4.80, ROW_A - bh/2), 2.1, bh, 'Conformers',
         extra_text='n = 50')
arrow(ax, 6.90, ROW_A, 7.30, ROW_A)

# Features
draw_box(ax, (7.30, ROW_A - bh/2), 1.85, bh, 'Features',
         extra_text=r'$x_i \in \mathbb{R}^D$')
arrow(ax, 9.15, ROW_A, 9.55, ROW_A)

# Statistics
draw_box(ax, (9.55, ROW_A - bh/2), 1.65, bh, 'Statistics',
         extra_text=r'$\mu, \Sigma$')
arrow(ax, 11.20, ROW_A, 11.55, ROW_A)

# DKO (subtle accent)
draw_box(ax, (11.55, ROW_A - bh/2), 1.90, bh, 'DKO',
         facecolor=DKO_FILL, fontweight='bold',
         extra_text=r'$K\!=\!LL^\top$ + MLP')
arrow(ax, 13.45, ROW_A, 13.80, ROW_A)

# Prediction
draw_box(ax, (13.60, ROW_A - bh/2), 0.70, bh, r'$\hat{y}$',
         fontsize=18, fontweight='bold')

# ── Path B: Hybrid ────────────────────────────────────────────────────
ax.text(7.0, 4.15, r'Path B: Hybrid (FP + $\mu$ + $\sigma_5$ $\rightarrow$ XGBoost)',
        ha='center', fontsize=15, fontweight='bold', color=TEXT_CLR,
        style='italic')
ax.plot([0.3, 14.7], [3.92, 3.92], color='#CCCCCC', lw=0.8, ls='--')

ROW_B = 2.45
SUB_TOP = ROW_B + 0.75
SUB_BOT = ROW_B - 0.75
sub_h = 0.95

# SMILES
draw_box(ax, (0.2, ROW_B - bh/2), 1.6, bh, 'SMILES', fontweight='bold')

# Split arrows
arrow(ax, 1.8, ROW_B, 2.5, SUB_TOP)
arrow(ax, 1.8, ROW_B, 2.5, SUB_BOT)

# Morgan FP
draw_box(ax, (2.5, SUB_TOP - sub_h/2), 2.5, sub_h, 'Morgan FP',
         extra_text='2048-bit')

# Conformers
draw_box(ax, (2.5, SUB_BOT - sub_h/2), 2.5, sub_h, 'Conformers',
         extra_text=r'$\mu, \sigma_5 \in \mathbb{R}^D$')

# Concatenate (plain text, merge arrows)
cat_x = 6.2
ax.text(cat_x, ROW_B, 'concatenate', ha='center', va='center',
        fontsize=13, color=SUB_CLR, style='italic', zorder=4)

arrow(ax, 5.0, SUB_TOP, cat_x - 0.15, ROW_B + 0.14)
arrow(ax, 5.0, SUB_BOT, cat_x - 0.15, ROW_B - 0.14)
arrow(ax, cat_x + 0.65, ROW_B, 7.4, ROW_B)

# XGBoost (subtle accent)
draw_box(ax, (7.4, ROW_B - bh/2), 1.9, bh, 'XGBoost',
         facecolor=XGB_FILL, fontweight='bold')
arrow(ax, 9.3, ROW_B, 9.7, ROW_B)

# Prediction
draw_box(ax, (9.50, ROW_B - bh/2), 0.70, bh, r'$\hat{y}$',
         fontsize=18, fontweight='bold')

# ── Bottom notation legend ────────────────────────────────────────────
ax.text(7.0, 0.45,
        r'$\mu$ = conformer mean,  $\Sigma$ = conformer covariance,  '
        r'$\sigma_5$ = 5 covariance summary statistics '
        r'(trace, log-det, $\|\cdot\|_F$, eigenvalue ratios),  '
        r'$D$ = feature dimension',
        ha='center', fontsize=13, color=SUB_CLR, style='italic')

plt.tight_layout(pad=0.5)
outdir = os.path.join('C:', os.sep, 'Users', 'zhaoz', 'Downloads', 'dko',
                      'paper', 'figures')
fig.savefig(os.path.join(outdir, 'fig_architecture.pdf'),
            dpi=300, bbox_inches='tight', facecolor=WHITE)
plt.close()
print('Saved fig_architecture.pdf')
