import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

LIGHT_BLUE   = '#D6EAF8'
MED_BLUE     = '#85C1E9'
DARK_BLUE    = '#2471A3'
LIGHT_GRAY   = '#F2F3F4'
MED_GRAY     = '#D5D8DC'
DARK_GRAY    = '#566573'
ACCENT_ORANGE = '#F39C12'
ACCENT_GREEN  = '#27AE60'
WHITE         = '#FFFFFF'
BOX_EDGE      = '#5D6D7E'

fig, ax = plt.subplots(figsize=(14, 7.5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7.5)
ax.axis('off')
fig.patch.set_facecolor(WHITE)

def draw_box(ax, xy, w, h, text, facecolor=LIGHT_BLUE, edgecolor=BOX_EDGE,
             fontsize=11, fontweight='normal', text_color='black', lw=1.3,
             style='round,pad=0.12', zorder=2, ha='center', va='center',
             extra_text=None, extra_fs=8.5):
    box = FancyBboxPatch(xy, w, h, boxstyle=style,
                         facecolor=facecolor, edgecolor=edgecolor, lw=lw, zorder=zorder)
    ax.add_patch(box)
    cx, cy = xy[0] + w / 2, xy[1] + h / 2
    if extra_text:
        ax.text(cx, cy + 0.13, text, ha=ha, va=va, fontsize=fontsize,
                fontweight=fontweight, color=text_color, zorder=zorder + 1)
        ax.text(cx, cy - 0.18, extra_text, ha=ha, va=va, fontsize=extra_fs,
                color=DARK_GRAY, style='italic', zorder=zorder + 1)
    else:
        ax.text(cx, cy, text, ha=ha, va=va, fontsize=fontsize,
                fontweight=fontweight, color=text_color, zorder=zorder + 1)

def arrow(ax, x1, y1, x2, y2, color=DARK_GRAY, lw=1.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw), zorder=1)

def draw_conformer_blob(ax, cx, cy, r=0.22, color=MED_BLUE, alpha=0.6):
    offsets = [(-0.08, 0.08), (0.1, 0.05), (0.0, -0.08)]
    for dx, dy in offsets:
        c = plt.Circle((cx + dx, cy + dy), r * 0.45, fc=color, ec=BOX_EDGE,
                        lw=0.7, alpha=alpha, zorder=3)
        ax.add_patch(c)
    ax.plot([cx - 0.08, cx + 0.1], [cy + 0.08, cy + 0.05],
            color=BOX_EDGE, lw=0.7, zorder=3)
    ax.plot([cx + 0.1, cx + 0.0], [cy + 0.05, cy - 0.08],
            color=BOX_EDGE, lw=0.7, zorder=3)

# Title
ax.text(7.0, 7.15, 'Distribution Kernel Operator (DKO) Pipeline',
        ha='center', va='center', fontsize=15, fontweight='bold', color=DARK_BLUE)

ax.text(7.0, 6.65, 'Path A: DKO (Neural)', ha='center', fontsize=11,
        fontweight='bold', color=DARK_BLUE, style='italic')
ax.plot([0.3, 13.7], [6.5, 6.5], color=MED_GRAY, lw=0.8, ls='--')

ROW_A = 5.5
bh = 0.8

draw_box(ax, (0.3, ROW_A - bh/2), 1.4, bh, 'SMILES', fontsize=12,
         fontweight='bold', facecolor=LIGHT_GRAY)
arrow(ax, 1.7, ROW_A, 2.2, ROW_A)

draw_box(ax, (2.2, ROW_A - bh/2), 1.8, bh, 'RDKit ETKDG',
         facecolor=LIGHT_BLUE, extra_text='conformer gen.', fontsize=11)
arrow(ax, 4.0, ROW_A, 4.55, ROW_A)

ens_x, ens_w = 4.55, 1.9
draw_box(ax, (ens_x, ROW_A - bh/2), ens_w, bh, '',
         facecolor='#EBF5FB', edgecolor=BOX_EDGE)
ax.text(ens_x + ens_w / 2, ROW_A + 0.2, 'n = 50 conformers',
        ha='center', fontsize=9, color=DARK_GRAY, style='italic', zorder=4)
for i, dx in enumerate(np.linspace(0.3, 1.6, 5)):
    draw_conformer_blob(ax, ens_x + dx, ROW_A - 0.12,
                        color=MED_BLUE, alpha=0.5 + 0.1 * (i % 2))
arrow(ax, 6.45, ROW_A, 7.05, ROW_A)

draw_box(ax, (7.05, ROW_A - bh/2), 1.6, bh, 'Features',
         facecolor=LIGHT_BLUE, extra_text=r'$x_i \in \mathbb{R}^D$', fontsize=11)
arrow(ax, 8.65, ROW_A, 9.2, ROW_A)

draw_box(ax, (9.2, ROW_A - bh/2), 1.5, bh, 'Statistics',
         facecolor=LIGHT_BLUE, extra_text=r'$\mu, \Sigma$', fontsize=11)
arrow(ax, 10.7, ROW_A, 11.25, ROW_A)

draw_box(ax, (11.25, ROW_A - bh/2), 1.5, bh, 'DKO',
         facecolor='#D5F5E3', edgecolor='#1E8449', fontweight='bold',
         extra_text=r'$K\!=\!LL^T$ + MLP', fontsize=12)
arrow(ax, 12.75, ROW_A, 13.2, ROW_A)

draw_box(ax, (13.0, ROW_A - bh/2), 0.7, bh, r'$\hat{y}$',
         facecolor='#FADBD8', edgecolor='#C0392B', fontsize=13, fontweight='bold')

# PATH B - Hybrid
ax.text(7.0, 3.95, r'Path B: Hybrid (FP + $\mu$ + $\sigma_5$ $\rightarrow$ XGBoost)',
        ha='center', fontsize=11, fontweight='bold', color=ACCENT_ORANGE, style='italic')
ax.plot([0.3, 13.7], [3.8, 3.8], color=MED_GRAY, lw=0.8, ls='--')

ROW_B = 2.8
draw_box(ax, (0.3, ROW_B - bh/2), 1.4, bh, 'SMILES', fontsize=12,
         fontweight='bold', facecolor=LIGHT_GRAY)

SUB_TOP = ROW_B + 0.55
SUB_BOT = ROW_B - 0.55

arrow(ax, 1.7, ROW_B, 2.4, SUB_TOP, lw=1.3)
arrow(ax, 1.7, ROW_B, 2.4, SUB_BOT, lw=1.3)

draw_box(ax, (2.4, SUB_TOP - 0.35), 2.2, 0.7, 'Morgan FP',
         facecolor='#EBF5FB', extra_text='2048-bit', fontsize=11)
draw_box(ax, (2.4, SUB_BOT - 0.35), 2.2, 0.7, 'Conformers',
         facecolor='#EBF5FB', extra_text=r'$\mu, \sigma_5 \in \mathbb{R}^D$', fontsize=11)

cat_x, cat_y = 5.8, ROW_B
circle = plt.Circle((cat_x, cat_y), 0.35, fc='#FCF3CF', ec=ACCENT_ORANGE,
                     lw=1.5, zorder=3)
ax.add_patch(circle)
ax.text(cat_x, cat_y, r'$\oplus$', ha='center', va='center', fontsize=18,
        fontweight='bold', color=ACCENT_ORANGE, zorder=4)
ax.text(cat_x, cat_y - 0.52, 'concat', ha='center', fontsize=9,
        color=DARK_GRAY, style='italic')

arrow(ax, 4.6, SUB_TOP, 5.48, cat_y + 0.1, lw=1.3)
arrow(ax, 4.6, SUB_BOT, 5.48, cat_y - 0.1, lw=1.3)
arrow(ax, 6.15, cat_y, 7.0, cat_y, lw=1.3)

draw_box(ax, (7.0, ROW_B - bh/2), 1.8, bh, 'XGBoost',
         facecolor='#FDEBD0', edgecolor=ACCENT_ORANGE, fontsize=12, fontweight='bold')
arrow(ax, 8.8, ROW_B, 9.4, ROW_B, lw=1.3)

draw_box(ax, (9.4, ROW_B - bh/2), 0.7, bh, r'$\hat{y}$',
         facecolor='#FADBD8', edgecolor='#C0392B', fontsize=13, fontweight='bold')

# Key finding callout
draw_box(ax, (10.5, 2.25), 3.1, 1.1, '',
         facecolor='#FDFEFE', edgecolor=ACCENT_GREEN, lw=1.8, style='round,pad=0.15')
ax.text(12.05, 2.98, 'Key finding', ha='center', fontsize=10.5,
        fontweight='bold', color=ACCENT_GREEN, zorder=5)
ax.text(12.05, 2.65, 'Hybrid improves solvation', ha='center',
        fontsize=9.5, color=DARK_GRAY, zorder=5)
ax.text(12.05, 2.42, 'RMSE by 4\u201310%', ha='center',
        fontsize=9.5, color=DARK_GRAY, zorder=5)

# Bottom note
ax.text(7.0, 0.35,
        r'$\mu$ = conformer mean,  $\Sigma$ = conformer covariance,  '
        r'$\sigma_5$ = top-5 singular values of $\Sigma$,  '
        r'$D$ = feature dimension',
        ha='center', fontsize=9, color=DARK_GRAY, style='italic')

plt.tight_layout(pad=0.5)
outdir = os.path.join('C:', os.sep, 'Users', 'zhaoz', 'Downloads', 'dko', 'paper', 'figures')
fig.savefig(os.path.join(outdir, 'fig_architecture.pdf'),
            dpi=300, bbox_inches='tight', facecolor=WHITE)
plt.close()
print('Saved fig_architecture.pdf')
