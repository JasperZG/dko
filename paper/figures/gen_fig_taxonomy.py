import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({
    'font.size': 14,
    'font.family': 'sans-serif',
    'axes.linewidth': 0.8,
})

# ── Colours ───────────────────────────────────────────────────────────
BAR_CLR     = '#5B7C99'   # muted steel blue
ZERO_CLR    = '#D4D4D4'   # light gray for zero bars
TEXT_CLR    = '#1A1A1A'   # near-black
LABEL_CLR   = '#333333'   # dark charcoal
GRID_CLR    = '#CCCCCC'   # light gray
BRACE_CLR   = '#555555'   # bracket annotation color

# ── Data from Table 5 (hybrid Δ vs FP only) ──────────────────────────
datasets = [
    'ESOL',           # Solvation, -9.9%
    'FreeSolv',       # Solvation, -3.9%
    'Lipophilicity',  # Solvation,  0%
    'QM9-HOMO',       # Electronic, -4.2%
    'QM9-Gap',        # Electronic,  0%
    'QM9-LUMO',       # Electronic,  0%
]
improvements = [9.9, 3.9, 0.0, 4.2, 0.0, 0.0]  # % RMSE reduction

fig, ax = plt.subplots(figsize=(8, 5))

y_pos = np.arange(len(datasets))
bar_colors = [BAR_CLR if v > 0 else ZERO_CLR for v in improvements]

bars = ax.barh(y_pos, improvements, height=0.55, color=bar_colors,
               edgecolor='none', zorder=3)

# ── Percentage labels at bar tips ─────────────────────────────────────
for i, (val, bar) in enumerate(zip(improvements, bars)):
    if val > 0:
        ax.text(val + 0.25, i, f'{val:.1f}%', ha='left', va='center',
                fontsize=15, color=TEXT_CLR, fontweight='bold')
    else:
        ax.text(0.25, i, '0%', ha='left', va='center',
                fontsize=15, color='#999999')

# ── Axes styling ──────────────────────────────────────────────────────
ax.set_yticks(y_pos)
ax.set_yticklabels(datasets, fontsize=15, color=LABEL_CLR)
ax.invert_yaxis()
ax.set_xlabel('Hybrid RMSE improvement over FP-only baseline (%)',
              fontsize=15, color=LABEL_CLR, labelpad=10)
ax.set_xlim(-0.5, 15.5)

# Grid and spines
ax.xaxis.grid(True, color=GRID_CLR, linewidth=0.5, zorder=0)
ax.yaxis.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color(LABEL_CLR)
ax.spines['bottom'].set_color(LABEL_CLR)
ax.tick_params(axis='x', colors=LABEL_CLR, labelsize=14)
ax.tick_params(axis='y', length=0)

# ── Category brackets on right margin ─────────────────────────────────
bx = 13.5
tick_len = 0.3


def draw_bracket(y_top, y_bot, label):
    """Vertical bracket with horizontal ticks and centered label."""
    mid = (y_top + y_bot) / 2
    ax.plot([bx, bx], [y_top, y_bot], color=BRACE_CLR, lw=1.0,
            clip_on=False, zorder=5)
    ax.plot([bx - tick_len, bx], [y_top, y_top], color=BRACE_CLR, lw=1.0,
            clip_on=False, zorder=5)
    ax.plot([bx - tick_len, bx], [y_bot, y_bot], color=BRACE_CLR, lw=1.0,
            clip_on=False, zorder=5)
    ax.text(bx + 0.2, mid, label, ha='left', va='center', fontsize=14,
            color=BRACE_CLR, clip_on=False, style='italic')


draw_bracket(0, 2, 'Solvation')    # ESOL, FreeSolv, Lipo
draw_bracket(3, 5, 'Electronic')   # QM9-HOMO, QM9-Gap, QM9-LUMO

# Title
ax.set_title('Conformer utility by property type',
             fontsize=17, fontweight='bold', color=LABEL_CLR, pad=14)

plt.tight_layout(pad=0.8)
outdir = os.path.join('C:', os.sep, 'Users', 'zhaoz', 'Downloads', 'dko',
                      'paper', 'figures')
fig.savefig(os.path.join(outdir, 'fig_taxonomy.pdf'),
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved fig_taxonomy.pdf')
