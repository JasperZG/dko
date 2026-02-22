import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({
    'font.size': 13,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'font.family': 'sans-serif',
})

# Data
datasets = ['ESOL', 'FreeSolv', 'Lipo.', 'QM9-Gap', 'QM9-HOMO', 'QM9-LUMO']
fp_rmse   = [1.507, 2.939, 0.910, 0.020, 0.0142, 0.019]
hyb_rmse  = [1.358, 2.824, 0.957, 0.021, 0.0136, 0.019]

# Normalize to FP = 1.0
fp_rel  = [1.0] * 6
hyb_rel = [h / f for h, f in zip(hyb_rmse, fp_rmse)]

# Percentage improvement (negative means hybrid is better)
pct = [(h - f) / f * 100 for h, f in zip(hyb_rmse, fp_rmse)]

x = np.arange(len(datasets))
width = 0.32

fig, ax = plt.subplots(figsize=(10, 5.5))

bars_fp  = ax.bar(x - width/2, fp_rel,  width, label='FP-only (Morgan + XGBoost)',
                  color='#3498DB', edgecolor='#2471A3', lw=0.8, zorder=3)
bars_hyb = ax.bar(x + width/2, hyb_rel, width, label=r'Hybrid (FP + $\mu$ + $\sigma_5$)',
                  color='#E67E22', edgecolor='#CA6F1E', lw=0.8, zorder=3)

# Dashed baseline at 1.0
ax.axhline(1.0, color='#566573', ls='--', lw=1.0, zorder=2)

# Annotate improvements
for i, p in enumerate(pct):
    if p < -1.0:  # meaningful improvement
        ax.annotate(f'{p:.1f}%',
                    xy=(x[i] + width/2, hyb_rel[i]),
                    xytext=(0, 8), textcoords='offset points',
                    ha='center', va='bottom', fontsize=10, fontweight='bold',
                    color='#27AE60')
    elif p > 1.0:  # hybrid worse
        ax.annotate(f'+{p:.1f}%',
                    xy=(x[i] + width/2, hyb_rel[i]),
                    xytext=(0, 8), textcoords='offset points',
                    ha='center', va='bottom', fontsize=10, fontweight='bold',
                    color='#C0392B')

ax.set_ylabel('Relative RMSE (FP-only = 1.0)')
ax.set_xlabel('Dataset')
ax.set_title('FP-only vs. Hybrid: Relative RMSE Comparison')
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.set_ylim(0.82, 1.15)
ax.legend(loc='upper left', framealpha=0.9, fontsize=11)
ax.grid(axis='y', alpha=0.3, zorder=0)
ax.set_axisbelow(True)

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
outdir = os.path.join('C:', os.sep, 'Users', 'zhaoz', 'Downloads', 'dko', 'paper', 'figures')
fig.savefig(os.path.join(outdir, 'fig_performance.pdf'),
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved fig_performance.pdf')
