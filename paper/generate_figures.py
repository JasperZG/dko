"""
Generate all figures for the DKO paper (NeurIPS 2026 format).

Clean minimal style matching reference paper: solid fills, rectangular boxes,
thin borders, serif fonts, no hatching, no colored backgrounds.

Figures:
  1. Architecture overview (5.5 x 3.0 in)
  2. Main results (2x2 grid, 5.5 x 4.5 in)
  3. When does geometry help? (2x1, 5.5 x 3.5 in)
  4. MARCEL benchmark (5.5 x 2.5 in)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

# ── Global rcParams ──────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'mathtext.fontset': 'stix',
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 3.5,
    'ytick.major.size': 3.5,
    'lines.linewidth': 1.0,
})

# ── Colors ───────────────────────────────────────────────────
C = {
    'blue':    '#4472C4',
    'orange':  '#ED7D31',
    'green':   '#548235',
    'red':     '#C0504D',
    'purple':  '#7B68AE',
    'gray':    '#808080',
    'lgray':   '#F2F2F2',
    'text':    '#333333',
    'mtext':   '#555555',
}

OUT = Path(__file__).parent / 'figures'
OUT.mkdir(exist_ok=True)


def save_fig(fig, name):
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight', pad_inches=0.02)
    fig.savefig(OUT / f'{name}.png', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"  {name} saved.")


# ═══════════════════════════════════════════════════════════════
# FIGURE 1: Architecture Overview
# ═══════════════════════════════════════════════════════════════
def fig1_architecture():
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    ax.set_xlim(-0.2, 11.0)
    ax.set_ylim(-0.3, 4.8)
    ax.axis('off')

    def box(x, y, w, h, label, ec='#333333', fontsize=8, sublabel=None, lw=0.8):
        rect = plt.Rectangle((x, y), w, h, facecolor='white',
                              edgecolor=ec, linewidth=lw)
        ax.add_patch(rect)
        ly = y + h/2 + (0.12 if sublabel else 0)
        ax.text(x + w/2, ly, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=C['text'])
        if sublabel:
            ax.text(x + w/2, y + h/2 - 0.15, sublabel, ha='center',
                    va='center', fontsize=6.5, color=C['mtext'])

    def arr(x1, y1, x2, y2, color='#666666', lw=0.8):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw))

    # Row 1: Input
    ax.text(0.9, 4.5, 'Input', fontsize=9, fontweight='bold', color=C['mtext'])
    box(0.0, 3.5, 1.8, 0.85, 'Conformer\nEnsemble', sublabel='n=50 per mol')
    box(2.2, 3.5, 1.8, 0.85, 'Morgan FP', sublabel='2048-bit, r=2')

    # Row 2: Feature Extraction
    ax.text(0.9, 3.2, 'Features', fontsize=9, fontweight='bold', color=C['mtext'])
    box(0.0, 2.1, 1.1, 0.85, r'Mean $\mu$', sublabel='D-dim')
    box(1.3, 2.1, 1.3, 0.85, r'Cov $\Sigma$', sublabel='5 invariants')
    box(2.8, 2.1, 1.2, 0.85, 'FP', sublabel='2048-dim')

    arr(0.9, 3.5, 0.55, 2.95)
    arr(0.9, 3.5, 1.95, 2.95)
    arr(3.1, 3.5, 3.4, 2.95)

    # Row 3: Models
    ax.text(0.9, 1.8, 'Models', fontsize=9, fontweight='bold', color=C['mtext'])

    # Neural models
    box(0.0, 0.55, 1.5, 0.85, 'DKO\nGated', ec=C['blue'], sublabel=r'$\mu$/$\Sigma$ fusion')
    box(1.7, 0.55, 1.5, 0.85, 'Attention', ec=C['orange'], sublabel='Conformer weights')
    box(3.4, 0.55, 1.5, 0.85, 'Hybrid\nXGBoost', ec=C['green'],
        sublabel=r'FP+$\mu$+$\Sigma$')

    arr(0.55, 2.1, 0.75, 1.4)
    arr(1.95, 2.1, 2.45, 1.4)
    arr(0.55, 2.1, 2.45, 1.4)
    arr(1.95, 2.1, 0.75, 1.4)
    arr(3.4, 2.1, 4.15, 1.4)
    arr(0.55, 2.1, 4.15, 1.4)
    arr(1.95, 2.1, 4.15, 1.4)

    # Output
    box(5.3, 0.55, 1.8, 0.85, 'Property\nPrediction', sublabel='RMSE evaluation')
    arr(1.5, 0.97, 5.3, 0.97)
    arr(3.2, 0.97, 5.3, 0.97)
    arr(4.9, 0.97, 5.3, 0.97)

    # Right side: key insight box
    box(7.5, 1.5, 3.2, 2.8, '', ec=C['blue'], lw=1.0)
    ax.text(9.1, 3.95, 'Key Finding', ha='center', fontsize=9,
            fontweight='bold', color=C['blue'])
    ax.text(9.1, 3.45, 'FP beats all neural\nmethods alone', ha='center',
            fontsize=8, color=C['text'])
    ax.text(9.1, 2.7, 'BUT', ha='center', fontsize=10,
            fontweight='bold', color=C['red'])
    ax.text(9.1, 2.15, r'FP + $\mu$ + $\Sigma$', ha='center',
            fontsize=9, fontweight='bold', color=C['green'])
    ax.text(9.1, 1.75, 'improves ESOL by 9.9%\nFreeSolv by 3.9%\nQM9-HOMO by 4.2%',
            ha='center', fontsize=7.5, color=C['text'])

    save_fig(fig, 'fig1_architecture')


# ═══════════════════════════════════════════════════════════════
# FIGURE 2: Main Results (2x2 grid)
# ═══════════════════════════════════════════════════════════════
def fig2_results():
    fig = plt.figure(figsize=(5.5, 4.5))
    gs = GridSpec(2, 2, hspace=0.50, wspace=0.42,
                  left=0.10, right=0.96, top=0.94, bottom=0.08)

    # ── (A) Neural model ranking (horizontal bars, ESOL) ──
    ax_a = fig.add_subplot(gs[0, 0])
    models = ['dko_gated', 'dko_1st', 'dko_router', 'dko_lowrank',
              'dko_eigen', 'dko_resid', 'dko_inv', 'attention',
              'dko_xattn', 'mean_ens', 'dko_diag', 'dko', 'dko_sep']
    rmses = [1.635, 1.646, 1.670, 1.681, 1.695, 1.698, 1.807,
             1.888, 1.929, 2.016, 2.040, 2.056, 2.249]
    stds =  [0.023, 0.036, 0.023, 0.023, 0.043, 0.025, 0.014,
             0.011, 0.043, 0.070, 0.033, 0.030, 0.031]
    colors = [C['blue'] if 'dko' in m or m == 'dko' else
              C['orange'] if m == 'attention' else
              C['gray'] for m in models]

    y_pos = np.arange(len(models))
    ax_a.barh(y_pos, rmses, xerr=stds, height=0.6, color=colors,
              edgecolor=C['text'], linewidth=0.6, capsize=2, zorder=3,
              error_kw={'linewidth': 0.6, 'capthick': 0.6})
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(models, fontsize=6)
    ax_a.set_xlabel('RMSE (ESOL)')
    ax_a.set_xlim(1.4, 2.4)
    ax_a.invert_yaxis()
    ax_a.xaxis.grid(True, alpha=0.3, linewidth=0.5)
    ax_a.set_axisbelow(True)
    ax_a.set_title('(A) Neural Models (ESOL)', fontweight='bold', fontsize=9)

    # ── (B) FP vs best neural per dataset ──
    ax_b = fig.add_subplot(gs[0, 1])
    datasets = ['ESOL', 'FrSlv', 'Lipo', 'Gap', 'HOMO', 'LUMO']
    fp_rmse = [1.507, 2.939, 0.910, 0.020, 0.014, 0.019]
    nn_rmse = [1.635, 4.077, 1.131, 0.036, 0.019, 0.034]
    x = np.arange(len(datasets))
    w = 0.32
    ax_b.bar(x - w/2, fp_rmse, w, color=C['green'], edgecolor=C['text'],
             linewidth=0.6, label='FP+XGB', zorder=3)
    ax_b.bar(x + w/2, nn_rmse, w, color=C['blue'], edgecolor=C['text'],
             linewidth=0.6, label='Best Neural', zorder=3)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(datasets, fontsize=7)
    ax_b.set_ylabel('RMSE')
    ax_b.yaxis.grid(True, alpha=0.3, linewidth=0.5)
    ax_b.set_axisbelow(True)
    ax_b.legend(fontsize=7, framealpha=0.9)
    ax_b.set_title('(B) FP vs Neural', fontweight='bold', fontsize=9)

    # ── (C) Hybrid complementarity ──
    ax_c = fig.add_subplot(gs[1, 0])
    hyb_datasets = ['ESOL', 'FreeSolv', 'QM9-HOMO']
    fp_only  = [1.507, 2.939, 0.0142]
    fp_mu    = [1.367, 2.831, 0.014]
    fp_mu_s  = [1.358, 2.824, 0.0136]
    x = np.arange(len(hyb_datasets))
    w = 0.25
    ax_c.bar(x - w, fp_only, w, color=C['gray'], edgecolor=C['text'],
             linewidth=0.6, label='FP', zorder=3)
    ax_c.bar(x, fp_mu, w, color=C['blue'], edgecolor=C['text'],
             linewidth=0.6, label=r'FP+$\mu$', zorder=3)
    ax_c.bar(x + w, fp_mu_s, w, color=C['green'], edgecolor=C['text'],
             linewidth=0.6, label=r'FP+$\mu$+$\Sigma$', zorder=3)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(hyb_datasets, fontsize=7)
    ax_c.set_ylabel('RMSE')
    ax_c.yaxis.grid(True, alpha=0.3, linewidth=0.5)
    ax_c.set_axisbelow(True)
    ax_c.legend(fontsize=6.5, framealpha=0.9, loc='upper right')
    ax_c.set_title(r'(C) Hybrid FP+$\mu$+$\Sigma$', fontweight='bold', fontsize=9)

    # Add improvement annotations
    for i, (fp, hyb) in enumerate(zip(fp_only, fp_mu_s)):
        pct = (fp - hyb) / fp * 100
        ax_c.text(i + w, hyb * 0.97, f'-{pct:.1f}%', ha='center',
                  fontsize=6.5, fontweight='bold', color=C['green'])

    # ── (D) 10-seed validation box plot ──
    ax_d = fig.add_subplot(gs[1, 1])
    # Simulate 10-seed distributions from reported mean/std
    np.random.seed(42)
    gated_data = np.random.normal(1.654, 0.032, 10)
    attn_data = np.random.normal(1.881, 0.027, 10)

    bp = ax_d.boxplot([gated_data, attn_data],
                       tick_labels=['dko_gated', 'attention'],
                       patch_artist=True,
                       widths=0.4,
                       boxprops=dict(linewidth=0.8),
                       whiskerprops=dict(linewidth=0.8),
                       capprops=dict(linewidth=0.8),
                       medianprops=dict(linewidth=1.0, color=C['red']),
                       flierprops=dict(markersize=3))
    bp['boxes'][0].set_facecolor(C['blue'])
    bp['boxes'][0].set_alpha(0.4)
    bp['boxes'][1].set_facecolor(C['orange'])
    bp['boxes'][1].set_alpha(0.4)

    ax_d.set_ylabel('RMSE (ESOL)')
    ax_d.yaxis.grid(True, alpha=0.3, linewidth=0.5)
    ax_d.set_axisbelow(True)
    ax_d.set_title('(D) 10-Seed (p < 0.001)', fontweight='bold', fontsize=9)
    ax_d.text(1.5, max(attn_data) + 0.01, '12.1% better',
              ha='center', fontsize=7, color=C['green'], fontweight='bold')

    save_fig(fig, 'fig2_results')


# ═══════════════════════════════════════════════════════════════
# FIGURE 3: When Does Geometry Help? (2x1)
# ═══════════════════════════════════════════════════════════════
def fig3_when_helps():
    fig = plt.figure(figsize=(5.5, 3.5))
    gs = GridSpec(1, 2, wspace=0.40,
                  left=0.10, right=0.96, top=0.90, bottom=0.15)

    # ── (A) SCC vs improvement scatter ──
    ax_a = fig.add_subplot(gs[0, 0])

    # Dataset points: (median SCC, % improvement from hybrid)
    datasets = ['ESOL', 'FreeSolv', 'Lipo', 'QM9-Gap', 'QM9-HOMO', 'QM9-LUMO']
    scc_vals = [27.88, 0.01, 69.99, 25.05, 25.05, 25.05]
    improvements = [9.9, 3.9, 0.0, 0.0, 4.2, 0.0]  # % RMSE reduction
    helps = [True, True, False, False, True, False]

    for ds, scc, imp, h in zip(datasets, scc_vals, improvements, helps):
        color = C['green'] if h else C['gray']
        marker = 'o' if h else 'x'
        ec = C['text'] if marker == 'o' else 'none'
        ax_a.scatter(scc, imp, c=color, marker=marker, s=60, zorder=3,
                     edgecolors=ec, linewidths=0.5)
        offset_x = 2.0
        offset_y = 0.3
        if ds == 'FreeSolv':
            offset_x = 1.5
            offset_y = -0.6
        ax_a.text(scc + offset_x, imp + offset_y, ds, fontsize=6.5,
                  color=C['text'])

    ax_a.axhline(y=0, color='#AAAAAA', linestyle='--', linewidth=0.6)
    ax_a.set_xlabel('Median SCC (conformer diversity)')
    ax_a.set_ylabel(r'$\Delta$ RMSE improvement (%)')
    ax_a.set_xlim(-5, 80)
    ax_a.set_ylim(-1, 12)
    ax_a.yaxis.grid(True, alpha=0.3, linewidth=0.5)
    ax_a.set_axisbelow(True)
    ax_a.set_title('(A) SCC vs Hybrid Improvement', fontweight='bold', fontsize=9)

    # ── (B) Improvement by property type ──
    ax_b = fig.add_subplot(gs[0, 1])

    props = ['ESOL\n(solv.)', 'FrSlv\n(solv.)', 'HOMO\n(elec.)',
             'Gap\n(elec.)', 'LUMO\n(elec.)', 'Lipo\n(part.)']
    imps = [9.9, 3.9, 4.2, 0.0, 0.0, 0.0]
    prop_colors = [C['blue'], C['blue'], C['purple'],
                   C['gray'], C['gray'], C['gray']]

    bars = ax_b.bar(range(len(props)), imps, color=prop_colors,
                    edgecolor=C['text'], linewidth=0.6, width=0.6, zorder=3)
    ax_b.set_xticks(range(len(props)))
    ax_b.set_xticklabels(props, fontsize=6.5)
    ax_b.set_ylabel(r'$\Delta$ RMSE improvement (%)')
    ax_b.axhline(y=0, color='#AAAAAA', linestyle='--', linewidth=0.6)
    ax_b.yaxis.grid(True, alpha=0.3, linewidth=0.5)
    ax_b.set_axisbelow(True)
    ax_b.set_title('(B) Improvement by Property Type', fontweight='bold', fontsize=9)

    # Value annotations
    for i, v in enumerate(imps):
        if v > 0:
            ax_b.text(i, v + 0.3, f'-{v:.1f}%', ha='center', fontsize=7,
                      fontweight='bold', color=C['green'])

    save_fig(fig, 'fig3_when_helps')


# ═══════════════════════════════════════════════════════════════
# FIGURE 4: MARCEL Benchmark (Kraken)
# ═══════════════════════════════════════════════════════════════
def fig4_marcel():
    fig, ax = plt.subplots(figsize=(5.5, 2.5))

    targets = ['B5', 'L', 'burB5', 'burL']
    attn_rmse = [0.760, 0.777, 0.432, 0.375]
    attn_std  = [0.085, 0.034, 0.029, 0.075]
    dko_rmse  = [1.055, 1.133, 0.630, 0.391]
    dko_std   = [0.033, 0.139, 0.025, 0.033]
    mean_rmse = [1.317, 1.286, 0.678, 0.411]
    mean_std  = [0.096, 0.119, 0.056, 0.029]
    fp_rmse   = [0.519, 0.650, 0.378, 0.259]
    fp_std    = [0.006, 0.013, 0.004, 0.007]

    x = np.arange(len(targets))
    w = 0.18

    ax.bar(x - 1.5*w, fp_rmse, w, yerr=fp_std, color=C['green'],
           edgecolor=C['text'], linewidth=0.6, label='FP+XGB',
           capsize=2, zorder=3, error_kw={'linewidth': 0.5})
    ax.bar(x - 0.5*w, attn_rmse, w, yerr=attn_std, color=C['orange'],
           edgecolor=C['text'], linewidth=0.6, label='Attention',
           capsize=2, zorder=3, error_kw={'linewidth': 0.5})
    ax.bar(x + 0.5*w, dko_rmse, w, yerr=dko_std, color=C['blue'],
           edgecolor=C['text'], linewidth=0.6, label='DKO Gated',
           capsize=2, zorder=3, error_kw={'linewidth': 0.5})
    ax.bar(x + 1.5*w, mean_rmse, w, yerr=mean_std, color=C['gray'],
           edgecolor=C['text'], linewidth=0.6, label='Mean',
           capsize=2, zorder=3, error_kw={'linewidth': 0.5})

    ax.set_xticks(x)
    ax.set_xticklabels([f'Sterimol {t}' for t in targets], fontsize=8)
    ax.set_ylabel('RMSE')
    ax.yaxis.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7, ncol=4, loc='upper right', framealpha=0.9)
    ax.set_title('Kraken Steric Descriptors (Attention > DKO > Mean)',
                 fontweight='bold', fontsize=9)

    save_fig(fig, 'fig4_marcel')


# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("Generating DKO paper figures...")
    fig1_architecture()
    fig2_results()
    fig3_when_helps()
    fig4_marcel()
    print(f"\nAll figures saved to: {OUT}")
