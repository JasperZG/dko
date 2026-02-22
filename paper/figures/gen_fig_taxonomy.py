import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

plt.rcParams.update({
    'font.size': 13,
    'font.family': 'sans-serif',
})

fig, ax = plt.subplots(figsize=(11, 5.5))
ax.set_xlim(0, 11)
ax.set_ylim(0, 5.5)
ax.axis('off')
fig.patch.set_facecolor('#FFFFFF')

# Colours
GREEN  = '#27AE60'
YELLOW = '#F1C40F'
RED    = '#E74C3C'
LIGHT_GREEN  = '#D5F5E3'
LIGHT_YELLOW = '#FEF9E7'
LIGHT_RED    = '#FADBD8'
DARK_GRAY = '#2C3E50'
MED_GRAY  = '#7F8C8D'
LIGHT_GRAY = '#F2F3F4'
BOX_EDGE   = '#5D6D7E'
DARK_BLUE  = '#2471A3'
LIGHT_BLUE = '#D6EAF8'

# Title
ax.text(5.5, 5.15, 'Property Taxonomy: When Do Conformer Statistics Help?',
        ha='center', va='center', fontsize=15, fontweight='bold', color=DARK_BLUE)

# Column headers
header_y = 4.55
col_x = [0.3, 2.6, 5.8, 9.0]
headers = ['Category', 'Datasets', 'Finding', 'Verdict']
for x, h in zip(col_x, headers):
    ax.text(x + 0.05, header_y, h, ha='left', va='center',
            fontsize=12, fontweight='bold', color=DARK_GRAY)

ax.plot([0.2, 10.8], [4.3, 4.3], color=BOX_EDGE, lw=1.2)

# ---- Row 1: Solvation ----
row1_y = 3.65
row1_h = 1.0

# Category box
box1 = FancyBboxPatch((0.2, row1_y - row1_h/2), 2.2, row1_h,
       boxstyle='round,pad=0.1', facecolor=LIGHT_GREEN, edgecolor=GREEN, lw=1.3)
ax.add_patch(box1)
ax.text(1.3, row1_y + 0.15, 'Solvation', ha='center', fontsize=12,
        fontweight='bold', color=GREEN)
ax.text(1.3, row1_y - 0.15, 'Properties', ha='center', fontsize=10, color=MED_GRAY)

# Datasets
ax.text(2.65, row1_y + 0.12, 'ESOL', ha='left', fontsize=11, color=DARK_GRAY, fontweight='bold')
ax.text(2.65, row1_y - 0.18, 'FreeSolv', ha='left', fontsize=11, color=DARK_GRAY, fontweight='bold')

# Finding
ax.text(5.85, row1_y + 0.12, 'Hybrid improves', ha='left', fontsize=11, color=DARK_GRAY)
ax.text(5.85, row1_y - 0.18, 'RMSE by 4\u201310%', ha='left', fontsize=11,
        fontweight='bold', color=GREEN)

# Verdict - green checkmark
circle1 = plt.Circle((9.55, row1_y), 0.3, fc=LIGHT_GREEN, ec=GREEN, lw=1.5, zorder=3)
ax.add_patch(circle1)
ax.text(9.55, row1_y, '\u2713', ha='center', va='center', fontsize=20,
        fontweight='bold', color=GREEN, zorder=4)

# ---- Row 2: Steric/Boltzmann ----
row2_y = 2.4
ax.plot([0.2, 10.8], [row2_y + row1_h/2 + 0.15, row2_y + row1_h/2 + 0.15],
        color='#D5D8DC', lw=0.8)

box2 = FancyBboxPatch((0.2, row2_y - row1_h/2), 2.2, row1_h,
       boxstyle='round,pad=0.1', facecolor=LIGHT_YELLOW, edgecolor=YELLOW, lw=1.3)
ax.add_patch(box2)
ax.text(1.3, row2_y + 0.15, 'Steric /', ha='center', fontsize=12,
        fontweight='bold', color='#B7950B')
ax.text(1.3, row2_y - 0.15, 'Boltzmann', ha='center', fontsize=10, color=MED_GRAY)

# Datasets
ax.text(2.65, row2_y, 'Kraken', ha='left', fontsize=11, color=DARK_GRAY, fontweight='bold')

# Finding
ax.text(5.85, row2_y + 0.12, 'Attention > DKO > Mean', ha='left', fontsize=11, color=DARK_GRAY)
ax.text(5.85, row2_y - 0.18, '(partial benefit)', ha='left', fontsize=10,
        color='#B7950B', style='italic')

# Verdict - yellow partial
circle2 = plt.Circle((9.55, row2_y), 0.3, fc=LIGHT_YELLOW, ec=YELLOW, lw=1.5, zorder=3)
ax.add_patch(circle2)
ax.text(9.55, row2_y, '~', ha='center', va='center', fontsize=22,
        fontweight='bold', color='#B7950B', zorder=4)

# ---- Row 3: Electronic ----
row3_y = 1.15
ax.plot([0.2, 10.8], [row3_y + row1_h/2 + 0.15, row3_y + row1_h/2 + 0.15],
        color='#D5D8DC', lw=0.8)

box3 = FancyBboxPatch((0.2, row3_y - row1_h/2), 2.2, row1_h,
       boxstyle='round,pad=0.1', facecolor=LIGHT_RED, edgecolor=RED, lw=1.3)
ax.add_patch(box3)
ax.text(1.3, row3_y + 0.15, 'Electronic', ha='center', fontsize=12,
        fontweight='bold', color=RED)
ax.text(1.3, row3_y - 0.15, 'Properties', ha='center', fontsize=10, color=MED_GRAY)

# Datasets
ax.text(2.65, row3_y + 0.22, 'QM9-Gap', ha='left', fontsize=11, color=DARK_GRAY, fontweight='bold')
ax.text(2.65, row3_y - 0.02, 'BDE', ha='left', fontsize=11, color=DARK_GRAY, fontweight='bold')
ax.text(2.65, row3_y - 0.26, 'Drugs-75K', ha='left', fontsize=11, color=DARK_GRAY, fontweight='bold')

# Finding
ax.text(5.85, row3_y + 0.12, 'No improvement from', ha='left', fontsize=11, color=DARK_GRAY)
ax.text(5.85, row3_y - 0.18, 'conformer statistics', ha='left', fontsize=11,
        fontweight='bold', color=RED)

# Verdict - red X
circle3 = plt.Circle((9.55, row3_y), 0.3, fc=LIGHT_RED, ec=RED, lw=1.5, zorder=3)
ax.add_patch(circle3)
ax.text(9.55, row3_y, '\u2717', ha='center', va='center', fontsize=20,
        fontweight='bold', color=RED, zorder=4)

# Bottom annotation
ax.text(5.5, 0.25,
        'Conformer ensemble statistics are most useful for properties that depend on '
        'molecular shape in solution.',
        ha='center', fontsize=10, color=MED_GRAY, style='italic')

plt.tight_layout(pad=0.5)
outdir = os.path.join('C:', os.sep, 'Users', 'zhaoz', 'Downloads', 'dko', 'paper', 'figures')
fig.savefig(os.path.join(outdir, 'fig_taxonomy.pdf'),
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved fig_taxonomy.pdf')
