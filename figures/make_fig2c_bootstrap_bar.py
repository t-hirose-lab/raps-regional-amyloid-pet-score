"""
Figure 2c: Bootstrap ROI selection frequency bar chart (horizontal)
Top 25 ROIs by selection_freq_pct, 4-color scheme
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

matplotlib.rcParams['svg.fonttype'] = 'none'
# Use Nimbus Sans (Arial-equivalent) for rendering; SVG font name replaced to Arial in post-processing
matplotlib.rcParams['font.family'] = 'Nimbus Sans'

# ── Data ──────────────────────────────────────────────────────────────────────
df = pd.read_csv('/home/takumi/pj3-adni-roi/results/tables/bootstrap_roi_stability.csv')
top25 = df.nlargest(25, 'selection_freq_pct').copy()

# ── Color classification ──────────────────────────────────────────────────────
# Color 1: ElasticNet final ∩ bootstrap ≥80% (8 ROIs)
RAPS_ROIS = {
    'LEFT_HIPPOCAMPUS_SUVR',
    'CTX_LH_TRANSVERSETEMPORAL_SUVR',
    'CTX_LH_FUSIFORM_SUVR',
    'RIGHT_PALLIDUM_SUVR',
    'CTX_LH_LINGUAL_SUVR',
    'CTX_RH_ENTORHINAL_SUVR',
    'LEFT_AMYGDALA_SUVR',
    'LEFT_PUTAMEN_SUVR',
}

# Color 3: ElasticNet final but bootstrap <80% (3 ROIs)
ELASTICNET_ONLY_ROIS = {
    'CTX_LH_BANKSSTS_SUVR',
    'LEFT_ACCUMBENS_AREA_SUVR',
    'CTX_RH_PRECUNEUS_SUVR',
}

COLOR_RAPS           = '#C0392B'   # dark red  – RAPS (ElasticNet ∩ bootstrap ≥80%)
COLOR_STABLE_ONLY    = '#E67E22'   # orange    – bootstrap ≥80% only
COLOR_ELASTICNET_ONLY = '#8E44AD'  # purple    – ElasticNet only (bootstrap <80%)
COLOR_BELOW          = '#BDC3C7'   # gray      – below threshold

def assign_color(row):
    if row['roi'] in RAPS_ROIS:
        return COLOR_RAPS
    elif row['roi'] in ELASTICNET_ONLY_ROIS:
        return COLOR_ELASTICNET_ONLY
    elif row['selection_freq_pct'] >= 80.0:
        return COLOR_STABLE_ONLY
    else:
        return COLOR_BELOW

top25['color'] = top25.apply(assign_color, axis=1)

# ── ROI label cleanup ─────────────────────────────────────────────────────────
def clean_label(name):
    name = name.replace('_SUVR', '')
    name = name.replace('CTX_LH_', 'L ')
    name = name.replace('CTX_RH_', 'R ')
    name = name.replace('LEFT_',  'L ')
    name = name.replace('RIGHT_', 'R ')
    parts = name.split(' ', 1)
    if len(parts) == 2:
        side, region = parts
        region = region.replace('_', ' ').title()
        return f'{side} {region}'
    return name.replace('_', ' ').title()

top25['label'] = top25['roi'].apply(clean_label)

# Sort: highest freq at top -> ascending so barh puts highest at top
top25 = top25.sort_values('selection_freq_pct', ascending=True)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(4, 6))

y_pos = np.arange(len(top25))

ax.barh(
    y_pos,
    top25['selection_freq_pct'].values,
    color=top25['color'].values,
    edgecolor='none',
    height=0.72,
)

# 80% cutoff line
ax.axvline(80, color='black', linewidth=0.9, linestyle='--', zorder=3)

# y-axis tick labels
ax.set_yticks(y_pos)
ax.set_yticklabels(top25['label'].values, fontsize=7)

# Axis labels
ax.set_xlabel('Bootstrap selection frequency (%)', fontsize=8)
ax.set_xlim(0, 108)
ax.set_ylim(-0.6, len(top25) - 0.4)

# x-axis ticks
ax.set_xticks([0, 20, 40, 60, 80, 100])
ax.tick_params(axis='x', labelsize=7)

# Panel label "C" at top-left (10pt bold) + plot title
ax.set_title('Bootstrap Selection Frequency', fontsize=10, loc='center', pad=4)
ax.text(-0.10, 1.02, 'C', transform=ax.transAxes,
        fontsize=10, fontweight='bold', va='bottom', ha='left')

# Spines: remove top, right, and left
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='y', length=0)

# Legend (4 items)
legend_handles = [
    mpatches.Patch(color=COLOR_RAPS,            label='RAPS (8 ROIs)'),
    mpatches.Patch(color=COLOR_STABLE_ONLY,     label='Bootstrap stable only'),
    mpatches.Patch(color=COLOR_ELASTICNET_ONLY, label='ElasticNet only'),
    mpatches.Patch(color=COLOR_BELOW,           label='Below threshold'),
]
ax.legend(
    handles=legend_handles,
    fontsize=6,
    frameon=False,
    loc='lower right',
    handlelength=1.0,
    handleheight=0.8,
    borderpad=0.2,
)

# White background
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

plt.tight_layout(pad=0.5)

out_path = '/home/takumi/pj3-adni-roi/submit_v2/figures/fig2c_bootstrap_bar.svg'
plt.savefig(out_path, format='svg', bbox_inches='tight', dpi=150)
plt.close()

# Post-process SVG: replace Nimbus Sans with Arial so the SVG specifies Arial
with open(out_path, 'r', encoding='utf-8') as f:
    svg_text = f.read()
svg_text = svg_text.replace('Nimbus Sans', 'Arial')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(svg_text)
print(f'Saved: {out_path}')
