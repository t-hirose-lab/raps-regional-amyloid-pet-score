"""
Figure 2A: ElasticNet ROI Coefficients (horizontal bar chart)
Publication-quality SVG for submit_v2
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
ROI_COEFS = {
    "LEFT_HIPPOCAMPUS":          -0.1395,
    "RIGHT_PALLIDUM":            -0.0356,
    "CTX_LH_LINGUAL":            +0.0365,
    "CTX_LH_TRANSVERSETEMPORAL": +0.0363,
    "CTX_LH_BANKSSTS":           +0.0347,
    "CTX_RH_ENTORHINAL":         -0.0285,
    "CTX_LH_FUSIFORM":           +0.0276,
    "LEFT_AMYGDALA":             +0.0255,
    "LEFT_PUTAMEN":              +0.0241,
    "LEFT_ACCUMBENS_AREA":       +0.0196,
    "CTX_RH_PRECUNEUS":          +0.0121,
}

RAPS_ROIS = {
    "LEFT_HIPPOCAMPUS", "CTX_LH_TRANSVERSETEMPORAL", "CTX_LH_FUSIFORM",
    "RIGHT_PALLIDUM", "CTX_LH_LINGUAL", "CTX_RH_ENTORHINAL",
    "LEFT_AMYGDALA", "LEFT_PUTAMEN",
}

# ── Color mapping ──────────────────────────────────────────────────────────────
COLOR_RAPS_NEG   = '#2980B9'   # dark blue
COLOR_RAPS_POS   = '#C0392B'   # dark red
COLOR_ELNET_NEG  = '#A9CCE3'   # light blue
COLOR_ELNET_POS  = '#E6A8A0'   # light red/pink

def get_color(roi, coef):
    if roi in RAPS_ROIS:
        return COLOR_RAPS_POS if coef >= 0 else COLOR_RAPS_NEG
    else:
        return COLOR_ELNET_POS if coef >= 0 else COLOR_ELNET_NEG

# ── Label cleaning ─────────────────────────────────────────────────────────────
def clean_label(roi):
    label = roi
    label = label.replace('CTX_LH_', 'L ')
    label = label.replace('CTX_RH_', 'R ')
    label = label.replace('LEFT_',   'L ')
    label = label.replace('RIGHT_',  'R ')
    label = label.replace('_SUVR', '')
    # title case for readability
    parts = label.split(' ', 1)
    if len(parts) == 2:
        side, name = parts
        name = name.replace('_', ' ').title()
        label = f"{side} {name}"
    return label

# ── Sort: most negative at bottom → most positive at top ─────────────────────
items = sorted(ROI_COEFS.items(), key=lambda x: x[1])  # ascending
rois   = [r for r, _ in items]
coefs  = [c for _, c in items]
labels = [clean_label(r) for r in rois]
colors = [get_color(r, c) for r, c in zip(rois, coefs)]

# ── Plot ───────────────────────────────────────────────────────────────────────
matplotlib.rcParams['svg.fonttype'] = 'none'
# Arial is not installed on this system; Nimbus Sans is the closest equivalent
# (Helvetica/Arial-compatible). SVG embeds font names so Word/Illustrator will
# substitute Arial automatically when the file is opened on Windows.
matplotlib.rcParams['font.family']  = 'Nimbus Sans'

fig, ax = plt.subplots(figsize=(4.5, 4.5))

y_pos = np.arange(len(rois))
bars = ax.barh(y_pos, coefs, color=colors, height=0.65, edgecolor='none')

# zero line
ax.axvline(0, color='black', linewidth=0.8, linestyle='--', zorder=3)

# β annotations next to each bar
for i, (coef, bar) in enumerate(zip(coefs, bars)):
    x_offset = 0.003 if coef >= 0 else -0.003
    ha = 'left' if coef >= 0 else 'right'
    ax.text(coef + x_offset, i, f'{coef:+.4f}',
            va='center', ha=ha, fontsize=7, color='#333333')

# y-axis tick labels
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=7)
ax.tick_params(axis='x', labelsize=7)
ax.tick_params(axis='y', length=0)   # hide tick marks on y

# axis label
ax.set_xlabel('ElasticNet Coefficient (β)', fontsize=8)

# x limits: leave room for annotations
x_min = min(coefs) - 0.025
x_max = max(coefs) + 0.030
ax.set_xlim(x_min, x_max)

# spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

# title
ax.set_title('ElasticNet ROI Coefficients', fontsize=10, pad=6)

# ── Legend ────────────────────────────────────────────────────────────────────
patch_raps_pos  = mpatches.Patch(color=COLOR_RAPS_POS,  label='RAPS – positive (8 ROIs)')
patch_raps_neg  = mpatches.Patch(color=COLOR_RAPS_NEG,  label='RAPS – negative (8 ROIs)')
patch_elnet_pos = mpatches.Patch(color=COLOR_ELNET_POS, label='ElasticNet only – positive')
patch_elnet_neg = mpatches.Patch(color=COLOR_ELNET_NEG, label='ElasticNet only – negative')

legend = ax.legend(
    handles=[patch_raps_pos, patch_raps_neg, patch_elnet_pos, patch_elnet_neg],
    fontsize=6,
    loc='lower right',
    frameon=True,
    framealpha=0.9,
    edgecolor='#cccccc',
    handlelength=1.2,
    handleheight=0.8,
)

# ── Panel label "A" ───────────────────────────────────────────────────────────
ax.text(-0.10, 1.04, 'A', transform=ax.transAxes,
        fontsize=10, fontweight='bold', va='top', ha='left')

plt.tight_layout()

out_path = '/home/takumi/pj3-adni-roi/submit_v2/figures/fig2a_elasticnet_coef.svg'
fig.savefig(out_path, format='svg', bbox_inches='tight', dpi=300)
print(f"Saved: {out_path}")
plt.close()
