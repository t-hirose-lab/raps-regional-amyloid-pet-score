"""
Figure 6: OASIS-3 External Validation
4-panel (2x2):
  A: ROC curve CDR-SB slope > 1.0/yr
  B: ROC curve CDR-SB slope > 1.5/yr
  C: CL 20-60 boxplot + strip (split by RAPS median)
  D: Scatter CENTILOIDS vs RAPS in CL 20-60, colored by CDR-SB slope
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, roc_auc_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv('/home/takumi/pj3-adni-roi/results/tables/v5_oasis3_predictions.csv')
print(f"Total N = {len(df)}")
print(f"CENTILOIDS missing = {df['CENTILOIDS'].isna().sum()}")

# ── Palette ─────────────────────────────────────────────────────────────────
COL_RAPS = '#C0392B'
COL_CL   = '#3498DB'
COL_DIAG = '#7F8C8D'

# ── Key derived values ───────────────────────────────────────────────────────
# RAPS median from full cohort (all 429) — simulates prospective use
raps_median_all = df['RAPS'].median()
print(f"RAPS median (all N=429): {raps_median_all:.4f}")

# CL 20-60 subgroup
cl2060 = df[(df['CENTILOIDS'] >= 20) & (df['CENTILOIDS'] < 60)].copy()
print(f"CL 20-60 subgroup n = {len(cl2060)}")

# Split CL 20-60 by full-cohort RAPS median
cl2060['RAPS_group'] = np.where(cl2060['RAPS'] >= raps_median_all, 'High', 'Low')
high_slope = cl2060[cl2060['RAPS_group'] == 'High']['CDRSB_slope'].values
low_slope  = cl2060[cl2060['RAPS_group'] == 'Low']['CDRSB_slope'].values
print(f"  High RAPS n={len(high_slope)}, Low RAPS n={len(low_slope)}")

# Cohen's d
t_stat, p_val_box = stats.ttest_ind(high_slope, low_slope)
pooled_std = np.sqrt(
    ((len(high_slope) - 1) * high_slope.std() ** 2 +
     (len(low_slope)  - 1) * low_slope.std()  ** 2) /
    (len(high_slope) + len(low_slope) - 2)
)
cohens_d = (high_slope.mean() - low_slope.mean()) / pooled_std
print(f"  Cohen's d = {cohens_d:.3f}, p = {p_val_box:.4e}")

# ── Figure setup ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(8, 7))
gs  = gridspec.GridSpec(2, 2, figure=fig,
                        hspace=0.45, wspace=0.42,
                        left=0.09, right=0.96,
                        top=0.94, bottom=0.09)
axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]
panel_labels = ['A', 'B', 'C', 'D']

# ── Helper: common spine/tick style ─────────────────────────────────────────
def style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=7)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(width=0.8, length=3)

# ── Panels A & B: ROC curves ─────────────────────────────────────────────────
roc_specs = [
    (1.0, 'A', 'ROC: CDR-SB slope >1.0/yr'),
    (1.5, 'B', 'ROC: CDR-SB slope >1.5/yr'),
]

# For CL ROC: drop missing CL
df_cl = df.dropna(subset=['CENTILOIDS'])
print(f"n for CL ROC analysis = {len(df_cl)}")

for idx, (thr, lbl, title) in enumerate(roc_specs):
    ax = axes[idx]

    # RAPS (all 429)
    y_bin_raps = (df['CDRSB_slope'] > thr).astype(int)
    fpr_r, tpr_r, _ = roc_curve(y_bin_raps, df['RAPS'])
    auc_r = roc_auc_score(y_bin_raps, df['RAPS'])

    # CL (n=428, drop NaN)
    y_bin_cl = (df_cl['CDRSB_slope'] > thr).astype(int)
    fpr_c, tpr_c, _ = roc_curve(y_bin_cl, df_cl['CENTILOIDS'])
    auc_c = roc_auc_score(y_bin_cl, df_cl['CENTILOIDS'])

    ax.plot(fpr_r, tpr_r, color=COL_RAPS, lw=1.4,
            label=f'RAPS (AUC = {auc_r:.3f})')
    ax.plot(fpr_c, tpr_c, color=COL_CL,   lw=1.4,
            label=f'Centiloid (AUC = {auc_c:.3f})')
    ax.plot([0, 1], [0, 1], color=COL_DIAG, lw=0.8, ls='--', zorder=0)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('1 – Specificity', fontsize=8, fontname='Arial')
    ax.set_ylabel('Sensitivity',     fontsize=8, fontname='Arial')
    ax.set_title(title, fontsize=10, fontname='Arial', pad=4, loc='center')

    ax.legend(fontsize=7, frameon=False,
              loc='lower right',
              prop={'family': 'Arial', 'size': 7})

    style_ax(ax)

    ax.text(-0.18, 1.05, lbl, transform=ax.transAxes,
            fontsize=10, fontweight='bold', fontname='Arial',
            va='top', ha='left')

    print(f"Panel {lbl}: threshold={thr}, RAPS AUC={auc_r:.4f}, CL AUC={auc_c:.4f}, "
          f"pos={y_bin_raps.sum()}")

# ── Panel C: Boxplot + strip for CL 20-60 ────────────────────────────────────
ax = axes[2]

rng = np.random.default_rng(42)

groups   = ['RAPS-High', 'RAPS-Low']
data_arr = [high_slope, low_slope]
colors   = [COL_RAPS, COL_CL]
positions = [0, 1]

for pos, data, color in zip(positions, data_arr, colors):
    # Box
    bp = ax.boxplot(data, positions=[pos], widths=0.35,
                    patch_artist=True,
                    medianprops=dict(color='black', linewidth=1.5),
                    boxprops=dict(facecolor=color, alpha=0.45, linewidth=0.8),
                    whiskerprops=dict(color=color, linewidth=0.8),
                    capprops=dict(color=color, linewidth=0.8),
                    flierprops=dict(marker='', markersize=0),
                    showfliers=False)

    # Jitter strip
    jitter = rng.uniform(-0.12, 0.12, size=len(data))
    ax.scatter(pos + jitter, data,
               color=color, s=10, alpha=0.55,
               linewidths=0, zorder=3)

ax.set_xticks([0, 1])
ax.set_xticklabels(groups, fontsize=7, fontname='Arial')
ax.set_ylabel('CDR-SB slope (/yr)', fontsize=8, fontname='Arial')
ax.set_title('RAPS Stratification: CL 20\u201360', fontsize=10, fontname='Arial', pad=4, loc='center')

# Significance bracket
y_max = max(np.concatenate([high_slope, low_slope])) * 1.05
bracket_h = y_max * 0.04
ax.plot([0, 0, 1, 1],
        [y_max, y_max + bracket_h, y_max + bracket_h, y_max],
        color='black', lw=0.8)

# p-value string
if p_val_box < 0.001:
    p_str = f'p < 0.001'
elif p_val_box < 0.01:
    p_str = f'p = {p_val_box:.3f}'
else:
    p_str = f'p = {p_val_box:.2f}'

annot_y = y_max + bracket_h * 1.5
ax.text(0.5, annot_y,
        f"d = {cohens_d:.3f}, {p_str}",
        ha='center', va='bottom',
        fontsize=7, fontname='Arial')

# Extend y-limit to fit annotation
ax.set_ylim(bottom=min(np.concatenate([high_slope, low_slope])) - 0.05,
            top=annot_y + bracket_h * 2)

style_ax(ax)

ax.text(-0.18, 1.05, 'C', transform=ax.transAxes,
        fontsize=10, fontweight='bold', fontname='Arial',
        va='top', ha='left')

# ── Panel D: Scatter CENTILOIDS vs RAPS in CL 20-60 ──────────────────────────
ax = axes[3]

vmin_val = cl2060['CDRSB_slope'].quantile(0.01)
vmax_val = cl2060['CDRSB_slope'].quantile(0.99)

sc = ax.scatter(cl2060['CENTILOIDS'], cl2060['RAPS'],
                c=cl2060['CDRSB_slope'], cmap='coolwarm',
                s=18, alpha=0.80, linewidths=0,
                vmin=vmin_val, vmax=vmax_val)

# Horizontal dashed line at RAPS median (full cohort)
ax.axhline(raps_median_all, color='black', lw=0.9, ls='--', zorder=5,
           label=f'RAPS median')

# Colorbar
cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('CDR-SB slope (/yr)', fontsize=7, fontname='Arial')
cbar.ax.tick_params(labelsize=7)
for label in cbar.ax.get_yticklabels():
    label.set_fontname('Arial')

ax.set_xlabel('Centiloid', fontsize=8, fontname='Arial')
ax.set_ylabel('RAPS',      fontsize=8, fontname='Arial')
ax.set_title('RAPS vs Centiloid: CL 20\u201360', fontsize=10, fontname='Arial', pad=4, loc='center')

# Median line label
ax.text(0.98, raps_median_all + 0.04,
        'Median',
        ha='right', va='bottom',
        fontsize=7, fontname='Arial',
        color='black',
        transform=ax.get_yaxis_transform())

style_ax(ax)

ax.text(-0.18, 1.05, 'D', transform=ax.transAxes,
        fontsize=10, fontweight='bold', fontname='Arial',
        va='top', ha='left')

# ── Save ──────────────────────────────────────────────────────────────────────
import os
out_dir = '/home/takumi/pj3-adni-roi/submit_v2/figures'
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'fig6_oasis3.svg')

fig.savefig(out_path, format='svg', dpi=300, bbox_inches='tight')
print(f"\nSaved: {out_path}")
plt.close(fig)
