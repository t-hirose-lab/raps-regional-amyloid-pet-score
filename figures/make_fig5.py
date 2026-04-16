"""
Figure 5: CL 20-60 subgroup analysis in ADNI
4-panel figure: ROC curves, boxplot, scatter, partial correlations
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['svg.fonttype'] = 'none'

# Register Arial from Windows fonts if available
import matplotlib.font_manager as fm
_arial_path = '/mnt/c/Windows/Fonts/arial.ttf'
try:
    fm.fontManager.addfont(_arial_path)
    matplotlib.rcParams['font.family'] = 'Arial'
    print("Arial font registered successfully.")
except Exception as e:
    print(f"Arial not available ({e}), falling back to DejaVu Sans")
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy import stats
from sklearn.metrics import roc_curve, auc

# ─── Load data ───────────────────────────────────────────────────────────────
df_all = pd.read_csv('/home/takumi/pj3-adni-roi/results/tables/v5_adni_predictions.csv')
print(f"Total N = {len(df_all)}")

# Overall RAPS median (N=515)
raps_median_overall = df_all['RAPS'].median()
print(f"Overall RAPS median (N={len(df_all)}) = {raps_median_overall:.4f}")

# CL 20-60 subgroup
df_sub = df_all[(df_all['CENTILOIDS'] >= 20) & (df_all['CENTILOIDS'] < 60)].copy()
print(f"CL 20-60 subgroup N = {len(df_sub)}")

# Median split using OVERALL median
df_sub['RAPS_group'] = np.where(df_sub['RAPS'] >= raps_median_overall, 'High', 'Low')
print(df_sub['RAPS_group'].value_counts())

# ─── Key statistics ───────────────────────────────────────────────────────────
cohens_d  = 1.245
partial_r_raps = 0.5076
partial_r_cl   = 0.1593

# ─── Colors ──────────────────────────────────────────────────────────────────
COL_RAPS = '#C0392B'
COL_CL   = '#3498DB'

# ─── Figure layout ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(8, 7))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.38)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 0])
ax_d = fig.add_subplot(gs[1, 1])

def despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def panel_label(ax, label, fontsize=10):
    ax.text(-0.18, 1.05, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight='bold', va='top', ha='left',
            fontfamily='Arial')

def panel_title(ax, title, fontsize=10):
    ax.text(0.5, 1.05, title, transform=ax.transAxes,
            fontsize=fontsize, va='top', ha='center',
            fontfamily='Arial')

# ─── Panel A: ROC curves ──────────────────────────────────────────────────────
thresholds = [1.0, 1.5]
linestyles = {1.0: '-', 1.5: '--'}

for thresh in thresholds:
    y_true = (df_sub['y_actual'] > thresh).astype(int)
    if y_true.sum() == 0:
        print(f"WARNING: No positive cases for threshold {thresh}")
        continue

    # RAPS
    fpr_r, tpr_r, _ = roc_curve(y_true, df_sub['RAPS'])
    auc_r = auc(fpr_r, tpr_r)
    ax_a.plot(fpr_r, tpr_r,
              color=COL_RAPS, lw=1.5, ls=linestyles[thresh],
              label=f'RAPS >{thresh:.1f} (AUC={auc_r:.2f})')

    # CL
    fpr_c, tpr_c, _ = roc_curve(y_true, df_sub['CENTILOIDS'])
    auc_c = auc(fpr_c, tpr_c)
    ax_a.plot(fpr_c, tpr_c,
              color=COL_CL, lw=1.5, ls=linestyles[thresh],
              label=f'CL >{thresh:.1f} (AUC={auc_c:.2f})')

ax_a.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.5)
ax_a.set_xlabel('1 – Specificity', fontsize=8, fontfamily='Arial')
ax_a.set_ylabel('Sensitivity', fontsize=8, fontfamily='Arial')
ax_a.tick_params(labelsize=7)
ax_a.legend(fontsize=7, loc='lower right', frameon=False)
ax_a.set_xlim(0, 1)
ax_a.set_ylim(0, 1)
despine(ax_a)
panel_label(ax_a, 'A')
panel_title(ax_a, 'ROC: CL 20\u201360 Subgroup')

# ─── Panel B: Boxplot + strip plot ────────────────────────────────────────────
groups = ['High', 'Low']
colors_b = {'Low': COL_CL, 'High': COL_RAPS}

rng = np.random.default_rng(42)

for i, grp in enumerate(groups):
    d = df_sub.loc[df_sub['RAPS_group'] == grp, 'y_actual'].values
    col = colors_b[grp]

    # Box
    bp = ax_b.boxplot(d, positions=[i], widths=0.35,
                      patch_artist=True,
                      medianprops=dict(color='black', lw=1.5),
                      boxprops=dict(facecolor=col, alpha=0.45, linewidth=0),
                      whiskerprops=dict(color=col, lw=1.2),
                      capprops=dict(color=col, lw=1.2),
                      flierprops=dict(marker='', linestyle='none'),
                      showfliers=False)

    # Strip
    x_jitter = rng.uniform(-0.12, 0.12, size=len(d))
    ax_b.scatter(i + x_jitter, d, color=col, s=8, alpha=0.55, zorder=3, linewidths=0)

# Statistics
high_vals = df_sub.loc[df_sub['RAPS_group'] == 'High', 'y_actual'].values
low_vals  = df_sub.loc[df_sub['RAPS_group'] == 'Low',  'y_actual'].values
t_stat, p_val = stats.ttest_ind(high_vals, low_vals)
p_text = f'p = {p_val:.2e}' if p_val < 0.001 else f'p = {p_val:.3f}'

# Annotation
y_max = df_sub['y_actual'].max()
y_pad = (y_max - df_sub['y_actual'].min()) * 0.12
ax_b.annotate('', xy=(1, y_max + y_pad*0.5), xytext=(0, y_max + y_pad*0.5),
              arrowprops=dict(arrowstyle='-', color='black', lw=1.0))
ax_b.text(0.5, y_max + y_pad*0.7,
          f"Cohen's d = {cohens_d:.3f}\n{p_text}",
          ha='center', va='bottom', fontsize=7, fontfamily='Arial')

ax_b.set_xticks([0, 1])
ax_b.set_xticklabels(['RAPS-High', 'RAPS-Low'], fontsize=7)
ax_b.set_ylabel('CDR-SB slope (/yr)', fontsize=8, fontfamily='Arial')
ax_b.tick_params(labelsize=7)
despine(ax_b)
panel_label(ax_b, 'B')
panel_title(ax_b, 'RAPS Stratification: CL 20\u201360')

# ─── Panel C: Scatter plot ────────────────────────────────────────────────────
norm_c = Normalize(vmin=df_sub['y_actual'].min(), vmax=df_sub['y_actual'].max())
sc = ax_c.scatter(df_sub['CENTILOIDS'], df_sub['RAPS'],
                  c=df_sub['y_actual'], cmap='coolwarm',
                  norm=norm_c, s=20, alpha=0.75, linewidths=0, zorder=3)

# Horizontal dashed line at overall RAPS median
ax_c.axhline(raps_median_overall, color='black', lw=1.0, ls='--', alpha=0.7,
             label=f'RAPS median = {raps_median_overall:.2f}')

ax_c.set_xlabel('Centiloids', fontsize=8, fontfamily='Arial')
ax_c.set_ylabel('RAPS', fontsize=8, fontfamily='Arial')
ax_c.tick_params(labelsize=7)
ax_c.legend(fontsize=7, loc='upper right', frameon=False)

# Colorbar
cbar = fig.colorbar(sc, ax=ax_c, pad=0.02, aspect=20, shrink=0.85)
cbar.set_label('CDR-SB slope (/yr)', fontsize=7, fontfamily='Arial')
cbar.ax.tick_params(labelsize=6)

despine(ax_c)
panel_label(ax_c, 'C')
panel_title(ax_c, 'RAPS vs Centiloid: CL 20\u201360')

# ─── Panel D: Partial correlations bar chart ──────────────────────────────────
bar_labels = ['RAPS | CL', 'CL | RAPS']
bar_vals   = [partial_r_raps, partial_r_cl]
bar_colors = [COL_RAPS, COL_CL]

bars = ax_d.bar(bar_labels, bar_vals, color=bar_colors, width=0.45, alpha=0.80)

# Value labels on bars
for bar, val in zip(bars, bar_vals):
    ax_d.text(bar.get_x() + bar.get_width() / 2,
              val + 0.01,
              f'r = {val:.4f}',
              ha='center', va='bottom', fontsize=7, fontfamily='Arial')

ax_d.set_ylabel('Partial correlation (r)', fontsize=8, fontfamily='Arial')
ax_d.set_ylim(0, max(bar_vals) * 1.35)
ax_d.tick_params(labelsize=7)
ax_d.text(0.5, 0.92, 'Controlling for the other predictor',
          transform=ax_d.transAxes, ha='center', va='top',
          fontsize=7, fontfamily='Arial', color='#444444')
despine(ax_d)
panel_label(ax_d, 'D')
panel_title(ax_d, 'Partial Correlations: CL 20\u201360')

# ─── Save ─────────────────────────────────────────────────────────────────────
out_path = '/home/takumi/pj3-adni-roi/submit_v2/figures/fig5_cl2060.svg'
plt.savefig(out_path, format='svg', bbox_inches='tight', dpi=150)
print(f"Saved: {out_path}")
plt.close()
