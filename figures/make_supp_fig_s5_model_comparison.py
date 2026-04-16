import matplotlib
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['font.family'] = 'Nimbus Sans'  # Arial-compatible on Linux

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── data ──────────────────────────────────────────────────────────────────────
df = pd.read_csv('/home/takumi/pj3-adni-roi/results/tables/v5_supp/s10_model_comparison.csv')

roi8  = df[df['model'] == '8-ROI'].iloc[0]
roi80 = df[df['model'] == '80→11-ROI'].iloc[0]

# metrics to plot
metrics       = ['CV R²', 'CV r', 'AUC >1.0', 'AUC >1.5', 'AUC >2.0']
keys_mean     = ['cv_r2_mean', 'cv_r_mean', 'auc_1.0', 'auc_1.5', 'auc_2.0']
keys_sd       = ['cv_r2_sd',   'cv_r_sd',   None,      None,      None]

vals_8roi  = [roi8[k]  for k in keys_mean]
vals_80roi = [roi80[k] for k in keys_mean]
errs_8roi  = [roi8[k]  if k else 0 for k in keys_sd]
errs_80roi = [roi80[k] if k else 0 for k in keys_sd]

# ── layout ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

n = len(metrics)
x = np.arange(n)
w = 0.32
gap = 0.04

color_8roi  = '#C0392B'
color_80roi = '#95A5A6'

# draw bars
bars1 = ax.bar(x - w/2 - gap/2, vals_8roi,  width=w,
               color=color_8roi,  alpha=0.88, zorder=3,
               yerr=errs_8roi,  error_kw=dict(elinewidth=0.8, capsize=3,
                                               capthick=0.8, ecolor='#7B241C'))
bars2 = ax.bar(x + w/2 + gap/2, vals_80roi, width=w,
               color=color_80roi, alpha=0.88, zorder=3,
               yerr=errs_80roi, error_kw=dict(elinewidth=0.8, capsize=3,
                                               capthick=0.8, ecolor='#717D7E'))

# value labels
for bar, val, err in zip(bars1, vals_8roi, errs_8roi):
    top = bar.get_height() + (err if err else 0)
    ax.text(bar.get_x() + bar.get_width()/2, top + 0.008,
            f'{val:.3f}', ha='center', va='bottom',
            fontsize=7, fontfamily='Nimbus Sans', color='#2C3E50')

for bar, val, err in zip(bars2, vals_80roi, errs_80roi):
    top = bar.get_height() + (err if err else 0)
    ax.text(bar.get_x() + bar.get_width()/2, top + 0.008,
            f'{val:.3f}', ha='center', va='bottom',
            fontsize=7, fontfamily='Nimbus Sans', color='#2C3E50')

# ── axes style ────────────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=7, fontfamily='Nimbus Sans')
ax.set_ylabel('Value', fontsize=8, fontfamily='Nimbus Sans')
ax.set_xlabel('Metric', fontsize=8, fontfamily='Nimbus Sans')
ax.set_title('8-ROI RAPS vs Full 80-ROI Model Comparison',
             fontsize=10, fontfamily='Nimbus Sans', pad=10)

ax.tick_params(axis='both', labelsize=7)
ax.set_ylim(0, 1.10)
ax.yaxis.set_tick_params(labelsize=7)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.8)
ax.spines['bottom'].set_linewidth(0.8)
ax.tick_params(width=0.8)

ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle='--', linewidth=0.4, color='#CCCCCC', alpha=0.7)

# legend
patch1 = mpatches.Patch(color=color_8roi,  alpha=0.88, label='8-ROI RAPS')
patch2 = mpatches.Patch(color=color_80roi, alpha=0.88, label='Full 80-ROI Model')
ax.legend(handles=[patch1, patch2], fontsize=7, frameon=False,
          loc='upper right', handlelength=1.2, handleheight=0.9)

# ── save ──────────────────────────────────────────────────────────────────────
out = '/home/takumi/pj3-adni-roi/submit_v2/figures/supp_fig_s5_model_comparison.svg'
fig.tight_layout()
fig.savefig(out, format='svg', dpi=150, bbox_inches='tight')
print(f'Saved: {out}')
plt.close(fig)
