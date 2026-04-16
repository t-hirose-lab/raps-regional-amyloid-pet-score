"""
Figure S6: APOE ε4 × RAPS Interaction
- RAPS quartiles (Q1-Q4) × APOE ε4 status (carrier vs non-carrier)
- Line plot with SEM error bars
- OLS interaction p-value annotation
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import stats
import statsmodels.formula.api as smf

# ── Data ────────────────────────────────────────────────────────────────────
df = pd.read_csv('/home/takumi/pj3-adni-roi/results/tables/v5_adni_predictions.csv')

# APOE ε4 carrier flag
df['apoe_carrier'] = (df['APOE_E4_COUNT'] > 0).astype(int)

# RAPS quartiles (based on continuous RAPS column)
df['raps_q'] = pd.qcut(df['RAPS'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

# ── OLS interaction model ────────────────────────────────────────────────────
model = smf.ols('y_actual ~ RAPS * apoe_carrier', data=df).fit()
# Interaction term p-value
interaction_pval = model.pvalues['RAPS:apoe_carrier']

# ── Group statistics ─────────────────────────────────────────────────────────
groups = df.groupby(['raps_q', 'apoe_carrier'], observed=True)['y_actual']
means = groups.mean().unstack('apoe_carrier')       # columns: 0, 1
sems  = groups.sem().unstack('apoe_carrier')

x_labels = ['Q1', 'Q2', 'Q3', 'Q4']
x_pos = np.arange(len(x_labels))

mean_neg = means[0].values   # APOE ε4−
mean_pos = means[1].values   # APOE ε4+
sem_neg  = sems[0].values
sem_pos  = sems[1].values

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))

# Lines + markers + error bars
ax.errorbar(x_pos, mean_pos, yerr=sem_pos,
            color='#D62728', marker='o', markersize=6,
            linewidth=1.5, capsize=4, capthick=1.2,
            label='APOE ε4+')
ax.errorbar(x_pos, mean_neg, yerr=sem_neg,
            color='#1F77B4', marker='s', markersize=6,
            linewidth=1.5, capsize=4, capthick=1.2,
            label='APOE ε4−')

# Axes
ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels, fontsize=7, fontfamily='Arial')
ax.set_xlabel('RAPS Quartile', fontsize=8, fontfamily='Arial')
ax.set_ylabel('Mean CDR-SB slope (/yr)', fontsize=8, fontfamily='Arial')
ax.tick_params(axis='both', labelsize=7)
for lbl in ax.get_yticklabels():
    lbl.set_fontfamily('Arial')

# Title
ax.set_title('APOE ε4 × RAPS Interaction', fontsize=10, fontfamily='Arial', pad=10)

# Spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend
legend = ax.legend(fontsize=7, frameon=False, prop={'family': 'Arial', 'size': 7})

# Interaction p-value annotation
if interaction_pval < 0.001:
    pval_str = f'Interaction P = {interaction_pval:.2e}'
elif interaction_pval < 0.05:
    pval_str = f'Interaction P = {interaction_pval:.3f}'
else:
    pval_str = f'Interaction P = {interaction_pval:.3f}'

ax.text(0.97, 0.97, pval_str,
        transform=ax.transAxes,
        ha='right', va='top',
        fontsize=7, fontfamily='Arial',
        color='#333333')

plt.tight_layout()

out_path = '/home/takumi/pj3-adni-roi/submit_v2/figures/supp_fig_s6_apoe_interaction.svg'
fig.savefig(out_path, format='svg', bbox_inches='tight')
plt.close()

print(f"Saved: {out_path}")
print(f"\nOLS interaction p-value: {interaction_pval:.4f}")
print(f"\nGroup means (CDR-SB slope):")
print(means.rename(columns={0: 'APOE_e4neg', 1: 'APOE_e4pos'}))
print(f"\nGroup n:")
print(df.groupby(['raps_q', 'apoe_carrier'], observed=True).size().unstack('apoe_carrier').rename(columns={0: 'APOE_e4neg', 1: 'APOE_e4pos'}))
