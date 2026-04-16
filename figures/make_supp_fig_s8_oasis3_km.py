"""
Supplementary Figure S8: OASIS-3 Kaplan-Meier Survival Curves
RAPS vs Centiloid stratification (8-ROI RAPS)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['font.family'] = 'Arial'

# ---- Load data ----
df = pd.read_csv('/home/takumi/pj3-adni-roi/results/tables/v5_oasis3_predictions.csv')
print(f"Loaded N={len(df)} rows")
print(df[['RAPS', 'CDRSB_slope', 'CENTILOIDS']].describe())

# ---- Compute time-to-event ----
# time = min(1/slope, 4.0) if slope > 0, else 4.0; clip min at 0.1
def compute_time(slope):
    if slope > 0:
        return max(min(1.0 / slope, 4.0), 0.1)
    else:
        return 4.0

df['time'] = df['CDRSB_slope'].apply(compute_time)
df['event'] = (df['CDRSB_slope'] > 0).astype(int)

# Censor at time >= 4.0 (set event=0)
df.loc[df['time'] >= 4.0, 'event'] = 0

print(f"\nEvent counts: {df['event'].sum()} events out of {len(df)}")
print(f"Time range: {df['time'].min():.3f} - {df['time'].max():.3f}")

# ---- Stratify by RAPS median ----
raps_median = df['RAPS'].median()
cl_median = df['CENTILOIDS'].median()
print(f"\nRAPS median: {raps_median:.4f}")
print(f"CL median: {cl_median:.4f}")

df['RAPS_group'] = (df['RAPS'] >= raps_median).map({True: 'High', False: 'Low'})
df['CL_group'] = (df['CENTILOIDS'] >= cl_median).map({True: 'High', False: 'Low'})

print(f"\nRAPS group counts: {df['RAPS_group'].value_counts().to_dict()}")
print(f"CL group counts: {df['CL_group'].value_counts().to_dict()}")

# ---- Plot ----
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

colors = {'High': '#d62728', 'Low': '#1f77b4'}  # red / blue

for ax, group_col, title_label in zip(
    axes,
    ['RAPS_group', 'CL_group'],
    ['RAPS', 'Centiloid']
):
    for group_label in ['High', 'Low']:
        mask = df[group_col] == group_label
        t = df.loc[mask, 'time']
        e = df.loc[mask, 'event']
        n = mask.sum()

        kmf = KaplanMeierFitter()
        kmf.fit(t, event_observed=e, label=f'{group_label} (n={n})')
        kmf.plot_survival_function(
            ax=ax,
            ci_show=True,
            color=colors[group_label],
            linewidth=1.5,
            show_censors=True,
            censor_styles={'ms': 5, 'marker': '+'}
        )

    # Log-rank test
    mask_high = df[group_col] == 'High'
    mask_low  = df[group_col] == 'Low'
    result = logrank_test(
        df.loc[mask_high, 'time'], df.loc[mask_low, 'time'],
        event_observed_A=df.loc[mask_high, 'event'],
        event_observed_B=df.loc[mask_low, 'event']
    )
    p_val = result.p_value
    p_str = f'p = {p_val:.3f}' if p_val >= 0.001 else f'p < 0.001'

    ax.text(0.05, 0.12, f'Log-rank {p_str}',
            transform=ax.transAxes, fontsize=7,
            fontfamily='Arial', va='bottom')

    ax.set_xlim(0, 3.9)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Time (years)', fontsize=8, fontfamily='Arial')
    ax.set_ylabel('Survival probability', fontsize=8, fontfamily='Arial')
    ax.set_title(f'{title_label} Stratification', fontsize=9, fontfamily='Arial')
    ax.tick_params(labelsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=7, frameon=False)

fig.suptitle('OASIS-3 Kaplan-Meier: RAPS vs Centiloid Stratification',
             fontsize=10, fontfamily='Arial', y=1.01)

plt.tight_layout()

out_path = '/home/takumi/pj3-adni-roi/submit_v2/figures/supp_fig_s8_oasis3_km.svg'
fig.savefig(out_path, format='svg', bbox_inches='tight', dpi=150)
print(f"\nSaved: {out_path}")
plt.close()
