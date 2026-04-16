"""
Supplementary scatter plots for RAPS manuscript.
S1: RAPS (z-scored predicted) vs y_actual
S2: RAPS score (raps_raw) vs y_actual
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

# ---- style ----
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['font.family'] = 'Arial'

DATA = '/home/takumi/pj3-adni-roi/results/tables/v5_adni_predictions.csv'
OUT_S1 = '/home/takumi/pj3-adni-roi/submit_v2/figures/supp_fig_s1_predicted_vs_actual.svg'
OUT_S2 = '/home/takumi/pj3-adni-roi/submit_v2/figures/supp_fig_s2_raps_vs_slope.svg'

POINT_KW = dict(alpha=0.4, s=15, color='#3498DB', linewidths=0)
REG_KW   = dict(color='red', lw=1.5)

def make_scatter(x, y, xlabel, ylabel, title, outpath):
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(x, y, **POINT_KW)

    # regression line
    slope, intercept, r, p, _ = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 200)
    ax.plot(x_line, slope * x_line + intercept, **REG_KW)

    # annotation
    p_str = f'P = {p:.2e}' if p < 0.001 else f'P = {p:.3f}'
    annot = f'r = {r:.3f}, {p_str}'
    ax.text(0.05, 0.95, annot,
            transform=ax.transAxes,
            va='top', ha='left',
            fontsize=7, fontfamily='Arial')

    # spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # labels / title
    ax.set_xlabel(xlabel, fontsize=8, fontfamily='Arial')
    ax.set_ylabel(ylabel, fontsize=8, fontfamily='Arial')
    ax.set_title(title, fontsize=10, fontfamily='Arial')
    ax.tick_params(axis='both', labelsize=7)

    fig.tight_layout()
    fig.savefig(outpath, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {outpath}')


df = pd.read_csv(DATA)
print(f'N = {len(df)}')
print(df[['RAPS', 'raps_raw', 'y_actual']].describe())

# S1: RAPS (z-scored) vs y_actual
make_scatter(
    x=df['RAPS'],
    y=df['y_actual'],
    xlabel='RAPS (z-scored predicted)',
    ylabel='CDR-SB Slope (actual)',
    title='RAPS Predicted vs Actual CDR-SB Slope (N=515)',
    outpath=OUT_S1
)

# S2: raps_raw vs y_actual
make_scatter(
    x=df['raps_raw'],
    y=df['y_actual'],
    xlabel='RAPS Score',
    ylabel='CDR-SB Slope',
    title='RAPS Score vs CDR-SB Slope (N=515)',
    outpath=OUT_S2
)

print('Done.')
