"""
Figure 3: ROC curves (panels A-C) and scatter plot (panel D)
for ADNI CDR-SB slope prediction.
Layout: 3 columns on top (ROC A/B/C) + 1 wide panel on bottom (scatter D).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['svg.fonttype'] = 'none'
# Arial → Nimbus Sans (metrically compatible) → sans-serif fallback
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Nimbus Sans', 'Liberation Sans',
                                           'DejaVu Sans', 'sans-serif']
import warnings
import logging
warnings.filterwarnings('ignore', category=UserWarning, message='.*findfont.*')
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, roc_auc_score
from scipy import stats

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv('/home/takumi/pj3-adni-roi/results/tables/v5_adni_predictions.csv')

# ── Palette ─────────────────────────────────────────────────────────────────
COL_RAPS = '#C0392B'
COL_CL   = '#3498DB'
COL_DIAG = '#7F8C8D'

# ── Figure setup ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(10, 7))
gs  = gridspec.GridSpec(2, 3, figure=fig,
                        hspace=0.48, wspace=0.38,
                        left=0.07, right=0.97,
                        top=0.93, bottom=0.09)

# Row 0: panels A, B, C (one per column)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[0, 2])

# Row 1: panel D spanning all 3 columns
ax_d = fig.add_subplot(gs[1, :])

# ── Helper: apply common spine / tick style ──────────────────────────────────
def style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=7)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(width=0.8, length=3)

# ── Panels A–C: ROC curves ───────────────────────────────────────────────────
thresholds_info = [
    (1.0, 'A', 'ROC: CDR-SB slope >1.0/yr', ax_a),
    (1.5, 'B', 'ROC: CDR-SB slope >1.5/yr', ax_b),
    (2.0, 'C', 'ROC: CDR-SB slope >2.0/yr', ax_c),
]

for thr, lbl, panel_title, ax in thresholds_info:
    y_bin = (df['y_actual'] >= thr).astype(int)

    # RAPS
    fpr_r, tpr_r, _ = roc_curve(y_bin, df['RAPS'])
    auc_r = roc_auc_score(y_bin, df['RAPS'])

    # Centiloid
    fpr_c, tpr_c, _ = roc_curve(y_bin, df['CENTILOIDS'])
    auc_c = roc_auc_score(y_bin, df['CENTILOIDS'])

    ax.plot(fpr_r, tpr_r, color=COL_RAPS, lw=1.4,
            label=f'RAPS (AUC = {auc_r:.3f})')
    ax.plot(fpr_c, tpr_c, color=COL_CL,   lw=1.4,
            label=f'Centiloid (AUC = {auc_c:.3f})')
    ax.plot([0, 1], [0, 1], color=COL_DIAG, lw=0.8, ls='--', zorder=0)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('1 \u2013 Specificity', fontsize=8, fontname='Arial')
    ax.set_ylabel('Sensitivity', fontsize=8, fontname='Arial')

    # Panel title (10pt, centered)
    ax.set_title(panel_title, fontsize=10, fontname='Arial', pad=5,
                 loc='center')

    ax.legend(fontsize=7, frameon=False,
              loc='lower right',
              prop={'family': 'Arial', 'size': 7})

    style_ax(ax)

    # Panel label top-left (10pt bold)
    ax.text(-0.18, 1.08, lbl, transform=ax.transAxes,
            fontsize=10, fontweight='bold', fontname='Arial',
            va='top', ha='left')

# ── Panel D: Scatter plot (X=CENTILOIDS, Y=RAPS, color=y_actual) ─────────────
sc = ax_d.scatter(df['CENTILOIDS'], df['RAPS'],
                  c=df['y_actual'], cmap='coolwarm',
                  s=10, alpha=0.75, linewidths=0,
                  vmin=df['y_actual'].quantile(0.01),
                  vmax=df['y_actual'].quantile(0.99))

# Colorbar
cbar = fig.colorbar(sc, ax=ax_d, fraction=0.020, pad=0.02)
cbar.set_label('CDR-SB slope (/yr)', fontsize=7, fontname='Arial')
cbar.ax.tick_params(labelsize=7)
for label in cbar.ax.get_yticklabels():
    label.set_fontname('Arial')

# Linear regression (X=CENTILOIDS, Y=RAPS)
slope, intercept, r_val, p_val, _ = stats.linregress(df['CENTILOIDS'], df['RAPS'])
x_line = np.linspace(df['CENTILOIDS'].min(), df['CENTILOIDS'].max(), 200)
ax_d.plot(x_line, slope * x_line + intercept,
          color='#2C3E50', lw=1.2, ls='-', zorder=5)

# Pearson r annotation
p_str = f'p = {p_val:.2e}' if p_val >= 1e-4 else 'p < 0.0001'
ax_d.text(0.05, 0.93,
          f'r = {r_val:.3f}\n{p_str}',
          transform=ax_d.transAxes,
          fontsize=7, fontname='Arial',
          va='top', ha='left',
          bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none', alpha=0.8))

ax_d.set_xlabel('CENTILOIDS', fontsize=8, fontname='Arial')
ax_d.set_ylabel('RAPS', fontsize=8, fontname='Arial')

# Panel title (10pt, centered)
ax_d.set_title('RAPS vs Centiloid', fontsize=10, fontname='Arial', pad=5,
               loc='center')

style_ax(ax_d)

# Panel label top-left (10pt bold)
ax_d.text(-0.06, 1.08, 'D', transform=ax_d.transAxes,
          fontsize=10, fontweight='bold', fontname='Arial',
          va='top', ha='left')

# ── Save ─────────────────────────────────────────────────────────────────────
out_path = '/home/takumi/pj3-adni-roi/submit_v2/figures/fig3_roc_scatter.svg'
fig.savefig(out_path, format='svg', dpi=300, bbox_inches='tight')
print(f'Saved: {out_path}')
plt.close(fig)
