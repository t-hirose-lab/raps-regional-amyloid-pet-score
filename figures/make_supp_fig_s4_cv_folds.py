"""
Supplementary Figure S4: Nested Cross-Validation Performance by Fold
8-ROI RAPS
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# SVG font settings
matplotlib.rcParams['svg.fonttype'] = 'none'
# Use Nimbus Sans (Arial-equivalent) for rendering; SVG font name replaced to Arial in post-processing
matplotlib.rcParams['font.family'] = 'Nimbus Sans'

# Load data
df = pd.read_csv('/home/takumi/pj3-adni-roi/results/tables/v5_adni_cv_performance.csv')

# Metrics to plot
metrics = [
    ('r2',      'R²',                    'R² (test)'),
    ('r',       'Pearson r',             'Pearson r (test)'),
    ('nonzero', 'Non-zero coefficients', 'Non-zero coefficients'),
]

bar_color = '#3498DB'
mean_line_color = '#E74C3C'

fig, axes = plt.subplots(1, 3, figsize=(10, 4))
fig.suptitle('Nested Cross-Validation Performance by Fold',
             fontsize=10, fontweight='normal')

fold_labels = [f'Fold {i}' for i in df['fold']]
x = np.arange(len(df))
bar_width = 0.55

for ax, (col, panel_title, ylabel) in zip(axes, metrics):
    values = df[col].values
    mean_val = values.mean()
    sd_val   = values.std(ddof=1)

    # Bar chart
    ax.bar(x, values, width=bar_width, color=bar_color, zorder=2)

    # Horizontal dashed mean line
    ax.axhline(mean_val, color=mean_line_color, linestyle='--', linewidth=1.2,
               zorder=3)

    # Axis labels & ticks
    ax.set_title(panel_title, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xlabel('Fold', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(fold_labels, fontsize=7)
    ax.tick_params(axis='y', labelsize=7)

    # Remove top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Annotation: mean ± SD below panel
    annot_text = f'Mean \u00b1 SD: {mean_val:.3f} \u00b1 {sd_val:.3f}'
    ax.text(0.5, -0.22, annot_text,
            ha='center', va='top', transform=ax.transAxes,
            fontsize=7, color='#333333')

    # Y-axis padding
    y_range = values.max() - values.min() if values.max() != values.min() else abs(values.max()) * 0.2 + 0.1
    y_pad = y_range * 0.15
    y_bottom = min(values.min(), mean_val) - y_pad
    y_top    = max(values.max(), mean_val) + y_pad
    ax.set_ylim(y_bottom, y_top)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])

out_path = '/home/takumi/pj3-adni-roi/submit_v2/figures/supp_fig_s4_cv_folds.svg'
plt.savefig(out_path, format='svg', bbox_inches='tight')
plt.close()

# Post-process SVG: replace Nimbus Sans with Arial so the SVG specifies Arial
with open(out_path, 'r', encoding='utf-8') as f:
    svg_text = f.read()
svg_text = svg_text.replace('Nimbus Sans', 'Arial')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(svg_text)

print(f'Saved: {out_path}')
