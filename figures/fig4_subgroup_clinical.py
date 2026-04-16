"""
Figure 4: Subgroup analysis and clinical utility
4-panel figure (2×2):
  A: Forest plot (subgroup correlations)
  B: CL-stratified heatmap (ElasticNet coefficients)
  C: Kaplan-Meier curves (RAPS High vs Low)
  D: Cox forest plot (Multivariate HR)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
# Arial (preferred) → Nimbus Sans (WSL fallback, Ghostscript Arial clone) → sans-serif
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Nimbus Sans", "DejaVu Sans"]

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler

# ── paths ─────────────────────────────────────────────────────────────────────
DIR_TABLES = "/home/takumi/pj3-adni-roi/results/tables"
DIR_PROC   = "/home/takumi/pj3-adni-roi/data/processed"
OUT_PATH   = "/home/takumi/pj3-adni-roi/submit_v2/figures/fig4_subgroup_clinical.svg"

# ── data ──────────────────────────────────────────────────────────────────────
df_sg  = pd.read_csv(f"{DIR_TABLES}/v5_adni_subgroup.csv")
df_km  = pd.read_csv(f"{DIR_TABLES}/v5_adni_km.csv")
df_cox = pd.read_csv(f"{DIR_TABLES}/v5_adni_cox.csv")
df_pred = pd.read_csv(f"{DIR_TABLES}/v5_adni_predictions.csv")
df_slopes = pd.read_csv(f"{DIR_PROC}/individual_slopes.csv")

# ── Panel A helpers ───────────────────────────────────────────────────────────
def fisher_ci(r, n, alpha=0.05):
    z    = np.arctanh(r)
    se   = 1.0 / np.sqrt(n - 3)
    z_lo = z - 1.96 * se
    z_hi = z + 1.96 * se
    return np.tanh(z_lo), np.tanh(z_hi)

# ── Panel B helpers ───────────────────────────────────────────────────────────
ROIS = [
    "LEFT_HIPPOCAMPUS_SUVR",
    "CTX_LH_TRANSVERSETEMPORAL_SUVR",
    "CTX_LH_FUSIFORM_SUVR",
    "RIGHT_PALLIDUM_SUVR",
    "CTX_LH_LINGUAL_SUVR",
    "CTX_RH_ENTORHINAL_SUVR",
    "LEFT_AMYGDALA_SUVR",
    "LEFT_PUTAMEN_SUVR",
]

ROI_LABELS = {
    "LEFT_HIPPOCAMPUS_SUVR":         "L Hippocampus",
    "CTX_LH_TRANSVERSETEMPORAL_SUVR":"L Transverse Temp.",
    "CTX_LH_FUSIFORM_SUVR":          "L Fusiform",
    "RIGHT_PALLIDUM_SUVR":           "R Pallidum",
    "CTX_LH_LINGUAL_SUVR":           "L Lingual",
    "CTX_RH_ENTORHINAL_SUVR":        "R Entorhinal",
    "LEFT_AMYGDALA_SUVR":            "L Amygdala",
    "LEFT_PUTAMEN_SUVR":             "L Putamen",
}

COVARS = ["AGE", "PTGENDER_num", "PTEDUCAT", "APOE_E4_COUNT", "CDRSB_bl"]

def fit_elasticnet_coefs(df_merged, stratum_mask):
    sub = df_merged[stratum_mask].dropna(subset=ROIS + COVARS + ["CDRSB_slope"])
    X = sub[ROIS + COVARS].values
    y = sub["CDRSB_slope"].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = ElasticNetCV(cv=5, l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
                         max_iter=10000, random_state=42)
    model.fit(X_scaled, y)
    coefs = model.coef_[:len(ROIS)]  # ROI coefficients only
    return coefs

# merge slopes with predictions on RID
df_slopes_sub = df_slopes[["RID", "CDRSB_slope", "CENTILOIDS"] + ROIS].copy()
df_merged = df_pred.merge(df_slopes_sub[["RID"] + ROIS + ["CDRSB_slope"]], on="RID", how="inner")

mask_lo = df_merged["CENTILOIDS"].between(20, 60, inclusive="both")
mask_hi = df_merged["CENTILOIDS"] >= 60

coefs_lo = fit_elasticnet_coefs(df_merged, mask_lo)
coefs_hi = fit_elasticnet_coefs(df_merged, mask_hi)

# ── Panel D variable ordering / labeling ──────────────────────────────────────
VAR_LABEL = {
    "RAPS":          "RAPS",
    "CENTILOIDS":    "Centiloid",
    "AGE":           "Age",
    "PTGENDER_num":  "Sex",
    "PTEDUCAT":      "Education",
    "APOE_E4_COUNT": "APOE ε4",
    "CDRSB_bl":      "Baseline CDR-SB",
}

df_mv = df_cox[df_cox["model"] == "Multivariate"].copy()
df_mv["label"] = df_mv["var"].map(VAR_LABEL)

# ── figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(9, 8))
gs  = gridspec.GridSpec(2, 2, figure=fig,
                        left=0.10, right=0.96,
                        top=0.95, bottom=0.08,
                        wspace=0.38, hspace=0.38)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 0])
ax_d = fig.add_subplot(gs[1, 1])

PANEL_KW   = dict(fontsize=10, fontweight="bold", ha="left", va="top",
                  transform=None)
TICK_SIZE  = 7
LABEL_SIZE = 8
ANNOT_SIZE = 7

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Panel A – Forest plot
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ax = ax_a

subgroup_order = [
    "Overall",
    "CL 20-60", "CL ≥60",
    "APOE ε4+", "APOE ε4−",
    "Female", "Male",
    "CN (CDR-SB=0)", "MCI (CDR-SB>0)",
]

# Label mapping: subgroup name → display label with (n=XX)
subgroup_label_map = {
    "Overall":        "Overall",
    "CL 20-60":       "CL 20–60",
    "CL ≥60":         "CL ≥60",
    "APOE ε4+":       "APOE ε4+",
    "APOE ε4−":       "APOE ε4−",
    "Female":         "Female",
    "Male":           "Male",
    "CN (CDR-SB=0)":  "CN",
    "MCI (CDR-SB>0)": "MCI",
}

df_sg["subgroup"] = df_sg["subgroup"].str.strip()
df_plot = pd.DataFrame({"subgroup": subgroup_order}).merge(df_sg, on="subgroup", how="left")

n_rows = len(df_plot)
y_pos  = np.arange(n_rows)[::-1]   # top = first row

POINT_COLOR = "#2C3E50"  # single dark blue-gray for all points

for i, (_, row) in enumerate(df_plot.iterrows()):
    y = y_pos[i]
    r = row["r"]
    n = int(row["n"])
    ci_lo, ci_hi = fisher_ci(r, n)

    ax.plot([ci_lo, ci_hi], [y, y], color=POINT_COLOR, lw=1.2, zorder=2)
    ax.plot(r, y, "o", color=POINT_COLOR, ms=5, zorder=3)

    # r value on the right (outside plot area)
    ax.annotate(f"r = {r:.3f}", xy=(1.01, y),
                xycoords=("axes fraction", "data"),
                fontsize=ANNOT_SIZE - 0.5, va="center", color=POINT_COLOR)

# y-tick labels with sample sizes
ytick_labels = [
    f"{subgroup_label_map.get(sg, sg)} (n={int(df_plot.loc[df_plot['subgroup']==sg, 'n'].values[0])})"
    for sg in df_plot["subgroup"]
]

ax.axvline(0, color="black", lw=0.8, ls="--", zorder=1)
ax.set_yticks(y_pos)
ax.set_yticklabels(ytick_labels, fontsize=TICK_SIZE)
ax.set_xlabel("Pearson r", fontsize=LABEL_SIZE)
ax.set_xlim(-0.1, 0.85)
ax.tick_params(axis="x", labelsize=TICK_SIZE)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_title("A  Subgroup Correlations", fontsize=10, fontweight="bold", loc="left", pad=4)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Panel B – CL-stratified heatmap
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ax = ax_b

heatmap_data = np.column_stack([coefs_lo, coefs_hi])
row_labels   = [ROI_LABELS[r] for r in ROIS]
col_labels   = ["CL 20–60", "CL ≥60"]

vmax = max(abs(heatmap_data.max()), abs(heatmap_data.min()))
vmax = max(vmax, 1e-6)  # prevent zero scale

im = ax.imshow(heatmap_data, cmap="RdBu_r", aspect="auto",
               vmin=-vmax, vmax=vmax)

ax.set_xticks([0, 1])
ax.set_xticklabels(col_labels, fontsize=TICK_SIZE)
ax.set_yticks(range(len(row_labels)))
ax.set_yticklabels(row_labels, fontsize=TICK_SIZE)
ax.tick_params(axis="both", length=2)

# annotate cell values
for ri in range(len(ROIS)):
    for ci in range(2):
        val = heatmap_data[ri, ci]
        txt_col = "white" if abs(val) > 0.5 * vmax else "black"
        ax.text(ci, ri, f"{val:.3f}", ha="center", va="center",
                fontsize=ANNOT_SIZE - 0.5, color=txt_col)

cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.03)
cbar.set_label("ElasticNet coef.", fontsize=ANNOT_SIZE)
cbar.ax.tick_params(labelsize=ANNOT_SIZE)

ax.set_title("B  CL-Stratified ROI Coefficients", fontsize=10, fontweight="bold", loc="left", pad=4)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Panel C – Kaplan–Meier
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ax = ax_c

# Censor subjects at time=4.0 (cap artifact)
df_km.loc[df_km["time"] >= 4.0, "event"] = 0

grp_high = df_km[df_km["RAPS_group"] == "High"]
grp_low  = df_km[df_km["RAPS_group"] == "Low"]

lr = logrank_test(grp_high["time"], grp_low["time"],
                  event_observed_A=grp_high["event"],
                  event_observed_B=grp_low["event"])

kmf_high = KaplanMeierFitter()
kmf_low  = KaplanMeierFitter()

kmf_high.fit(grp_high["time"], event_observed=grp_high["event"], label="High RAPS")
kmf_low.fit(grp_low["time"],   event_observed=grp_low["event"],  label="Low RAPS")

kmf_high.plot_survival_function(ax=ax, ci_show=True,
                                 color="#C44E52", lw=1.5,
                                 show_censors=True, censor_styles={"ms": 3, "marker": "+"})
kmf_low.plot_survival_function(ax=ax, ci_show=True,
                                color="#4C72B0", lw=1.5,
                                show_censors=True, censor_styles={"ms": 3, "marker": "+"})

p_val = lr.p_value
p_str = f"Log-rank P = {p_val:.2e}"
ax.text(0.97, 0.97, p_str, transform=ax.transAxes,
        fontsize=ANNOT_SIZE, ha="right", va="top")

ax.set_xlabel("Time (years)", fontsize=LABEL_SIZE)
ax.set_ylabel("Progression-free probability", fontsize=LABEL_SIZE)
ax.tick_params(labelsize=TICK_SIZE)
ax.legend(fontsize=ANNOT_SIZE, frameon=False, loc="lower left")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_ylim(0, 1.05)
ax.set_xlim(0, 3.9)   # clip at 3.9 to avoid vertical drop artifact at TIME=4.0
ax.set_title("C  Kaplan-Meier: Clinical Progression", fontsize=10, fontweight="bold", loc="left", pad=4)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Panel D – Cox forest plot (Multivariate)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ax = ax_d

var_order = ["RAPS", "CENTILOIDS", "AGE", "PTGENDER_num",
             "PTEDUCAT", "APOE_E4_COUNT", "CDRSB_bl"]
df_mv_ord = pd.DataFrame({"var": var_order}).merge(df_mv, on="var", how="left")

n_vars = len(df_mv_ord)
y_pos_d = np.arange(n_vars)[::-1]

sig_color = "#C44E52"
ns_color  = "#666666"

for i, (_, row) in enumerate(df_mv_ord.iterrows()):
    y   = y_pos_d[i]
    hr  = row["HR"]
    lo  = row["CI_lo"]
    hi  = row["CI_hi"]
    p   = row["P"]
    col = sig_color if p < 0.05 else ns_color

    ax.plot([lo, hi], [y, y], color=col, lw=1.2, zorder=2)
    ax.plot(hr, y, "D", color=col, ms=4.5, zorder=3)

    # HR [CI] annotation
    ann = f"{hr:.2f} [{lo:.2f}–{hi:.2f}]"
    ax.annotate(ann, xy=(1.01, y),
                xycoords=("axes fraction", "data"),
                fontsize=ANNOT_SIZE - 0.5, va="center", color=col)

ax.axvline(1, color="black", lw=0.8, ls="--", zorder=1)
ax.set_yticks(y_pos_d)
ax.set_yticklabels(df_mv_ord["label"].tolist(), fontsize=TICK_SIZE)
ax.set_xlabel("Hazard ratio", fontsize=LABEL_SIZE)
ax.set_xscale("log")
ax.tick_params(axis="x", labelsize=TICK_SIZE)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_title("D  Hazard Ratio (95% CI)", fontsize=10, fontweight="bold", loc="left", pad=4)

# ── save ──────────────────────────────────────────────────────────────────────
fig.savefig(OUT_PATH, format="svg", bbox_inches="tight", dpi=150)
print(f"Saved: {OUT_PATH}")
