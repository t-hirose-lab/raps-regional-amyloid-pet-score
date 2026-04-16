#!/usr/bin/env python3
"""
22_raps_8roi_full_stats.py
===========================
Phase 0: 8-ROI RAPS の全統計量を算出。
Figure再生成・原稿更新に必要な全データを出力する。

Output:
  results/tables/v5_adni_predictions.csv       - ADNI 515人の8-ROI RAPS + covariates
  results/tables/v5_oasis3_predictions.csv     - OASIS-3 429人の8-ROI RAPS + covariates
  results/tables/v5_adni_subgroup.csv          - サブグループ解析
  results/tables/v5_adni_cox.csv               - Cox model結果
  results/tables/v5_adni_km.csv                - KM用データ
  results/tables/v5_oasis3_subgroup.csv        - OASIS-3サブグループ
  results/tables/v5_8roi_vs_11roi.csv          - 8-ROI vs 11-ROI比較（Supplementary用）
  results/tables/v5_bootstrap_stability.csv    - Bootstrap安定性（8 ROI focused）
  results/tables/v5_summary.json               - 全主要統計量のJSON
"""
import sys, warnings, json
import numpy as np, pandas as pd, joblib
from pathlib import Path
from scipy import stats as sp_stats
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, r2_score
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    PROCESSED_DIR, RESULTS_DIR, TABLES_DIR, MODELS_DIR,
    ANALYSIS_ROI_COLUMNS, COVARIATES,
    OUTER_FOLDS, INNER_FOLDS, L1_RATIOS, RANDOM_STATE,
    PRIMARY_OUTCOME, RAPID_DECLINE_THRESHOLDS,
)

ROIS_8 = [
    "LEFT_HIPPOCAMPUS_SUVR",
    "CTX_LH_TRANSVERSETEMPORAL_SUVR",
    "CTX_LH_FUSIFORM_SUVR",
    "RIGHT_PALLIDUM_SUVR",
    "CTX_LH_LINGUAL_SUVR",
    "CTX_RH_ENTORHINAL_SUVR",
    "LEFT_AMYGDALA_SUVR",
    "LEFT_PUTAMEN_SUVR",
]

OASIS3_ROI_MAP = {
    "LEFT_HIPPOCAMPUS_SUVR": "PET_fSUVR_L_HIPPOCAMPUS",
    "CTX_LH_TRANSVERSETEMPORAL_SUVR": "PET_fSUVR_L_CTX_TRANSTMP",
    "CTX_LH_FUSIFORM_SUVR": "PET_fSUVR_L_CTX_FUSIFORM",
    "RIGHT_PALLIDUM_SUVR": "PET_fSUVR_R_PALLIDUM",
    "CTX_LH_LINGUAL_SUVR": "PET_fSUVR_L_CTX_LINGUAL",
    "CTX_RH_ENTORHINAL_SUVR": "PET_fSUVR_R_CTX_ENTORHINAL",
    "LEFT_AMYGDALA_SUVR": "PET_fSUVR_L_AMYGDALA",
    "LEFT_PUTAMEN_SUVR": "PET_fSUVR_L_PUTAMEN",
}

EXCLUDED_ROIS = {"LEFT_CAUDATE_SUVR", "RIGHT_CAUDATE_SUVR"}


def steiger_z(r1, r2, r12, n):
    z1, z2 = np.arctanh(r1), np.arctanh(r2)
    denom = np.sqrt(2 * (1 - r12) / ((n - 3) * (1 + r12)))
    if denom == 0:
        return 0.0, 1.0
    z = (z1 - z2) / denom
    return float(z), float(2 * sp_stats.norm.sf(abs(z)))


def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    pooled_sd = np.sqrt(((n1-1)*g1.std()**2 + (n2-1)*g2.std()**2) / (n1+n2-2))
    return (g1.mean() - g2.mean()) / pooled_sd if pooled_sd > 0 else 0


def run_nested_cv(X, y, cov, rids, label=""):
    """Nested 5x5 CV, returns predictions and per-fold performance."""
    outer = KFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    preds, perfs = [], []
    for fi, (tr, te) in enumerate(outer.split(X)):
        cm = LinearRegression().fit(cov[tr], y[tr])
        res_tr = y[tr] - cm.predict(cov[tr])
        res_te = y[te] - cm.predict(cov[te])
        sc = StandardScaler().fit(X[tr])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            en = ElasticNetCV(l1_ratio=L1_RATIOS, cv=INNER_FOLDS,
                              random_state=RANDOM_STATE, max_iter=10000, n_jobs=-1)
            en.fit(sc.transform(X[tr]), res_tr)
        pred = en.predict(sc.transform(X[te]))
        r2 = r2_score(res_te, pred)
        r_val = float(np.corrcoef(res_te, pred)[0, 1])
        perfs.append({"fold": fi+1, "r2": r2, "r": r_val, "alpha": en.alpha_,
                       "l1": en.l1_ratio_, "nonzero": int(np.sum(en.coef_ != 0)),
                       "cov_r2_train": cm.score(cov[tr], y[tr])})
        for i, p in zip(te, pred):
            preds.append({"RID": rids[i], "raps_raw": float(p), "y_actual": float(y[i])})
        print(f"  {label} Fold {fi+1}: R2={r2:.4f}, r={r_val:.4f}, nonzero={en.coef_[en.coef_!=0].shape[0]}")
    df = pd.DataFrame(preds)
    df["RAPS"] = (df["raps_raw"] - df["raps_raw"].mean()) / df["raps_raw"].std()
    return df, pd.DataFrame(perfs)


def compute_subgroup_correlations(df, raps_col="RAPS", y_col="y_actual", cl_col="CENTILOIDS"):
    """Subgroup analysis: correlation between RAPS and slope across subgroups."""
    results = []
    # Overall
    r, p = sp_stats.pearsonr(df[raps_col], df[y_col])
    results.append({"subgroup": "Overall", "n": len(df), "r": r, "p": p})

    # CL strata
    for label, mask in [("CL 20-60", (df[cl_col]>=20)&(df[cl_col]<60)),
                         ("CL ≥60", df[cl_col]>=60)]:
        sub = df[mask]
        if len(sub) > 10:
            r, p = sp_stats.pearsonr(sub[raps_col], sub[y_col])
            results.append({"subgroup": label, "n": len(sub), "r": r, "p": p})

    # APOE (if available)
    if "APOE_E4_COUNT" in df.columns:
        for label, mask in [("APOE ε4+", df["APOE_E4_COUNT"]>0),
                             ("APOE ε4−", df["APOE_E4_COUNT"]==0)]:
            sub = df[mask]
            if len(sub) > 10:
                r, p = sp_stats.pearsonr(sub[raps_col], sub[y_col])
                results.append({"subgroup": label, "n": len(sub), "r": r, "p": p})

    # Sex (if available)
    if "PTGENDER_num" in df.columns:
        for label, val in [("Female", 0), ("Male", 1)]:
            sub = df[df["PTGENDER_num"]==val]
            if len(sub) > 10:
                r, p = sp_stats.pearsonr(sub[raps_col], sub[y_col])
                results.append({"subgroup": label, "n": len(sub), "r": r, "p": p})

    # Clinical stage (CDR-SB bl based)
    if "CDRSB_bl" in df.columns:
        for label, mask in [("CN (CDR-SB=0)", df["CDRSB_bl"]==0),
                             ("MCI (CDR-SB>0)", df["CDRSB_bl"]>0)]:
            sub = df[mask]
            if len(sub) > 10:
                r, p = sp_stats.pearsonr(sub[raps_col], sub[y_col])
                results.append({"subgroup": label, "n": len(sub), "r": r, "p": p})

    return pd.DataFrame(results)


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}

    # ===================================================================
    print("=" * 70)
    print("PART 1: ADNI 8-ROI RAPS")
    print("=" * 70)

    df = pd.read_csv(PROCESSED_DIR / "individual_slopes.csv")
    for c in ROIS_8:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Available 80 ROIs for 11-ROI comparison
    available_80 = [c for c in ANALYSIS_ROI_COLUMNS if c in df.columns and c not in EXCLUDED_ROIS]

    cols_8 = ["RID", PRIMARY_OUTCOME, "CENTILOIDS"] + COVARIATES + ROIS_8
    sub = df[cols_8].dropna()
    print(f"ADNI N = {len(sub)}")

    rids = sub["RID"].values
    y = sub[PRIMARY_OUTCOME].values.astype(float)
    cl = sub["CENTILOIDS"].values.astype(float)
    cov = sub[COVARIATES].values.astype(float)
    X8 = sub[ROIS_8].values.astype(float)

    # --- 8-ROI nested CV ---
    pred8, perf8 = run_nested_cv(X8, y, cov, rids, "8-ROI")

    # Add covariates to predictions
    cov_df = sub[["RID"] + COVARIATES + ["CENTILOIDS"]].copy()
    pred8 = pred8.merge(cov_df, on="RID", how="left")

    # --- 11-ROI nested CV (for comparison) ---
    print()
    cols_80 = ["RID", PRIMARY_OUTCOME, "CENTILOIDS"] + COVARIATES + available_80
    sub80 = df[cols_80].dropna()
    X80 = sub80[available_80].values.astype(float)
    y80 = sub80[PRIMARY_OUTCOME].values.astype(float)
    cov80 = sub80[COVARIATES].values.astype(float)
    rids80 = sub80["RID"].values
    pred11, perf11 = run_nested_cv(X80, y80, cov80, rids80, "80→11-ROI")

    # --- Primary comparisons ---
    print("\n" + "-" * 50)
    print("ADNI Primary: 8-ROI RAPS vs CL")
    r_raps = sp_stats.pearsonr(pred8["RAPS"], pred8["y_actual"])[0]
    r_cl = sp_stats.pearsonr(pred8["CENTILOIDS"], pred8["y_actual"])[0]
    r_rc = sp_stats.pearsonr(pred8["RAPS"], pred8["CENTILOIDS"])[0]
    sz, sp = steiger_z(r_raps, r_cl, r_rc, len(pred8))
    print(f"  r(RAPS, slope) = {r_raps:.4f}")
    print(f"  r(CL, slope)   = {r_cl:.4f}")
    print(f"  Steiger z={sz:.4f}, P={sp:.6f}")

    # AUC
    auc_results = []
    for thr in RAPID_DECLINE_THRESHOLDS:
        b = (pred8["y_actual"] >= thr).astype(int)
        npos = b.sum()
        if npos > 0 and npos < len(b):
            auc_r = roc_auc_score(b, pred8["RAPS"])
            auc_c = roc_auc_score(b, pred8["CENTILOIDS"])
            print(f"  AUC >{thr}: RAPS={auc_r:.4f}, CL={auc_c:.4f} (n={npos})")
            auc_results.append({"threshold": thr, "auc_raps": auc_r, "auc_cl": auc_c, "n_pos": npos})

    # NRI/IDI at >1.5
    from sklearn.linear_model import LogisticRegression
    b15 = (pred8["y_actual"] >= 1.5).astype(int)
    if b15.sum() > 5:
        kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        prob_raps, prob_cl = np.zeros(len(pred8)), np.zeros(len(pred8))
        for tr, te in kf.split(pred8):
            lr_r = LogisticRegression(max_iter=1000).fit(pred8["RAPS"].values[tr].reshape(-1,1), b15.values[tr])
            lr_c = LogisticRegression(max_iter=1000).fit(pred8["CENTILOIDS"].values[tr].reshape(-1,1), b15.values[tr])
            prob_raps[te] = lr_r.predict_proba(pred8["RAPS"].values[te].reshape(-1,1))[:,1]
            prob_cl[te] = lr_c.predict_proba(pred8["CENTILOIDS"].values[te].reshape(-1,1))[:,1]
        # NRI
        events = b15.values == 1
        nonevents = b15.values == 0
        nri_events = np.mean((prob_raps[events] > prob_cl[events]).astype(float) - (prob_raps[events] < prob_cl[events]).astype(float))
        nri_nonevents = np.mean((prob_raps[nonevents] < prob_cl[nonevents]).astype(float) - (prob_raps[nonevents] > prob_cl[nonevents]).astype(float))
        nri = nri_events + nri_nonevents
        # IDI
        idi = (prob_raps[events].mean() - prob_cl[events].mean()) - (prob_raps[nonevents].mean() - prob_cl[nonevents].mean())
        print(f"  NRI (>1.5): {nri:.4f}")
        print(f"  IDI (>1.5): {idi:.4f}")

    # --- CL 20-60 subgroup ---
    print("\n" + "-" * 50)
    print("CL 20-60 Subgroup")
    mask2060 = (pred8["CENTILOIDS"] >= 20) & (pred8["CENTILOIDS"] < 60)
    sub2060 = pred8[mask2060].copy()
    raps_median = pred8["RAPS"].median()
    high = sub2060[sub2060["RAPS"] >= raps_median]
    low = sub2060[sub2060["RAPS"] < raps_median]
    d = cohens_d(high["y_actual"], low["y_actual"])
    t, p_t = sp_stats.ttest_ind(high["y_actual"], low["y_actual"])
    print(f"  n={len(sub2060)}, high={len(high)}, low={len(low)}")
    print(f"  slope: high={high['y_actual'].mean():.3f}, low={low['y_actual'].mean():.3f}")
    print(f"  Cohen's d={d:.3f}, P={p_t:.6f}")
    # Partial correlations within CL 20-60
    from sklearn.linear_model import LinearRegression as LR
    raps_2060 = sub2060["RAPS"].values
    cl_2060 = sub2060["CENTILOIDS"].values
    y_2060 = sub2060["y_actual"].values
    res_raps = raps_2060 - LR().fit(cl_2060.reshape(-1,1), raps_2060).predict(cl_2060.reshape(-1,1))
    res_y_r = y_2060 - LR().fit(cl_2060.reshape(-1,1), y_2060).predict(cl_2060.reshape(-1,1))
    partial_raps = sp_stats.pearsonr(res_raps, res_y_r)[0]
    res_cl = cl_2060 - LR().fit(raps_2060.reshape(-1,1), cl_2060).predict(raps_2060.reshape(-1,1))
    res_y_c = y_2060 - LR().fit(raps_2060.reshape(-1,1), y_2060).predict(raps_2060.reshape(-1,1))
    partial_cl = sp_stats.pearsonr(res_cl, res_y_c)[0]
    print(f"  Partial r(RAPS|CL) = {partial_raps:.4f}")
    print(f"  Partial r(CL|RAPS) = {partial_cl:.4f}")

    # --- Subgroup analysis ---
    print("\n" + "-" * 50)
    print("Subgroup correlations")
    subgroup_df = compute_subgroup_correlations(pred8)
    print(subgroup_df.to_string(index=False))

    # --- Cox models ---
    print("\n" + "-" * 50)
    print("Cox models")
    cox_df = pred8[["RAPS", "y_actual", "CENTILOIDS"] + COVARIATES].copy()
    # Time to event: CDR-SB increase >= 1.0 point
    cox_df["event"] = (cox_df["y_actual"] > 0).astype(int)
    cox_df["time"] = np.where(cox_df["y_actual"] > 0,
                               np.minimum(1.0 / cox_df["y_actual"], 4.0),
                               4.0)
    cox_df["time"] = cox_df["time"].clip(lower=0.1)

    # Z-standardize predictors
    for c in ["RAPS", "CENTILOIDS"] + COVARIATES:
        cox_df[f"{c}_z"] = (cox_df[c] - cox_df[c].mean()) / cox_df[c].std()

    cox_results = []
    # Model 1: RAPS alone
    cph = CoxPHFitter()
    cph.fit(cox_df[["time", "event", "RAPS_z"]], duration_col="time", event_col="event")
    hr_raps = np.exp(cph.params_["RAPS_z"])
    ci_lo, ci_hi = np.exp(cph.confidence_intervals_.values[0])
    p_raps = cph.summary["p"]["RAPS_z"]
    conc_raps = cph.concordance_index_
    print(f"  Model 1 (RAPS): HR={hr_raps:.2f} [{ci_lo:.2f}-{ci_hi:.2f}], P={p_raps:.2e}, C={conc_raps:.3f}")
    cox_results.append({"model": "RAPS alone", "var": "RAPS", "HR": hr_raps,
                         "CI_lo": ci_lo, "CI_hi": ci_hi, "P": p_raps, "C": conc_raps})

    # Model 2: CL alone
    cph2 = CoxPHFitter()
    cph2.fit(cox_df[["time", "event", "CENTILOIDS_z"]], duration_col="time", event_col="event")
    hr_cl = np.exp(cph2.params_["CENTILOIDS_z"])
    ci_lo2, ci_hi2 = np.exp(cph2.confidence_intervals_.values[0])
    conc_cl = cph2.concordance_index_
    print(f"  Model 2 (CL):   HR={hr_cl:.2f} [{ci_lo2:.2f}-{ci_hi2:.2f}], C={conc_cl:.3f}")
    cox_results.append({"model": "CL alone", "var": "CL", "HR": hr_cl,
                         "CI_lo": ci_lo2, "CI_hi": ci_hi2, "P": cph2.summary["p"]["CENTILOIDS_z"], "C": conc_cl})

    # Model 3: Multivariate
    mv_cols = ["RAPS_z", "CENTILOIDS_z"] + [f"{c}_z" for c in COVARIATES]
    cph3 = CoxPHFitter()
    cph3.fit(cox_df[["time", "event"] + mv_cols], duration_col="time", event_col="event")
    print(f"  Model 3 (Multi): C={cph3.concordance_index_:.3f}")
    for var in mv_cols:
        hr = np.exp(cph3.params_[var])
        ci = np.exp(cph3.confidence_intervals_.loc[var].values)
        p = cph3.summary["p"][var]
        name = var.replace("_z", "")
        print(f"    {name:25s} HR={hr:.2f} [{ci[0]:.2f}-{ci[1]:.2f}] P={p:.4f}")
        cox_results.append({"model": "Multivariate", "var": name, "HR": hr,
                             "CI_lo": ci[0], "CI_hi": ci[1], "P": p, "C": cph3.concordance_index_})

    # --- Incremental C-index ---
    print("\n  Incremental C-index:")
    from lifelines import CoxPHFitter as CPH
    cov_z = [f"{c}_z" for c in COVARIATES]
    # Clinical only
    c1 = CPH().fit(cox_df[["time","event"]+cov_z], "time", "event")
    # Clinical + CL
    c2 = CPH().fit(cox_df[["time","event"]+cov_z+["CENTILOIDS_z"]], "time", "event")
    # Clinical + RAPS
    c3 = CPH().fit(cox_df[["time","event"]+cov_z+["RAPS_z"]], "time", "event")
    # Clinical + CL + RAPS
    c4 = CPH().fit(cox_df[["time","event"]+cov_z+["CENTILOIDS_z","RAPS_z"]], "time", "event")
    print(f"  Clinical only:      C = {c1.concordance_index_:.3f}")
    print(f"  Clinical + CL:      C = {c2.concordance_index_:.3f} (ΔC = +{c2.concordance_index_-c1.concordance_index_:.3f})")
    print(f"  Clinical + RAPS:    C = {c3.concordance_index_:.3f} (ΔC = +{c3.concordance_index_-c1.concordance_index_:.3f})")
    print(f"  Clinical + CL+RAPS: C = {c4.concordance_index_:.3f} (ΔC = +{c4.concordance_index_-c1.concordance_index_:.3f})")

    # --- KM data ---
    print("\n" + "-" * 50)
    print("KM data")
    km_df = cox_df[["time", "event", "RAPS_z", "CENTILOIDS_z"]].copy()
    km_df["RAPS_group"] = np.where(pred8["RAPS"].values >= raps_median, "High", "Low")
    cl_median = pred8["CENTILOIDS"].median()
    km_df["CL_group"] = np.where(pred8["CENTILOIDS"].values >= cl_median, "High", "Low")
    lr = logrank_test(km_df[km_df["RAPS_group"]=="High"]["time"],
                       km_df[km_df["RAPS_group"]=="Low"]["time"],
                       km_df[km_df["RAPS_group"]=="High"]["event"],
                       km_df[km_df["RAPS_group"]=="Low"]["event"])
    print(f"  RAPS log-rank: chi2={lr.test_statistic:.1f}, P={lr.p_value:.2e}")

    # ===================================================================
    print("\n" + "=" * 70)
    print("PART 2: OASIS-3 8-ROI RAPS")
    print("=" * 70)

    # Load model
    model = joblib.load(MODELS_DIR / "raps_8roi_final_model.joblib")
    final_enet = model["enet"]
    final_scaler = model["scaler"]
    final_cov_model = model["cov_model"]

    # Load OASIS-3 data
    oasis_meta = pd.read_csv(PROCESSED_DIR / "oasis3_individual_raps.csv")
    pup = pd.read_csv(Path("/home/takumi/pj3-adni-roi/data/raw/OASIS3/OASIS3_data_files/"
                            "OASIS3_data_files/scans/PUP-PUP_output/resources/csv/files/OASIS3_PUP.csv"))

    # Map ROIs
    for adni_name, oasis_name in OASIS3_ROI_MAP.items():
        pup[adni_name] = pd.to_numeric(pup[oasis_name], errors="coerce")

    # Merge
    id_col = pup.columns[0]
    pup["_merge_id"] = pup[id_col].astype(str).str.extract(r'(OAS\d+)')[0]
    oasis_meta["_merge_id"] = oasis_meta["OASISID"].astype(str)
    pup_first = pup.groupby("_merge_id").first().reset_index()
    oasis = oasis_meta.merge(pup_first[["_merge_id"] + ROIS_8], on="_merge_id", how="inner")
    print(f"  OASIS-3 N = {len(oasis)}")

    # Apply model
    X_o = oasis[ROIS_8].values.astype(float)
    y_o = oasis["CDRSB_slope"].values.astype(float)
    X_o_scaled = final_scaler.transform(X_o)
    raps_o_raw = final_enet.predict(X_o_scaled)
    oasis["RAPS"] = (raps_o_raw - raps_o_raw.mean()) / raps_o_raw.std()

    # CL
    cl_o = pd.to_numeric(oasis["CENTILOIDS"], errors="coerce").values
    valid = ~np.isnan(cl_o)
    raps_v = oasis["RAPS"].values[valid]
    cl_v = cl_o[valid]
    y_v = y_o[valid]
    n_v = int(valid.sum())

    r_raps_o = sp_stats.pearsonr(raps_v, y_v)[0]
    r_cl_o = sp_stats.pearsonr(cl_v, y_v)[0]
    r_rc_o = sp_stats.pearsonr(raps_v, cl_v)[0]
    sz_o, sp_o = steiger_z(r_raps_o, r_cl_o, r_rc_o, n_v)
    print(f"  r(RAPS, slope) = {r_raps_o:.4f}")
    print(f"  r(CL, slope)   = {r_cl_o:.4f}")
    print(f"  Steiger z={sz_o:.4f}, P={sp_o:.6f}")

    for thr in RAPID_DECLINE_THRESHOLDS:
        b = (y_v >= thr).astype(int)
        npos = b.sum()
        if npos > 0 and npos < len(b):
            auc_r = roc_auc_score(b, raps_v)
            auc_c = roc_auc_score(b, cl_v)
            print(f"  AUC >{thr}: RAPS={auc_r:.4f}, CL={auc_c:.4f} (n={npos})")

    # OASIS-3 CL 20-60
    mask2060_o = (cl_v >= 20) & (cl_v < 60)
    n2060_o = int(mask2060_o.sum())
    raps_med_o = np.median(raps_v)  # Use OASIS overall median? Or ADNI?
    # Use ADNI median applied to OASIS (prospective scenario)
    # Need ADNI RAPS median in the same scale... use 0 (z-scored)
    high_o = y_v[mask2060_o & (raps_v >= 0)]
    low_o = y_v[mask2060_o & (raps_v < 0)]
    if len(high_o) > 1 and len(low_o) > 1:
        d_o = cohens_d(pd.Series(high_o), pd.Series(low_o))
        t_o, p_o = sp_stats.ttest_ind(high_o, low_o)
        print(f"\n  OASIS-3 CL 20-60 (n={n2060_o}): high={len(high_o)}, low={len(low_o)}")
        print(f"    slope: high={high_o.mean():.3f}, low={low_o.mean():.3f}")
        print(f"    Cohen's d = {d_o:.3f}, P = {p_o:.6f}")

    # ===================================================================
    print("\n" + "=" * 70)
    print("PART 3: 8-ROI vs 11-ROI comparison (Supplementary)")
    print("=" * 70)

    # 11-ROI stats
    r_raps11 = sp_stats.pearsonr(pred11["RAPS"], pred11["y_actual"])[0]
    comp_rows = []
    for label, pred, perf in [("8-ROI", pred8, perf8), ("80→11-ROI", pred11, perf11)]:
        r_val = sp_stats.pearsonr(pred["RAPS"], pred["y_actual"])[0]
        aucs_comp = {}
        for thr in RAPID_DECLINE_THRESHOLDS:
            b = (pred["y_actual"] >= thr).astype(int)
            if b.sum() > 0 and b.sum() < len(b):
                aucs_comp[thr] = roc_auc_score(b, pred["RAPS"])
        comp_rows.append({
            "model": label,
            "n_rois_input": 8 if "8" in label else 80,
            "cv_r2_mean": perf["r2"].mean(),
            "cv_r2_sd": perf["r2"].std(),
            "cv_r_mean": perf["r"].mean(),
            "cv_r_sd": perf["r"].std(),
            "r_raps_slope": r_val,
            **{f"auc_{thr}": aucs_comp.get(thr, np.nan) for thr in RAPID_DECLINE_THRESHOLDS},
        })
    comp_df = pd.DataFrame(comp_rows)
    print(comp_df.to_string(index=False))

    # ===================================================================
    print("\n" + "=" * 70)
    print("SAVING ALL OUTPUTS")
    print("=" * 70)

    # Save predictions
    pred8.to_csv(TABLES_DIR / "v5_adni_predictions.csv", index=False)
    print(f"  {TABLES_DIR / 'v5_adni_predictions.csv'}")

    oasis[["OASISID", "RAPS", "CDRSB_slope", "CENTILOIDS"] + ROIS_8].to_csv(
        TABLES_DIR / "v5_oasis3_predictions.csv", index=False)
    print(f"  {TABLES_DIR / 'v5_oasis3_predictions.csv'}")

    subgroup_df.to_csv(TABLES_DIR / "v5_adni_subgroup.csv", index=False)
    print(f"  {TABLES_DIR / 'v5_adni_subgroup.csv'}")

    pd.DataFrame(cox_results).to_csv(TABLES_DIR / "v5_adni_cox.csv", index=False)
    print(f"  {TABLES_DIR / 'v5_adni_cox.csv'}")

    km_df.to_csv(TABLES_DIR / "v5_adni_km.csv", index=False)
    print(f"  {TABLES_DIR / 'v5_adni_km.csv'}")

    comp_df.to_csv(TABLES_DIR / "v5_8roi_vs_11roi.csv", index=False)
    print(f"  {TABLES_DIR / 'v5_8roi_vs_11roi.csv'}")

    perf8.to_csv(TABLES_DIR / "v5_adni_cv_performance.csv", index=False)
    print(f"  {TABLES_DIR / 'v5_adni_cv_performance.csv'}")

    # Summary JSON
    summary = {
        "adni_n": len(pred8),
        "adni_r_raps": round(r_raps, 4),
        "adni_r_cl": round(r_cl, 4),
        "adni_steiger_z": round(sz, 4),
        "adni_steiger_p": round(sp, 6),
        "adni_cv_r2_mean": round(perf8["r2"].mean(), 4),
        "adni_cv_r_mean": round(perf8["r"].mean(), 4),
        "adni_cl2060_d": round(d, 3),
        "adni_cl2060_n": len(sub2060),
        "adni_partial_raps_cl": round(partial_raps, 4),
        "adni_partial_cl_raps": round(partial_cl, 4),
        "adni_raps_median": round(float(raps_median), 4),
        "oasis3_n": n_v,
        "oasis3_r_raps": round(r_raps_o, 4),
        "oasis3_r_cl": round(r_cl_o, 4),
        "oasis3_steiger_z": round(sz_o, 4),
        "oasis3_steiger_p": round(sp_o, 6),
        "cox_raps_hr": round(float(hr_raps), 2),
        "cox_cl_hr": round(float(hr_cl), 2),
        "cox_multi_c": round(float(cph3.concordance_index_), 3),
        "c_clinical_only": round(float(c1.concordance_index_), 3),
        "c_clinical_cl": round(float(c2.concordance_index_), 3),
        "c_clinical_raps": round(float(c3.concordance_index_), 3),
        "c_clinical_cl_raps": round(float(c4.concordance_index_), 3),
        "km_raps_logrank_chi2": round(float(lr.test_statistic), 1),
        "km_raps_logrank_p": float(lr.p_value),
    }
    for thr in RAPID_DECLINE_THRESHOLDS:
        for a in auc_results:
            if a["threshold"] == thr:
                summary[f"adni_auc_raps_{thr}"] = round(a["auc_raps"], 4)
                summary[f"adni_auc_cl_{thr}"] = round(a["auc_cl"], 4)

    with open(TABLES_DIR / "v5_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  {TABLES_DIR / 'v5_summary.json'}")

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
