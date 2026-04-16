#!/usr/bin/env python3
"""
21_raps_8roi_full_validation.py
================================
8-ROI RAPS (final model ∩ bootstrap ≥80%) の完全検証。
1. ADNI refit + serialize
2. OASIS-3 適用 + 全統計
3. 主要結果の出力
"""
import sys, warnings, json
import numpy as np, pandas as pd, joblib
from pathlib import Path
from scipy import stats as sp_stats
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    PROCESSED_DIR, RESULTS_DIR, TABLES_DIR, MODELS_DIR,
    COVARIATES, OUTER_FOLDS, INNER_FOLDS, L1_RATIOS, RANDOM_STATE,
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

# OASIS-3 ROI name mapping (from Supplementary Table S18)
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


def steiger_z(r1, r2, r12, n):
    z1, z2 = np.arctanh(r1), np.arctanh(r2)
    denom = np.sqrt(2 * (1 - r12) / ((n - 3) * (1 + r12)))
    if denom == 0: return 0.0, 1.0
    z = (z1 - z2) / denom
    return float(z), float(2 * sp_stats.norm.sf(abs(z)))


def compute_aucs(raps, y, cl, thresholds):
    results = {}
    for thr in thresholds:
        b = (y >= thr).astype(int)
        npos = b.sum()
        if npos > 0 and npos < len(b):
            auc_r = roc_auc_score(b, raps)
            auc_c = roc_auc_score(b, cl)
            results[thr] = {"auc_raps": auc_r, "auc_cl": auc_c, "n_pos": int(npos)}
    return results


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STEP 1: ADNI nested CV (8-ROI)")
    print("=" * 70)

    df = pd.read_csv(PROCESSED_DIR / "individual_slopes.csv")
    for c in ROIS_8:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    cols = ["RID", PRIMARY_OUTCOME, "CENTILOIDS"] + COVARIATES + ROIS_8
    sub = df[cols].dropna()
    print(f"ADNI N = {len(sub)}")

    rids = sub["RID"].values
    y = sub[PRIMARY_OUTCOME].values.astype(float)
    cl = sub["CENTILOIDS"].values.astype(float)
    cov = sub[COVARIATES].values.astype(float)
    X = sub[ROIS_8].values.astype(float)

    # Nested CV
    outer = KFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    preds = []
    perfs = []
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
        perfs.append({"fold": fi+1, "r2": r2, "r": r_val,
                       "alpha": en.alpha_, "l1": en.l1_ratio_,
                       "nonzero": int(np.sum(en.coef_ != 0))})
        print(f"  Fold {fi+1}: R2={r2:.4f}, r={r_val:.4f}, nonzero={np.sum(en.coef_!=0)}")
        for i, p in zip(te, pred):
            preds.append({"RID": rids[i], "pred": float(p), "y": float(y[i]),
                           "cl": float(cl[np.where(rids==rids[i])[0][0]])})

    pred_df = pd.DataFrame(preds)
    pred_df["RAPS"] = (pred_df["pred"] - pred_df["pred"].mean()) / pred_df["pred"].std()
    perf_df = pd.DataFrame(perfs)

    print(f"\n  Mean R2 = {perf_df['r2'].mean():.4f} ± {perf_df['r2'].std():.4f}")
    print(f"  Mean r  = {perf_df['r'].mean():.4f} ± {perf_df['r'].std():.4f}")

    # ADNI primary stats
    r_raps = sp_stats.pearsonr(pred_df["RAPS"], pred_df["y"])[0]
    r_cl = sp_stats.pearsonr(pred_df["cl"], pred_df["y"])[0]
    r_raps_cl = sp_stats.pearsonr(pred_df["RAPS"], pred_df["cl"])[0]
    sz, sp = steiger_z(r_raps, r_cl, r_raps_cl, len(pred_df))

    print(f"\n  ADNI RAPS vs CL:")
    print(f"    r(RAPS, slope) = {r_raps:.4f}")
    print(f"    r(CL, slope)   = {r_cl:.4f}")
    print(f"    Steiger z = {sz:.4f}, P = {sp:.6f}")

    aucs = compute_aucs(pred_df["RAPS"].values, pred_df["y"].values,
                         pred_df["cl"].values, RAPID_DECLINE_THRESHOLDS)
    for thr, v in aucs.items():
        print(f"    AUC >{thr}: RAPS={v['auc_raps']:.4f}, CL={v['auc_cl']:.4f} (n_pos={v['n_pos']})")

    # CL 20-60 subgroup
    cl_vals = pred_df["cl"].values
    mask2060 = (cl_vals >= 20) & (cl_vals < 60)
    sub2060 = pred_df[mask2060].copy()
    n2060 = len(sub2060)
    raps_median = pred_df["RAPS"].median()  # Overall median
    high = sub2060[sub2060["RAPS"] >= raps_median]
    low = sub2060[sub2060["RAPS"] < raps_median]
    if len(high) > 1 and len(low) > 1:
        t, p_t = sp_stats.ttest_ind(high["y"], low["y"])
        d = (high["y"].mean() - low["y"].mean()) / np.sqrt(
            ((len(high)-1)*high["y"].std()**2 + (len(low)-1)*low["y"].std()**2) / (len(high)+len(low)-2))
        print(f"\n  CL 20-60 (n={n2060}): high={len(high)}, low={len(low)}")
        print(f"    Mean slope: high={high['y'].mean():.3f}, low={low['y'].mean():.3f}")
        print(f"    Cohen's d = {d:.3f}, P = {p_t:.6f}")

    # ===== STEP 2: Refit on all ADNI + serialize =====
    print("\n" + "=" * 70)
    print("STEP 2: Final model refit (all ADNI, 8-ROI)")
    print("=" * 70)

    cov_model = LinearRegression().fit(cov, y)
    residuals = y - cov_model.predict(cov)
    scaler = StandardScaler().fit(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_enet = ElasticNetCV(l1_ratio=L1_RATIOS, cv=INNER_FOLDS,
                                   random_state=RANDOM_STATE, max_iter=10000, n_jobs=-1)
        final_enet.fit(scaler.transform(X), residuals)

    print(f"  Alpha = {final_enet.alpha_:.6f}, L1 = {final_enet.l1_ratio_}")
    print(f"  Non-zero = {np.sum(final_enet.coef_ != 0)}/{len(ROIS_8)}")
    for i, roi in enumerate(ROIS_8):
        if final_enet.coef_[i] != 0:
            print(f"    {roi:45s} {final_enet.coef_[i]:+.6f}")

    model_dict = {
        "enet": final_enet,
        "scaler": scaler,
        "cov_model": cov_model,
        "roi_names": ROIS_8,
        "covariates": COVARIATES,
        "outcome": PRIMARY_OUTCOME,
        "alpha": final_enet.alpha_,
        "l1_ratio": final_enet.l1_ratio_,
        "n_nonzero": int(np.sum(final_enet.coef_ != 0)),
        "random_state": RANDOM_STATE,
    }
    model_path = MODELS_DIR / "raps_8roi_final_model.joblib"
    joblib.dump(model_dict, model_path)
    print(f"  Saved: {model_path}")

    # ===== STEP 3: OASIS-3 validation =====
    print("\n" + "=" * 70)
    print("STEP 3: OASIS-3 external validation (8-ROI)")
    print("=" * 70)

    # Load OASIS-3 SUVR data from PUP
    pup_path = Path("/home/takumi/pj3-adni-roi/data/raw/OASIS3/OASIS3_data_files/"
                     "OASIS3_data_files/scans/PUP-PUP_output/resources/csv/files/OASIS3_PUP.csv")
    # Load existing individual RAPS file for covariates/slope
    oasis_meta = pd.read_csv(PROCESSED_DIR / "oasis3_individual_raps.csv")
    pup = pd.read_csv(pup_path)
    print(f"  PUP data: {pup.shape}")
    print(f"  OASIS-3 meta: {len(oasis_meta)}")

    # Map ROI names from PUP to ADNI
    for adni_name, oasis_name in OASIS3_ROI_MAP.items():
        if oasis_name in pup.columns:
            pup[adni_name] = pd.to_numeric(pup[oasis_name], errors="coerce")

    # Merge PUP SUVR with meta (slopes, covariates)
    # PUP uses 'OASISID' or similar ID
    id_col_pup = None
    for c in ['OASISID', 'oasisid', 'Subject', 'subject', 'PUP_PUPTIMECOURSEDATA ID']:
        if c in pup.columns:
            id_col_pup = c
            break
    if id_col_pup is None:
        # Try first column
        id_col_pup = pup.columns[0]
    print(f"  PUP ID column: {id_col_pup}")

    # Extract OASISID from PUP ID (may need parsing)
    pup["_merge_id"] = pup[id_col_pup].astype(str).str.extract(r'(OAS\d+)')[0]
    oasis_meta["_merge_id"] = oasis_meta["OASISID"].astype(str)

    # Keep only AV45 scans
    if "tracer" in pup.columns:
        pup_av45 = pup[pup["tracer"].str.contains("AV45|florbetapir|FBP", case=False, na=False)]
    elif "PET_tracer" in pup.columns:
        pup_av45 = pup[pup["PET_tracer"].str.contains("AV45|florbetapir|FBP", case=False, na=False)]
    else:
        # Try to filter by available columns
        pup_av45 = pup
    print(f"  PUP AV45 scans: {len(pup_av45)}")

    # Take first scan per subject
    pup_first = pup_av45.sort_values(id_col_pup).groupby("_merge_id").first().reset_index()

    oasis = oasis_meta.merge(pup_first[["_merge_id"] + ROIS_8], on="_merge_id", how="inner")
    print(f"  Merged OASIS-3 with SUVR: {len(oasis)}")

    oasis_sub = oasis.dropna(subset=ROIS_8 + ["CDRSB_slope"])
    print(f"  OASIS-3 with complete data: {len(oasis_sub)}")

    X_oasis = oasis_sub[ROIS_8].values.astype(float)
    y_oasis = oasis_sub["CDRSB_slope"].values.astype(float)

    # Get covariates for OASIS-3
    oasis_cov_cols = []
    for c in COVARIATES:
        if c in oasis_sub.columns:
            oasis_cov_cols.append(c)
        elif c == "PTGENDER_num" and "PTGENDER_num" not in oasis_sub.columns:
            if "sex_num" in oasis_sub.columns:
                oasis_sub["PTGENDER_num"] = oasis_sub["sex_num"]
                oasis_cov_cols.append("PTGENDER_num")
            elif "SEX" in oasis_sub.columns:
                oasis_sub["PTGENDER_num"] = (oasis_sub["SEX"] == "M").astype(int)
                oasis_cov_cols.append("PTGENDER_num")
        elif c == "CDRSB_bl" and "CDRSB_bl" not in oasis_sub.columns:
            if "CDRSB_baseline" in oasis_sub.columns:
                oasis_sub["CDRSB_bl"] = oasis_sub["CDRSB_baseline"]
                oasis_cov_cols.append("CDRSB_bl")

    if len(oasis_cov_cols) == len(COVARIATES):
        cov_oasis = oasis_sub[COVARIATES].values.astype(float)
        cov_pred = cov_model.predict(cov_oasis)
    else:
        print(f"  WARNING: Missing covariates: {set(COVARIATES) - set(oasis_cov_cols)}")
        print(f"  Using available: {oasis_cov_cols}")
        cov_pred = np.zeros(len(oasis_sub))

    # Compute RAPS
    X_oasis_scaled = scaler.transform(X_oasis)
    raps_oasis_raw = final_enet.predict(X_oasis_scaled)
    raps_oasis = (raps_oasis_raw - raps_oasis_raw.mean()) / raps_oasis_raw.std()

    # Get CL
    cl_col = None
    for c in ["CENTILOIDS", "Centiloid", "centiloid", "CL"]:
        if c in oasis_sub.columns:
            cl_col = c
            break
    if cl_col is None:
        print("  ERROR: No Centiloid column found")
        return

    cl_oasis = pd.to_numeric(oasis_sub[cl_col], errors="coerce").values
    valid = ~np.isnan(cl_oasis) & ~np.isnan(raps_oasis) & ~np.isnan(y_oasis)
    raps_v = raps_oasis[valid]
    cl_v = cl_oasis[valid]
    y_v = y_oasis[valid]
    n_v = int(valid.sum())

    print(f"  Valid for comparison: {n_v}")

    r_raps_o = sp_stats.pearsonr(raps_v, y_v)[0]
    r_cl_o = sp_stats.pearsonr(cl_v, y_v)[0]
    r_rc_o = sp_stats.pearsonr(raps_v, cl_v)[0]
    sz_o, sp_o = steiger_z(r_raps_o, r_cl_o, r_rc_o, n_v)

    print(f"\n  OASIS-3 RAPS vs CL:")
    print(f"    r(RAPS, slope) = {r_raps_o:.4f}")
    print(f"    r(CL, slope)   = {r_cl_o:.4f}")
    print(f"    Steiger z = {sz_o:.4f}, P = {sp_o:.6f}")

    aucs_o = compute_aucs(raps_v, y_v, cl_v, RAPID_DECLINE_THRESHOLDS)
    for thr, v in aucs_o.items():
        print(f"    AUC >{thr}: RAPS={v['auc_raps']:.4f}, CL={v['auc_cl']:.4f} (n_pos={v['n_pos']})")

    # OASIS-3 CL 20-60
    mask2060_o = (cl_v >= 20) & (cl_v < 60)
    n2060_o = int(mask2060_o.sum())
    if n2060_o > 10:
        raps_2060 = raps_v[mask2060_o]
        y_2060 = y_v[mask2060_o]
        # Use ADNI overall median as cutoff
        high_o = y_2060[raps_2060 >= np.median(raps_oasis)]
        low_o = y_2060[raps_2060 < np.median(raps_oasis)]
        if len(high_o) > 1 and len(low_o) > 1:
            t_o, p_o = sp_stats.ttest_ind(high_o, low_o)
            pooled_sd = np.sqrt(((len(high_o)-1)*high_o.std()**2 + (len(low_o)-1)*low_o.std()**2)
                                / (len(high_o)+len(low_o)-2))
            d_o = (high_o.mean() - low_o.mean()) / pooled_sd if pooled_sd > 0 else 0
            print(f"\n  OASIS-3 CL 20-60 (n={n2060_o}): high={len(high_o)}, low={len(low_o)}")
            print(f"    Mean slope: high={high_o.mean():.3f}, low={low_o.mean():.3f}")
            print(f"    Cohen's d = {d_o:.3f}, P = {p_o:.6f}")

    # ===== Save summary =====
    print("\n" + "=" * 70)
    print("SUMMARY: 8-ROI RAPS")
    print("=" * 70)

    summary = {
        "adni_n": len(pred_df),
        "adni_r_raps": round(r_raps, 4),
        "adni_r_cl": round(r_cl, 4),
        "adni_steiger_z": round(sz, 4),
        "adni_steiger_p": round(sp, 6),
        "adni_cv_r2_mean": round(perf_df["r2"].mean(), 4),
        "adni_cv_r2_sd": round(perf_df["r2"].std(), 4),
        "adni_cv_r_mean": round(perf_df["r"].mean(), 4),
        "oasis3_n": n_v,
        "oasis3_r_raps": round(r_raps_o, 4),
        "oasis3_r_cl": round(r_cl_o, 4),
        "oasis3_steiger_z": round(sz_o, 4),
        "oasis3_steiger_p": round(sp_o, 6),
    }

    for thr in RAPID_DECLINE_THRESHOLDS:
        if thr in aucs:
            summary[f"adni_auc_raps_{thr}"] = round(aucs[thr]["auc_raps"], 4)
            summary[f"adni_auc_cl_{thr}"] = round(aucs[thr]["auc_cl"], 4)
        if thr in aucs_o:
            summary[f"oasis3_auc_raps_{thr}"] = round(aucs_o[thr]["auc_raps"], 4)
            summary[f"oasis3_auc_cl_{thr}"] = round(aucs_o[thr]["auc_cl"], 4)

    out_path = TABLES_DIR / "raps_8roi_validation_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {out_path}")

    # Also save ADNI predictions
    pred_df.to_csv(TABLES_DIR / "raps_8roi_adni_predictions.csv", index=False)
    print(f"  Saved: {TABLES_DIR / 'raps_8roi_adni_predictions.csv'}")


if __name__ == "__main__":
    main()
