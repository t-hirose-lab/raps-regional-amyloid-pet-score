#!/usr/bin/env python3
"""
04_raps_construction.py
=======================
Regional Amyloid PET Score (RAPS) construction using Nested Cross-Validation.
Prevents data leakage by performing all preprocessing within CV folds.

Key design decisions:
- Nested CV (5x5): outer for performance estimation, inner for hyperparameter tuning
- Covariate regression within folds: Age, Sex, Education, APOE, baseline CDR-SB
- CL NOT included in covariates -> enables direct RAPS vs CL comparison
- ElasticNet with l1_ratio grid search for optimal sparsity

References:
- Varma & Simon (2006) - Bias in error estimation with nested CV
- Zou & Hastie (2005) - ElasticNet regularization

Input:  data/processed/individual_slopes.csv
Output: data/processed/raps_scores.csv
        results/tables/nested_cv_results.csv
        results/tables/raps_roi_weights.csv
        results/models/raps_final_model.joblib
        results/figures/fig_nested_cv_performance.png
        results/figures/fig_raps_roi_weights.png
        results/figures/fig_raps_vs_outcome.png
        results/figures/fig_raps_predicted_vs_actual.png

Usage:
    python scripts/04_raps_construction.py
    python scripts/04_raps_construction.py --dry-run
"""

import sys
import argparse
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ---------------------------------------------------------------------------
# config import
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    PROCESSED_DIR, RESULTS_DIR, TABLES_DIR, FIGURES_DIR, MODELS_DIR,
    ANALYSIS_ROI_COLUMNS, COVARIATES,
    OUTER_FOLDS, INNER_FOLDS, L1_RATIOS, RANDOM_STATE,
    PRIMARY_OUTCOME, FIGURE_DPI, COLOR_PALETTE,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# matplotlib / seaborn
# ---------------------------------------------------------------------------
sns.set_style("whitegrid")
plt.rcParams["font.family"] = "DejaVu Sans"


# ===========================================================================
# Data loading & validation
# ===========================================================================

def load_and_validate(input_path: Path) -> tuple[pd.DataFrame, list[str]]:
    """individual_slopes.csv を読み込み、必要カラムの存在を確認する。"""
    logger.info("Loading data: %s", input_path)
    df = pd.read_csv(input_path)
    logger.info("  Rows: %d, Cols: %d", len(df), len(df.columns))

    # --- Required columns ---
    required = ["RID", PRIMARY_OUTCOME] + COVARIATES
    missing_req = [c for c in required if c not in df.columns]
    if missing_req:
        raise ValueError(f"Missing required columns: {missing_req}")

    # --- ROI SUVR columns ---
    available_rois = [c for c in ANALYSIS_ROI_COLUMNS if c in df.columns]
    # Remove columns that cannot be converted to numeric (e.g. date strings)
    non_numeric = []
    for c in available_rois:
        converted = pd.to_numeric(df[c], errors="coerce")
        if converted.isna().all() and df[c].notna().any():
            non_numeric.append(c)
        else:
            df[c] = converted
    if non_numeric:
        logger.warning("  Removed %d non-numeric ROI columns: %s",
                        len(non_numeric), non_numeric)
        available_rois = [c for c in available_rois if c not in non_numeric]
    missing_rois = set(ANALYSIS_ROI_COLUMNS) - set(available_rois)
    if missing_rois:
        logger.warning("  Missing %d ROI columns (using %d available)",
                        len(missing_rois), len(available_rois))
    if len(available_rois) == 0:
        raise ValueError("No ROI SUVR columns found in data")

    logger.info("  Available ROI columns: %d / %d",
                len(available_rois), len(ANALYSIS_ROI_COLUMNS))
    return df, available_rois


def prepare_analysis_data(
    df: pd.DataFrame,
    available_rois: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Listwise deletion of missing values, then return arrays.

    Returns
    -------
    X_roi : ROI SUVR matrix (n, p)
    y : outcome vector (n,)
    cov_matrix : covariate matrix (n, q)
    rids : RID array (n,)
    roi_names : list of ROI column names used
    """
    cols_needed = ["RID", PRIMARY_OUTCOME] + COVARIATES + available_rois
    subset = df[cols_needed].copy()
    n_before = len(subset)
    subset = subset.dropna()
    n_after = len(subset)
    n_dropped = n_before - n_after
    if n_dropped > 0:
        logger.warning("  Dropped %d rows with missing values (%d -> %d)",
                        n_dropped, n_before, n_after)
    else:
        logger.info("  No missing values (N=%d)", n_after)

    rids = subset["RID"].values
    y = subset[PRIMARY_OUTCOME].values.astype(float)
    cov_matrix = subset[COVARIATES].values.astype(float)
    X_roi = subset[available_rois].values.astype(float)

    return X_roi, y, cov_matrix, rids, available_rois


# ===========================================================================
# Nested Cross-Validation
# ===========================================================================

def run_nested_cv(
    X_roi: np.ndarray,
    y: np.ndarray,
    cov_matrix: np.ndarray,
    rids: np.ndarray,
    roi_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Nested CV (outer 5 x inner 5) for RAPS construction.

    Returns
    -------
    predictions_df : per-subject RAPS scores with fold info
    performance_df : per-fold performance metrics
    """
    logger.info("=" * 60)
    logger.info("Nested Cross-Validation (%d x %d)", OUTER_FOLDS, INNER_FOLDS)
    logger.info("=" * 60)

    outer_kfold = KFold(
        n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE,
    )

    outer_predictions = []
    outer_performance = []

    for fold_i, (train_idx, test_idx) in enumerate(outer_kfold.split(X_roi)):
        logger.info("  Outer fold %d/%d  (train=%d, test=%d)",
                     fold_i + 1, OUTER_FOLDS, len(train_idx), len(test_idx))

        X_train = X_roi[train_idx]
        X_test = X_roi[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        cov_train = cov_matrix[train_idx]
        cov_test = cov_matrix[test_idx]

        # ----- Step 2a: Covariate regression (fit on train) -----
        cov_model = LinearRegression()
        cov_model.fit(cov_train, y_train)
        residual_train = y_train - cov_model.predict(cov_train)
        residual_test = y_test - cov_model.predict(cov_test)

        cov_r2 = cov_model.score(cov_train, y_train)
        logger.info("    Covariate model R2 (train): %.4f", cov_r2)

        # ----- Step 2b: ROI scaling (fit on train) -----
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # ----- Step 2c: Inner CV for hyperparameter tuning -----
        inner_kfold = KFold(
            n_splits=INNER_FOLDS,
            shuffle=True,
            random_state=RANDOM_STATE + fold_i,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            enet = ElasticNetCV(
                l1_ratio=L1_RATIOS,
                cv=inner_kfold,
                random_state=RANDOM_STATE,
                max_iter=10000,
                n_jobs=-1,
            )
            enet.fit(X_train_scaled, residual_train)

        logger.info("    ElasticNet: alpha=%.6f, l1_ratio=%.2f, non-zero=%d/%d",
                     enet.alpha_, enet.l1_ratio_,
                     np.sum(enet.coef_ != 0), len(roi_names))

        # ----- Step 2d: Predict on outer test set -----
        pred_test = enet.predict(X_test_scaled)

        # ----- Step 2e: Performance metrics -----
        r2 = r2_score(residual_test, pred_test)
        pearson_r = float(np.corrcoef(residual_test, pred_test)[0, 1])
        mae = mean_absolute_error(residual_test, pred_test)

        logger.info("    Performance: R2=%.4f, r=%.4f, MAE=%.4f",
                     r2, pearson_r, mae)

        outer_performance.append({
            "fold": fold_i + 1,
            "r2": r2,
            "pearson_r": pearson_r,
            "mae": mae,
            "alpha": enet.alpha_,
            "l1_ratio": enet.l1_ratio_,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "n_nonzero_coefs": int(np.sum(enet.coef_ != 0)),
            "cov_r2_train": cov_r2,
        })

        # Per-subject predictions
        for idx, pred in zip(test_idx, pred_test):
            outer_predictions.append({
                "RID": rids[idx],
                "raps_raw": float(pred),
                "residual_actual": float(y[idx] - cov_model.predict(cov_matrix[idx:idx+1])[0]),
                "fold": fold_i + 1,
            })

    # Build DataFrames
    predictions_df = pd.DataFrame(outer_predictions)
    performance_df = pd.DataFrame(outer_performance)

    # Z-score standardize RAPS
    raps_mean = predictions_df["raps_raw"].mean()
    raps_std = predictions_df["raps_raw"].std()
    predictions_df["RAPS"] = (
        (predictions_df["raps_raw"] - raps_mean) / raps_std
    )

    # Summary
    logger.info("-" * 60)
    logger.info("Nested CV summary:")
    logger.info("  R2:  %.4f +/- %.4f",
                performance_df["r2"].mean(), performance_df["r2"].std())
    logger.info("  r:   %.4f +/- %.4f",
                performance_df["pearson_r"].mean(),
                performance_df["pearson_r"].std())
    logger.info("  MAE: %.4f +/- %.4f",
                performance_df["mae"].mean(), performance_df["mae"].std())
    logger.info("-" * 60)

    return predictions_df, performance_df


# ===========================================================================
# Final model (full data refit)
# ===========================================================================

def fit_final_model(
    X_roi: np.ndarray,
    y: np.ndarray,
    cov_matrix: np.ndarray,
    roi_names: list[str],
) -> tuple[dict, pd.DataFrame]:
    """
    Refit on all data to obtain the final model and ROI weights.

    Returns
    -------
    model_bundle : dict with enet, scaler, cov_model, roi_names, metadata
    weights_df : ROI weight table
    """
    logger.info("=" * 60)
    logger.info("Final model (all data refit)")
    logger.info("=" * 60)

    # Covariate regression
    cov_model = LinearRegression()
    cov_model.fit(cov_matrix, y)
    residuals = y - cov_model.predict(cov_matrix)
    logger.info("  Covariate model R2: %.4f", cov_model.score(cov_matrix, y))

    # Scale ROIs
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_roi)

    # ElasticNetCV on all data
    inner_kfold = KFold(
        n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        enet = ElasticNetCV(
            l1_ratio=L1_RATIOS,
            cv=inner_kfold,
            random_state=RANDOM_STATE,
            max_iter=10000,
            n_jobs=-1,
        )
        enet.fit(X_scaled, residuals)

    n_nonzero = int(np.sum(enet.coef_ != 0))
    logger.info("  Final ElasticNet: alpha=%.6f, l1_ratio=%.2f, non-zero=%d/%d",
                enet.alpha_, enet.l1_ratio_, n_nonzero, len(roi_names))

    # ROI weights table
    weights_df = pd.DataFrame({
        "roi": roi_names,
        "coefficient": enet.coef_,
        "abs_coefficient": np.abs(enet.coef_),
    })
    weights_df = weights_df.sort_values(
        "abs_coefficient", ascending=False,
    ).reset_index(drop=True)
    weights_df["rank"] = range(1, len(weights_df) + 1)

    # Model bundle for saving
    model_bundle = {
        "enet": enet,
        "scaler": scaler,
        "cov_model": cov_model,
        "roi_names": roi_names,
        "covariates": COVARIATES,
        "outcome": PRIMARY_OUTCOME,
        "alpha": enet.alpha_,
        "l1_ratio": enet.l1_ratio_,
        "n_nonzero": n_nonzero,
        "random_state": RANDOM_STATE,
    }

    return model_bundle, weights_df


# ===========================================================================
# Visualization
# ===========================================================================

def plot_nested_cv_performance(performance_df: pd.DataFrame, out_path: Path):
    """Bar plot of R2 per outer fold."""
    fig, ax = plt.subplots(figsize=(8, 5))

    folds = performance_df["fold"].values
    r2_vals = performance_df["r2"].values
    mean_r2 = r2_vals.mean()

    bars = ax.bar(folds, r2_vals, color=COLOR_PALETTE.get("raps", "#E74C3C"),
                  edgecolor="black", linewidth=0.5, width=0.6)
    ax.axhline(mean_r2, color="black", linestyle="--", linewidth=1,
               label=f"Mean R2 = {mean_r2:.4f}")

    ax.set_xlabel("Outer Fold", fontsize=12)
    ax.set_ylabel("R2", fontsize=12)
    ax.set_title("Nested CV Performance (R2 per Outer Fold)", fontsize=13)
    ax.set_xticks(folds)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, val in zip(bars, r2_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Figure saved: %s", out_path)


def plot_roi_weights(weights_df: pd.DataFrame, out_path: Path, top_n: int = 20):
    """Horizontal bar plot of top ROI coefficients."""
    df_plot = weights_df.head(top_n).copy()
    # Filter to non-zero only
    df_plot = df_plot[df_plot["coefficient"] != 0].copy()
    if df_plot.empty:
        logger.warning("  No non-zero ROI weights to plot")
        return

    df_plot = df_plot.sort_values("coefficient", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, max(4, len(df_plot) * 0.4)))

    colors = ["#2166ac" if c > 0 else "#d6604d" for c in df_plot["coefficient"]]
    ax.barh(range(len(df_plot)), df_plot["coefficient"],
            color=colors, height=0.6, edgecolor="black", linewidth=0.3)
    ax.set_yticks(range(len(df_plot)))
    ax.set_yticklabels(df_plot["roi"], fontsize=8)
    ax.axvline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("ElasticNet Coefficient (Final Model)", fontsize=11)
    ax.set_title(f"Top {len(df_plot)} ROI Weights for RAPS", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2166ac", label="Positive (more amyloid -> faster decline)"),
        Patch(facecolor="#d6604d", label="Negative (more amyloid -> slower decline)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Figure saved: %s", out_path)


def plot_raps_vs_outcome(
    predictions_df: pd.DataFrame,
    df_orig: pd.DataFrame,
    out_path: Path,
):
    """Scatter plot of RAPS score vs CDR-SB slope."""
    merged = predictions_df.merge(
        df_orig[["RID", PRIMARY_OUTCOME]],
        on="RID",
        how="left",
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(merged["RAPS"], merged[PRIMARY_OUTCOME],
               alpha=0.4, s=20, color=COLOR_PALETTE.get("raps", "#E74C3C"),
               edgecolors="none")

    # Fit line
    mask = merged[["RAPS", PRIMARY_OUTCOME]].dropna().index
    x = merged.loc[mask, "RAPS"].values
    y_vals = merged.loc[mask, PRIMARY_OUTCOME].values
    if len(x) > 2:
        z = np.polyfit(x, y_vals, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), "k--", linewidth=1.5, alpha=0.7)

        from scipy import stats
        r, pval = stats.pearsonr(x, y_vals)
        ax.set_title(f"RAPS vs {PRIMARY_OUTCOME}\n(r = {r:.3f}, p = {pval:.2e})",
                     fontsize=12)
    else:
        ax.set_title(f"RAPS vs {PRIMARY_OUTCOME}", fontsize=12)

    ax.set_xlabel("RAPS Score (z-scored)", fontsize=11)
    ax.set_ylabel(PRIMARY_OUTCOME, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Figure saved: %s", out_path)


def plot_predicted_vs_actual(predictions_df: pd.DataFrame, out_path: Path):
    """Scatter plot of nested CV predicted vs actual residuals."""
    fig, ax = plt.subplots(figsize=(6, 6))

    x = predictions_df["residual_actual"].values
    y_pred = predictions_df["raps_raw"].values

    ax.scatter(x, y_pred, alpha=0.4, s=20,
               color=COLOR_PALETTE.get("raps", "#E74C3C"),
               edgecolors="none")

    # Identity line
    all_vals = np.concatenate([x, y_pred])
    lo, hi = np.nanmin(all_vals), np.nanmax(all_vals)
    margin = (hi - lo) * 0.05
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
            "k--", linewidth=1, alpha=0.5, label="Identity line")

    from scipy import stats
    mask = ~(np.isnan(x) | np.isnan(y_pred))
    if mask.sum() > 2:
        r, pval = stats.pearsonr(x[mask], y_pred[mask])
        r2 = r2_score(x[mask], y_pred[mask])
        ax.set_title(
            f"Nested CV: Predicted vs Actual Residuals\n"
            f"(r = {r:.3f}, R2 = {r2:.3f})",
            fontsize=11,
        )
    else:
        ax.set_title("Nested CV: Predicted vs Actual Residuals", fontsize=11)

    ax.set_xlabel("Actual Residual (covariate-adjusted)", fontsize=10)
    ax.set_ylabel("Predicted Residual (RAPS raw)", fontsize=10)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Figure saved: %s", out_path)


# ===========================================================================
# Dummy data (--dry-run)
# ===========================================================================

def generate_dummy_data(n: int = 200) -> pd.DataFrame:
    """Generate dummy individual_slopes data for testing."""
    logger.info("[DRY-RUN] Generating dummy data (n=%d)", n)
    rng = np.random.default_rng(RANDOM_STATE)

    df = pd.DataFrame({
        "RID": 1000 + np.arange(n),
        "AGE": rng.normal(72, 7, n),
        "PTGENDER_num": rng.choice([0, 1], n),
        "PTEDUCAT": rng.integers(12, 21, n).astype(float),
        "APOE_E4_COUNT": rng.choice([0, 1, 2], n, p=[0.5, 0.35, 0.15]).astype(float),
        "CDRSB_bl": np.maximum(0, rng.normal(1.5, 2.0, n)),
        "CENTILOIDS": np.maximum(0, rng.normal(50, 35, n)),
    })

    # Outcome: CDR-SB slope with real-ish structure
    true_age_effect = 0.01 * (df["AGE"] - 72)
    true_apoe_effect = 0.3 * df["APOE_E4_COUNT"]
    noise = rng.normal(0, 0.5, n)
    df[PRIMARY_OUTCOME] = 0.5 + true_age_effect + true_apoe_effect + noise

    # Dummy ROI SUVR columns with embedded signal
    for i, roi in enumerate(ANALYSIS_ROI_COLUMNS):
        base = rng.normal(1.2, 0.25, n)
        # First few ROIs carry signal
        if i < 5:
            base += 0.1 * df[PRIMARY_OUTCOME].values
        df[roi] = np.maximum(0.8, base)

    return df


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="RAPS construction via Nested Cross-Validation"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run with dummy data for testing",
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Input CSV path (default: data/processed/individual_slopes.csv)",
    )
    args = parser.parse_args()

    logger.info("=" * 72)
    logger.info("04_raps_construction.py: RAPS via Nested CV")
    logger.info("=" * 72)

    # --- Ensure output directories ---
    for d in [PROCESSED_DIR, TABLES_DIR, FIGURES_DIR, MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    if args.dry_run:
        df = generate_dummy_data()
        available_rois = [c for c in ANALYSIS_ROI_COLUMNS if c in df.columns]
    else:
        input_path = Path(args.input) if args.input else PROCESSED_DIR / "individual_slopes.csv"
        if not input_path.exists():
            logger.error("Input file not found: %s", input_path)
            logger.error("Run 03_lme_slopes.R first.")
            sys.exit(1)
        df, available_rois = load_and_validate(input_path)

    # --- Prepare arrays ---
    X_roi, y, cov_matrix, rids, roi_names = prepare_analysis_data(df, available_rois)
    logger.info("Analysis data: N=%d, ROIs=%d, Covariates=%d",
                len(y), X_roi.shape[1], cov_matrix.shape[1])

    # --- Step 2: Nested CV ---
    predictions_df, performance_df = run_nested_cv(
        X_roi, y, cov_matrix, rids, roi_names,
    )

    # --- Step 4: Final model (all data) ---
    model_bundle, weights_df = fit_final_model(
        X_roi, y, cov_matrix, roi_names,
    )

    # ==== Save outputs ====
    logger.info("=" * 60)
    logger.info("Saving outputs")
    logger.info("=" * 60)

    # 1. RAPS scores
    raps_out = predictions_df[["RID", "RAPS", "fold"]].copy()
    raps_path = PROCESSED_DIR / "raps_scores.csv"
    raps_out.to_csv(raps_path, index=False, float_format="%.6f")
    logger.info("  Saved: %s (%d subjects)", raps_path, len(raps_out))

    # 2. Nested CV results
    cv_path = TABLES_DIR / "nested_cv_results.csv"
    performance_df.to_csv(cv_path, index=False, float_format="%.6f")
    logger.info("  Saved: %s", cv_path)

    # 3. ROI weights
    weights_path = TABLES_DIR / "raps_roi_weights.csv"
    weights_df.to_csv(weights_path, index=False, float_format="%.6f")
    logger.info("  Saved: %s (%d non-zero / %d total)",
                weights_path,
                int((weights_df["coefficient"] != 0).sum()),
                len(weights_df))

    # 4. Final model
    model_path = MODELS_DIR / "raps_final_model.joblib"
    joblib.dump(model_bundle, model_path)
    logger.info("  Saved: %s", model_path)

    # ==== Figures ====
    logger.info("=" * 60)
    logger.info("Generating figures")
    logger.info("=" * 60)

    plot_nested_cv_performance(
        performance_df,
        FIGURES_DIR / "fig_nested_cv_performance.png",
    )

    plot_roi_weights(
        weights_df,
        FIGURES_DIR / "fig_raps_roi_weights.png",
    )

    plot_raps_vs_outcome(
        predictions_df,
        df,
        FIGURES_DIR / "fig_raps_vs_outcome.png",
    )

    plot_predicted_vs_actual(
        predictions_df,
        FIGURES_DIR / "fig_raps_predicted_vs_actual.png",
    )

    # ==== Done ====
    logger.info("=" * 72)
    logger.info("04_raps_construction.py COMPLETE")
    logger.info("  RAPS scores: %s", raps_path)
    logger.info("  CV results:  %s", cv_path)
    logger.info("  ROI weights: %s", weights_path)
    logger.info("  Final model: %s", model_path)
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
