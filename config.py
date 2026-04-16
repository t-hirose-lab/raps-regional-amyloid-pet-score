"""
PJ3: RAPS Pipeline Configuration
全パイプライン共通設定
"""
from pathlib import Path

# ===== Directories =====
PROJECT_DIR = Path("/home/takumi/pj3-adni-roi")
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"

# ===== ROI Selection =====
# Desikan-Killiany atlas準拠
# 参照領域構成成分（WHOLECEREBELLUM, BRAINSTEM等）は除外
# CSF, VENTRICLE, WM_HYPOINTENSITIES等の非関心領域も除外
# Bilateral CTX (~34 regions) + Subcortical (~16 regions)

# NOTE: これらはUCBERKELEY_AMY_6MMデータの列名に対応
# SUVR値の列名は "{ROI}_SUVR" 形式
ANALYSIS_ROIS = [
    # --- Frontal Cortex ---
    "CTX_LH_CAUDALMIDDLEFRONTAL",
    "CTX_RH_CAUDALMIDDLEFRONTAL",
    "CTX_LH_LATERALORBITOFRONTAL",
    "CTX_RH_LATERALORBITOFRONTAL",
    "CTX_LH_MEDIALORBITOFRONTAL",
    "CTX_RH_MEDIALORBITOFRONTAL",
    "CTX_LH_ROSTRALMIDDLEFRONTAL",
    "CTX_RH_ROSTRALMIDDLEFRONTAL",
    "CTX_LH_SUPERIORFRONTAL",
    "CTX_RH_SUPERIORFRONTAL",
    "CTX_LH_FRONTALPOLE",
    "CTX_RH_FRONTALPOLE",
    "CTX_LH_PARSOPERCULARIS",
    "CTX_RH_PARSOPERCULARIS",
    "CTX_LH_PARSORBITALIS",
    "CTX_RH_PARSORBITALIS",
    "CTX_LH_PARSTRIANGULARIS",
    "CTX_RH_PARSTRIANGULARIS",
    "CTX_LH_PARACENTRAL",
    "CTX_RH_PARACENTRAL",
    "CTX_LH_PRECENTRAL",
    "CTX_RH_PRECENTRAL",
    # --- Parietal Cortex ---
    "CTX_LH_INFERIORPARIETAL",
    "CTX_RH_INFERIORPARIETAL",
    "CTX_LH_SUPERIORPARIETAL",
    "CTX_RH_SUPERIORPARIETAL",
    "CTX_LH_SUPRAMARGINAL",
    "CTX_RH_SUPRAMARGINAL",
    "CTX_LH_POSTCENTRAL",
    "CTX_RH_POSTCENTRAL",
    "CTX_LH_PRECUNEUS",
    "CTX_RH_PRECUNEUS",
    # --- Temporal Cortex ---
    "CTX_LH_ENTORHINAL",
    "CTX_RH_ENTORHINAL",
    "CTX_LH_FUSIFORM",
    "CTX_RH_FUSIFORM",
    "CTX_LH_INFERIORTEMPORAL",
    "CTX_RH_INFERIORTEMPORAL",
    "CTX_LH_MIDDLETEMPORAL",
    "CTX_RH_MIDDLETEMPORAL",
    "CTX_LH_SUPERIORTEMPORAL",
    "CTX_RH_SUPERIORTEMPORAL",
    "CTX_LH_TEMPORALPOLE",
    "CTX_RH_TEMPORALPOLE",
    "CTX_LH_TRANSVERSETEMPORAL",
    "CTX_RH_TRANSVERSETEMPORAL",
    "CTX_LH_BANKSSTS",
    "CTX_RH_BANKSSTS",
    "CTX_LH_PARAHIPPOCAMPAL",
    "CTX_RH_PARAHIPPOCAMPAL",
    # --- Occipital Cortex ---
    "CTX_LH_CUNEUS",
    "CTX_RH_CUNEUS",
    "CTX_LH_LATERALOCCIPITAL",
    "CTX_RH_LATERALOCCIPITAL",
    "CTX_LH_LINGUAL",
    "CTX_RH_LINGUAL",
    "CTX_LH_PERICALCARINE",
    "CTX_RH_PERICALCARINE",
    # --- Cingulate Cortex ---
    "CTX_LH_CAUDALANTERIORCINGULATE",
    "CTX_RH_CAUDALANTERIORCINGULATE",
    "CTX_LH_ISTHMUSCINGULATE",
    "CTX_RH_ISTHMUSCINGULATE",
    "CTX_LH_POSTERIORCINGULATE",
    "CTX_RH_POSTERIORCINGULATE",
    "CTX_LH_ROSTRALANTERIORCINGULATE",
    "CTX_RH_ROSTRALANTERIORCINGULATE",
    # --- Other Cortex ---
    "CTX_LH_INSULA",
    "CTX_RH_INSULA",
    # --- Subcortical ---
    "LEFT_THALAMUS_PROPER",
    "RIGHT_THALAMUS_PROPER",
    "LEFT_CAUDATE",
    "RIGHT_CAUDATE",
    "LEFT_PUTAMEN",
    "RIGHT_PUTAMEN",
    "LEFT_PALLIDUM",
    "RIGHT_PALLIDUM",
    "LEFT_HIPPOCAMPUS",
    "RIGHT_HIPPOCAMPUS",
    "LEFT_AMYGDALA",
    "RIGHT_AMYGDALA",
    "LEFT_ACCUMBENS_AREA",
    "RIGHT_ACCUMBENS_AREA",
]

# ROI列名（SUVR列名のリスト）
ANALYSIS_ROI_COLUMNS = [f"{roi}_SUVR" for roi in ANALYSIS_ROIS]

# DMNコンセンサス領域（文献ベース）
DMN_CORE_ROIS = [
    "CTX_LH_PRECUNEUS", "CTX_RH_PRECUNEUS",
    "CTX_LH_POSTERIORCINGULATE", "CTX_RH_POSTERIORCINGULATE",
    "CTX_LH_MEDIALORBITOFRONTAL", "CTX_RH_MEDIALORBITOFRONTAL",
    "CTX_LH_INFERIORPARIETAL", "CTX_RH_INFERIORPARIETAL",
    "LEFT_HIPPOCAMPUS", "RIGHT_HIPPOCAMPUS",
]

# ===== Covariates =====
COVARIATES = ["AGE", "PTGENDER_num", "PTEDUCAT", "APOE_E4_COUNT", "CDRSB_bl"]

# ===== Cross-Validation =====
OUTER_FOLDS = 5
INNER_FOLDS = 5
N_BOOTSTRAP = 1000
RANDOM_STATE = 42

# ===== Amyloid Positivity =====
CL_POSITIVE = 20       # Centiloid threshold for amyloid positivity
CL_EARLY_UPPER = 60    # Upper bound for "early" amyloid accumulation
CL_THRESHOLDS_SENSITIVITY = [50, 60, 70]  # Sensitivity analysis thresholds

# ===== Clinical Outcomes =====
PRIMARY_OUTCOME = "CDRSB_slope"
SECONDARY_OUTCOMES = ["MMSE_slope"]
RAPID_DECLINE_THRESHOLDS = [1.0, 1.5, 2.0]  # CDR-SB annual change thresholds

# ===== ElasticNet =====
L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9]

# ===== Reference Region Documentation =====
REFERENCE_REGION_DOC = """
Reference Region Information
============================
ROI SUVR: Composite Reference Region（全小脳 + 橋 + 浸食皮質下白質）で正規化
  - UCBerkeley pipeline uses composite reference (whole cerebellum + pons + eroded subcortical WM)
  - Column: COMPOSITE_REF_SUVR

Centiloid: 全小脳を使用した標準化スケール（Klunk et al., 2015）
  - Standardized across tracers (FBB/AV45)
  - Column: CENTILOIDS

注: 異なるトレーサー(FBB/AV45)間の比較にはCENTILOIDSを使用
注: PVE補正は適用されていない（UCBerkeley pipeline標準出力）
  → Limitation sectionで議論すべき事項
"""

# ===== Subgroup Analysis =====
APOE_GROUPS = {
    "noncarrier": 0,    # E4非保有
    "carrier": 1,       # E4保有（≥1アレル）
}

CLINICAL_STAGES = ["CN", "MCI"]

# ===== Plotting =====
FIGURE_DPI = 300
FIGURE_FORMAT = "png"
COLOR_PALETTE = {
    "raps": "#E74C3C",      # Red for RAPS
    "cl": "#3498DB",        # Blue for CL
    "fast": "#E74C3C",      # Red for fast decliners
    "slow": "#2ECC71",      # Green for slow decliners
    "normal": "#95A5A6",    # Gray for normal
}
