#!/usr/bin/env python3
"""Generate positive/negative NIfTI maps for 8-ROI RAPS (MRIcroGL用).

Outputs:
  - submit_v2/figures/raps_8roi_positive.nii  : positive coefficient ROIs (red in MRIcroGL)
  - submit_v2/figures/raps_8roi_negative.nii  : negative coefficient ROIs (blue in MRIcroGL)

8 ROIs (final model ∩ bootstrap ≥80%):
  Positive (5): LH transverse temporal, LH fusiform, LH lingual, L amygdala, L putamen
  Negative (3): L hippocampus, R pallidum, RH entorhinal
"""
import logging
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from nilearn.datasets import fetch_atlas_destrieux_2009, fetch_atlas_harvard_oxford
from nilearn.image import resample_to_img

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "submit_v2" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 8-ROI final model coefficients (from raps_8roi_final_model.joblib refit)
ROI_COEFS = {
    "LEFT_HIPPOCAMPUS":            -0.161705,
    "CTX_LH_TRANSVERSETEMPORAL":   +0.057244,
    "CTX_LH_FUSIFORM":            +0.056109,
    "RIGHT_PALLIDUM":              -0.061204,
    "CTX_LH_LINGUAL":             +0.042558,
    "CTX_RH_ENTORHINAL":          -0.057477,
    "LEFT_AMYGDALA":               +0.059904,
    "LEFT_PUTAMEN":                +0.061792,
}

# Atlas mappings (same as generate_score_nifti.py)
DESTRIEUX_MAP = {
    "CTX_LH_TRANSVERSETEMPORAL": "L G_temp_sup-G_T_transv",
    "CTX_LH_FUSIFORM":          "L G_oc-temp_lat-fusifor",
    "CTX_LH_LINGUAL":           "L G_oc-temp_med-Lingual",
    "CTX_RH_ENTORHINAL":        "R G_oc-temp_med-Parahip",
}

HARVARD_OXFORD_MAP = {
    "LEFT_HIPPOCAMPUS":  ("Left Hippocampus", 9),
    "RIGHT_PALLIDUM":    ("Right Pallidum", 18),
    "LEFT_AMYGDALA":     ("Left Amygdala", 10),
    "LEFT_PUTAMEN":      ("Left Putamen", 6),
}


def main():
    logger.info("=" * 60)
    logger.info("8-ROI RAPS NIfTI Generator (positive/negative split)")
    logger.info("=" * 60)

    # Load atlases
    logger.info("Loading atlases...")
    destrieux = fetch_atlas_destrieux_2009()
    destrieux_img = destrieux["maps"] if isinstance(destrieux["maps"], nib.Nifti1Image) else nib.load(str(destrieux["maps"]))
    destrieux_labels = [lb.decode() if isinstance(lb, bytes) else str(lb) for lb in destrieux["labels"]]

    ho_sub = fetch_atlas_harvard_oxford("sub-maxprob-thr25-1mm")
    ho_img = ho_sub["maps"] if isinstance(ho_sub["maps"], nib.Nifti1Image) else nib.load(str(ho_sub["maps"]))
    ho_labels = list(ho_sub["labels"])

    # Reference volume
    ref_data = np.zeros(ho_img.shape, dtype=np.float32)
    ref_img = nib.Nifti1Image(ref_data, ho_img.affine, ho_img.header)

    # Resample Destrieux to reference
    logger.info("Resampling Destrieux atlas...")
    destrieux_resampled = resample_to_img(destrieux_img, ref_img, interpolation="nearest")
    destrieux_data = np.asarray(destrieux_resampled.dataobj, dtype=np.int32)
    ho_data = np.asarray(ho_img.dataobj, dtype=np.int32)

    # Build full coefficient map
    score_data = np.zeros(ref_img.shape, dtype=np.float32)

    # Cortical ROIs (Destrieux)
    for roi_name, destrieux_label in DESTRIEUX_MAP.items():
        coef = ROI_COEFS[roi_name]
        try:
            label_idx = destrieux_labels.index(destrieux_label)
        except ValueError:
            logger.error("Label not found: %s", destrieux_label)
            sys.exit(1)
        mask = destrieux_data == label_idx
        score_data[mask] = coef
        logger.info("  %s -> %d voxels, coef=%+.4f", roi_name, int(mask.sum()), coef)

    # Subcortical ROIs (Harvard-Oxford, overwrite)
    for roi_name, (ho_label, expected_idx) in HARVARD_OXFORD_MAP.items():
        coef = ROI_COEFS[roi_name]
        try:
            label_idx = ho_labels.index(ho_label)
        except ValueError:
            logger.error("Label not found: %s", ho_label)
            sys.exit(1)
        mask = ho_data == label_idx
        score_data[mask] = coef
        logger.info("  %s -> %d voxels, coef=%+.4f", roi_name, int(mask.sum()), coef)

    # Split into positive and negative
    pos_data = np.where(score_data > 0, score_data, 0).astype(np.float32)
    neg_data = np.where(score_data < 0, np.abs(score_data), 0).astype(np.float32)

    # Save
    pos_img = nib.Nifti1Image(pos_data, ref_img.affine, ref_img.header)
    neg_img = nib.Nifti1Image(neg_data, ref_img.affine, ref_img.header)

    out_pos = OUTPUT_DIR / "raps_8roi_positive.nii"
    out_neg = OUTPUT_DIR / "raps_8roi_negative.nii"

    nib.save(pos_img, str(out_pos))
    nib.save(neg_img, str(out_neg))

    logger.info("")
    logger.info("Positive (red): %s", out_pos)
    logger.info("  ROIs: LH transverse temporal, LH fusiform, LH lingual, L amygdala, L putamen")
    logger.info("  Value range: [%.4f, %.4f]", pos_data.min(), pos_data.max())
    logger.info("  Non-zero voxels: %d", int((pos_data > 0).sum()))

    logger.info("")
    logger.info("Negative (blue): %s", out_neg)
    logger.info("  ROIs: L hippocampus, R pallidum, RH entorhinal")
    logger.info("  Value range: [%.4f, %.4f]", neg_data.min(), neg_data.max())
    logger.info("  Non-zero voxels: %d", int((neg_data > 0).sum()))

    logger.info("")
    logger.info("MRIcroGL usage:")
    logger.info("  1. Open MNI152 template")
    logger.info("  2. Add overlay: raps_8roi_positive.nii (colormap: red/warm)")
    logger.info("  3. Add overlay: raps_8roi_negative.nii (colormap: blue/cool)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
