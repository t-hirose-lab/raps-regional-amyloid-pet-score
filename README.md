# RAPS - Regional Amyloid PET Score

Code and model weights for computing the Regional Amyloid PET Score (RAPS), a data-driven 8-ROI composite that predicts cognitive decline in amyloid-positive individuals beyond global Centiloid.

## Reference

> Hirose T, et al. **A Data-Driven Regional Amyloid PET Score Predicts Cognitive Decline Beyond Centiloid.** *Alzheimer's & Dementia* (submitted).

## What is RAPS?

RAPS is a weighted sum of regional amyloid PET SUVRs from **8 FreeSurfer Desikan-Killiany ROIs**, selected by ElasticNet regression with bootstrap stability assessment (ElasticNet non-zero AND bootstrap selection frequency >= 80%). It outperforms global Centiloid in predicting cognitive decline (CDR-SB slope), particularly in the intermediate amyloid burden range (CL 20-60).

### The 8 RAPS ROIs

| ROI | Coefficient (beta) | Bootstrap Stability |
|-----|-------------------|-------------------|
| Left Hippocampus | -0.162 | 100.0% |
| Left Putamen | +0.062 | 80.5% |
| Right Pallidum | -0.061 | 90.4% |
| Left Amygdala | +0.060 | 86.3% |
| Left Transverse Temporal | +0.057 | 93.1% |
| Right Entorhinal | -0.057 | 87.3% |
| Left Fusiform | +0.056 | 91.0% |
| Left Lingual | +0.043 | 88.8% |

## Key Results

| Metric | RAPS | Centiloid | P |
|--------|------|-----------|---|
| Correlation with CDR-SB slope (ADNI, N=515) | r = 0.585 | r = 0.398 | < .001 (Steiger) |
| AUC for rapid decline >1.5/yr (ADNI) | 0.861 | 0.737 | < .001 (DeLong) |
| Incremental C-index (Clinical + RAPS) | +0.039 | +0.024 (CL) | - |
| CL 20-60 subgroup effect (ADNI, n=173) | Cohen's d = 1.25 | - | < .001 |
| AUC for rapid decline >1.5/yr (OASIS-3, N=429) | 0.881 | 0.807 | - |
| CL 20-60 subgroup effect (OASIS-3, n=87) | Cohen's d = 0.59 | - | .008 |

## Repository Structure

```
raps-regional-amyloid-pet-score/
├── README.md
├── LICENSE
├── requirements.txt
├── config.py                          # Pipeline configuration
├── scripts/
│   ├── raps_construction.py           # Nested 5x5 CV for RAPS (8-ROI)
│   ├── raps_validation_oasis3.py      # External validation in OASIS-3
│   ├── raps_supplementary_stats.py    # Supplementary table statistics
│   └── generate_nifti.py             # NIfTI maps for MRIcroGL
├── figures/
│   └── make_fig*.py                   # Figure generation scripts
├── model/
│   └── raps_8roi_final_model.joblib   # Trained model (ElasticNet + scaler)
└── data/
    ├── raps_roi_weights.csv           # 8-ROI coefficients
    └── bootstrap_roi_stability.csv    # Bootstrap stability (80 ROIs)
```

## Computing RAPS for a New Cohort

```python
import joblib
import numpy as np

# Load the trained model
model = joblib.load('model/raps_8roi_final_model.joblib')
enet = model['enet']           # ElasticNet with 8-ROI coefficients
scaler = model['scaler']       # StandardScaler (ADNI-fitted)
cov_model = model['cov_model'] # Covariate regression model

# For each participant, provide 8 ROI SUVRs in this order:
roi_names = [
    'LEFT_HIPPOCAMPUS_SUVR',
    'CTX_LH_TRANSVERSETEMPORAL_SUVR',
    'CTX_LH_FUSIFORM_SUVR',
    'RIGHT_PALLIDUM_SUVR',
    'CTX_LH_LINGUAL_SUVR',
    'CTX_RH_ENTORHINAL_SUVR',
    'LEFT_AMYGDALA_SUVR',
    'LEFT_PUTAMEN_SUVR',
]

# X_roi: (N, 8) array of SUVR values
X_scaled = scaler.transform(X_roi)
raps_raw = enet.predict(X_scaled)
raps_z = (raps_raw - raps_raw.mean()) / raps_raw.std()
```

**Note**: Site-specific recalibration of the StandardScaler may improve performance while keeping the ElasticNet weights fixed (see Supplementary Table S16 in the manuscript).

## Requirements

- Python >= 3.10
- scikit-learn >= 1.7
- numpy, pandas, scipy, statsmodels
- lifelines (for survival analysis)
- matplotlib, seaborn (for figures)
- nibabel, nilearn (for NIfTI generation)

See `requirements.txt` for exact versions.

## Data Availability

- **ADNI**: Available at [adni.loni.usc.edu](https://adni.loni.usc.edu) upon approval of a data use agreement
- **OASIS-3**: Available at [nitrc.org/ir/](https://www.nitrc.org/ir/) upon request

## License

MIT License

## Contact

Takumi Hirose, MD, PhD
Department of Psychiatry, Juntendo University School of Medicine
t-hirose@juntendo.ac.jp
