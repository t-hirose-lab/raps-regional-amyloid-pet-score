# RAPS - Regional Amyloid PET Score

Code and model weights for computing the Regional Amyloid PET Score (RAPS), a data-driven 7-ROI composite that predicts cognitive decline in amyloid-positive individuals beyond global Centiloid.

## Reference

> Hirose T, Akamatsu W, Kato T. **A Data-Driven Regional Amyloid PET Score Predicts Cognitive Decline Beyond Centiloid.** *Alzheimer's & Dementia* (submitted).

## What is RAPS?

RAPS is a weighted sum of regional amyloid PET SUVRs from **7 FreeSurfer Desikan-Killiany ROIs**, selected by ElasticNet regression with 1,000-iteration bootstrap stability assessment (ElasticNet non-zero AND bootstrap selection frequency ≥ 80%) on 82 ROIs in 515 amyloid-positive ADNI participants. It outperforms global Centiloid in predicting cognitive decline (CDR-SB slope), particularly in the intermediate amyloid burden range (CL 20–60).

### The 7 RAPS ROIs

| ROI | Coefficient (β) | Bootstrap Stability |
|-----|-------------------|-------------------|
| Left Hippocampus | −0.186 | 100.0% |
| Left Amygdala | +0.095 | 84.8% |
| Left Transverse Temporal | +0.089 | 92.9% |
| Right Entorhinal | −0.069 | 85.0% |
| Left Fusiform | +0.066 | 91.6% |
| Right Pallidum | −0.043 | 90.8% |
| Left Lingual | +0.039 | 87.5% |

*ElasticNet final-refit coefficients (α = 0.010, L1 ratio = 0.10).*

## Key Results

| Metric | RAPS | Centiloid | P |
|--------|------|-----------|---|
| Out-of-fold Pearson r (5×5 nested CV) | 0.396 | — | < .001 |
| Correlation with CDR-SB slope (ADNI, N = 515) | 0.574 | 0.398 | < .001 (Steiger) |
| AUC for rapid decline > 1.5/yr (ADNI) | 0.865 | 0.737 | < .001 (DeLong) |
| Cox HR per SD (multivariate, ADNI) | 1.55 | 1.25 | < .001 |
| Incremental ΔC-index over clinical baseline (5-fold CV) | +0.028 | +0.014 | — |
| IPCW time-dependent AUC (mean over 1–8 yr) | 0.742 | 0.696 | — |
| CL 20–60 subgroup effect (ADNI, n = 173) | Cohen's d = 1.185 | — | < .001 |
| External validation (OASIS-3, N = 429): AUC > 1.5/yr | 0.871 | 0.807 | — |
| External validation (OASIS-3, CL 20–60, n = 87) | Cohen's d = 0.596 | — | .012 |
| Cross-tracer (ADNI [18F]florbetaben, N = 72): AUC > 1.0/yr | 0.902 | 0.785 | — |

## Repository Structure

```
raps-regional-amyloid-pet-score/
├── README.md
├── LICENSE
├── requirements.txt
├── config.py                          # Pipeline configuration
├── scripts/
│   ├── raps_construction.py           # Nested 5×5 CV for RAPS construction
│   ├── raps_supplementary_stats.py    # Supplementary table statistics
│   └── generate_nifti.py              # NIfTI maps for MRIcroGL
├── figures/
│   └── make_fig*.py                   # Figure generation scripts
├── model/
│   └── raps_7roi_final_model.joblib   # Trained 7-ROI model (ElasticNet + scaler)
└── data/
    ├── raps_roi_weights.csv           # Final 7-ROI coefficients
    └── bootstrap_roi_stability.csv    # Bootstrap stability (all 82 ROIs)
```

## Computing RAPS for a New Cohort

```python
import joblib
import numpy as np

# Load the trained model
model = joblib.load('model/raps_7roi_final_model.joblib')
enet     = model['enet']      # ElasticNet with 7-ROI coefficients
scaler   = model['scaler']    # StandardScaler (ADNI-fitted)
cov_model = model['cov_model'] # Covariate regression model
roi_names = model['roi_names'] # 7-ROI name order

# For each participant, provide 7 ROI SUVRs in this exact order:
#   ['LEFT_HIPPOCAMPUS_SUVR',
#    'LEFT_AMYGDALA_SUVR',
#    'CTX_LH_TRANSVERSETEMPORAL_SUVR',
#    'CTX_RH_ENTORHINAL_SUVR',
#    'CTX_LH_FUSIFORM_SUVR',
#    'RIGHT_PALLIDUM_SUVR',
#    'CTX_LH_LINGUAL_SUVR']

# X_roi: (N, 7) array of SUVR values in the order above
X_scaled = scaler.transform(X_roi)
raps_raw = enet.predict(X_scaled)
raps_z   = (raps_raw - raps_raw.mean()) / raps_raw.std()
```

**Note**: Site-specific recalibration of the StandardScaler may improve performance while keeping the ElasticNet weights fixed (see Supplementary Table S17 in the manuscript).

## Requirements

- Python ≥ 3.10
- scikit-learn ≥ 1.7
- numpy, pandas, scipy, statsmodels
- lifelines, scikit-survival (for survival analysis and IPCW time-dependent AUC)
- matplotlib, seaborn (for figures)
- nibabel, nilearn (for NIfTI generation)

See `requirements.txt` for exact versions.

## Data Availability

- **ADNI**: Available at [adni.loni.usc.edu](https://adni.loni.usc.edu) upon approval of a data use agreement
- **OASIS-3**: Available at [nitrc.org/ir/](https://www.nitrc.org/ir/) upon request

## Version Notes

- **v4 (current, 7-ROI)**: Final published version. 82-ROI ElasticNet pipeline with bootstrap stability ≥ 80% selecting 7 ROIs. PUTAMEN dropped from prior 8-ROI version after caudate inclusion fix that reduced its bootstrap frequency from 80.5 % to 76.8 % (basal ganglia collinearity).
- **v3 (archived as `v3-8roi` tag, 8-ROI)**: Earlier 80-ROI pipeline (caudate inadvertently excluded). Predictive performance is essentially identical (pooled CV r = 0.330 vs 0.329).

## License

MIT License

## Contact

Takumi Hirose, MD, PhD  
Center for Genomic and Regenerative Medicine, Juntendo University Graduate School of Medicine  
Email: t-hirose@juntendo.ac.jp
