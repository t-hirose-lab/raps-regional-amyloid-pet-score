# RAPS - Regional Amyloid PET Score

Code and model weights for computing the Regional Amyloid PET Score (RAPS), a data-driven 9-ROI composite that predicts cognitive decline in amyloid-positive individuals beyond global Centiloid.

## Reference

> Hirose T, Akamatsu W, Kato T, for the Alzheimer's Disease Neuroimaging Initiative. **A Data-Driven Regional Amyloid PET Score Predicts Cognitive Decline Beyond Centiloid: Multi-Cohort Multi-Tracer Validation.** *Journal of Alzheimer's Disease* (under review).

## What is RAPS?

RAPS is a weighted sum of regional amyloid PET SUVRs from **9 FreeSurfer Desikan-Killiany ROIs**, selected by ElasticNet regression with 1,000-iteration bootstrap stability assessment (intersection of non-zero ElasticNet coefficients AND bootstrap selection frequency ≥ 80%) on 82 ROIs in **433 amyloid-positive ADNI participants** imaged with [¹⁸F]florbetapir. It outperforms global Centiloid in predicting cognitive decline (CDR-SB slope) and provides prognostic information complementary to global amyloid burden, with secondary exploratory signal in the intermediate amyloid burden range (CL 20–60).

### The 9 RAPS ROIs

| ROI | Coefficient (β) | Bootstrap Stability |
|-----|-------------------|-------------------|
| Left Hippocampus | −0.169 | 100.0% |
| Right Pallidum | −0.066 | 89.5% |
| Right Entorhinal | −0.061 | 85.0% |
| Left Fusiform | +0.058 | 93.6% |
| Left Putamen | +0.051 | 80.6% |
| Left Amygdala | +0.048 | 81.2% |
| Left Lingual | +0.042 | 85.4% |
| Left Transverse Temporal | +0.042 | 86.8% |
| Left Accumbens | +0.026 | 82.7% |

*ElasticNet final-refit coefficients (α = 0.058, L1 ratio = 0.10). Bootstrap stability is the selection frequency across 1,000 bootstrap iterations of the 82-ROI ElasticNet pipeline; all 9 final ROIs met the ≥ 80% stability criterion. ROIs are ordered by absolute coefficient (rank 1–9).*

## Key Results

| Metric | RAPS | Centiloid | P |
|--------|------|-----------|---|
| Out-of-fold Pearson r, covariate-adjusted (5×5 nested CV, ADNI) | 0.339 | — | — |
| Correlation with CDR-SB slope (ADNI, N = 433) | 0.550 | 0.311 | < .001 (Steiger, z = 5.45) |
| AUC for rapid decline > 1.0/yr (ADNI, primary) | 0.813 | 0.713 | .003 (DeLong) |
| AUC for rapid decline > 1.5/yr (ADNI) | 0.822 | 0.680 | < .001 (DeLong) |
| Cox HR per SD (multivariable M3, ADNI) | 1.64 | 1.24 | < .001 (concordance = 0.745) |
| CL 20–60 subgroup effect (ADNI, n = 173) | Cohen's d = 0.938 | — | — |
| NACC SCAN (N = 1,531): AUC > 1.0/yr | 0.803 | 0.750 | — |
| NACC SCAN (N = 1,531): Cox HR per SD (M3) | 1.21 | 1.37 | .035 (RAPS) |
| OASIS-3 (N = 428): AUC > 1.0/yr | 0.864 | 0.834 | .574 (DeLong) |
| OASIS-3 (M3 Cox, N = 425): Cox HR per SD | 1.35 | — | .019 |
| Cross-tracer ADNI [¹⁸F]florbetaben (N = 71): AUC > 1.0/yr | 0.905 | 0.781 | .182 (DeLong) |
| Cross-cohort meta-analysis: pooled ΔAUC (RAPS − CL, > 1.0/yr) | +0.069 [95% CI +0.031, +0.107] | — | < .001 (I² = 0%) |
| Cross-cohort meta-analysis: pooled Cox HR per SD (M3) | 1.37 [95% CI 1.14, 1.64] | — | — |

*Validation cohorts span four amyloid PET tracers ([¹⁸F]florbetapir, [¹⁸F]florbetaben, [¹¹C]PiB, [¹⁸F]NAV4694). The NACC SCAN analytic cohort (N = 1,531) is Centiloid-unrestricted. See the manuscript and Supplementary Information for full statistics, additional thresholds, and subgroup analyses.*

## Repository Structure

```
raps-regional-amyloid-pet-score/
├── README.md
├── LICENSE
├── requirements.txt
├── apply_raps.py                      # Apply RAPS to new SUVR data
├── model/
│   └── raps_9roi_final_model.joblib   # Trained 9-ROI model (ElasticNet + scaler)
└── data/
    ├── README.md
    ├── raps_roi_weights.csv           # Final 9-ROI coefficients
    └── bootstrap_roi_stability.csv    # Bootstrap stability (all 82 ROIs)
```

## Computing RAPS for a New Cohort

The trained model expects 9 SUVRs per participant in this exact order:

| Order | ROI |
|---|---|
| 1 | `LEFT_HIPPOCAMPUS_SUVR` |
| 2 | `CTX_LH_FUSIFORM_SUVR` |
| 3 | `RIGHT_PALLIDUM_SUVR` |
| 4 | `CTX_LH_TRANSVERSETEMPORAL_SUVR` |
| 5 | `CTX_LH_LINGUAL_SUVR` |
| 6 | `CTX_RH_ENTORHINAL_SUVR` |
| 7 | `LEFT_ACCUMBENS_AREA_SUVR` |
| 8 | `LEFT_AMYGDALA_SUVR` |
| 9 | `LEFT_PUTAMEN_SUVR` |

### Quick start (CLI)

```bash
python apply_raps.py input.csv output.csv
```

`input.csv` must contain the 9 columns above; `output.csv` will add a `RAPS` column (z-scored within the input cohort).

### Programmatic use

```python
import joblib

bundle = joblib.load("model/raps_9roi_final_model.joblib")
enet, scaler = bundle["enet"], bundle["scaler"]

# X_roi: (N, 9) numpy array of SUVRs in the order above
X_scaled = scaler.transform(X_roi)
raps_raw = enet.predict(X_scaled)
raps_z   = (raps_raw - raps_raw.mean()) / raps_raw.std()
```

**Note**: Site-specific recalibration of the StandardScaler may improve performance while keeping the ElasticNet weights fixed (see Supplementary Information in the manuscript).

## Requirements

- Python ≥ 3.10
- scikit-learn ≥ 1.7
- numpy, pandas, scipy, statsmodels, joblib
- lifelines, scikit-survival (for survival analysis and IPCW time-dependent AUC)

See `requirements.txt` for exact versions.

## Data Availability

- **ADNI**: Available at [adni.loni.usc.edu](https://adni.loni.usc.edu) upon approval of a data use agreement
- **NACC / SCAN**: Available at [naccdata.org](https://naccdata.org) upon request
- **OASIS-3**: Available at [nitrc.org/ir/](https://www.nitrc.org/ir/) upon request

## Version Notes

- **v5 (current, 9-ROI)**: Rebuilt on the ADNI discovery cohort of N = 433 amyloid-positive (baseline Centiloid ≥ 20) CN/MCI participants for the EJNMMI submission. The 82-ROI ElasticNet + bootstrap pipeline (intersection of non-zero coefficients and bootstrap stability ≥ 80%) selected 9 ROIs. Relative to the prior 7-ROI version (v4), the left putamen was re-selected and the left accumbens was newly added, yielding the 9-ROI model. Validation was extended to four cohorts spanning four tracers: ADNI (N = 433), NACC SCAN (analytic N = 1,531), OASIS-3 (N = 428), and an ADNI [¹⁸F]florbetaben cross-tracer subset (N = 71).
- **v4 (archived as `v4-7roi` tag, 7-ROI)**: Earlier model built on ADNI N = 515. 82-ROI ElasticNet pipeline with bootstrap stability ≥ 80% selecting 7 ROIs (putamen had been dropped after a caudate inclusion fix). Validated in ADNI (N = 515), OASIS-3 (N = 429), and an ADNI florbetaben subset (N = 72).
- **v3 (archived as `v3-8roi` tag, 8-ROI)**: Earliest 80-ROI pipeline (caudate inadvertently excluded).

## License

MIT License

## Contact

Takumi Hirose, MD, PhD  
Center for Genomic and Regenerative Medicine, Juntendo University Graduate School of Medicine  
Email: t-hirose@juntendo.ac.jp
