# RAPS — Regional Amyloid PET Score

Code and model weights for computing the Regional Amyloid PET Score (RAPS), a data-driven composite of 10 brain regions that predicts cognitive decline in amyloid-positive individuals.

## Reference

> Hirose T, et al. **A Data-Driven Regional Amyloid PET Score Predicts Cognitive Decline Beyond Centiloid.** *Alzheimer's & Dementia* (submitted).

## What is RAPS?

RAPS is a weighted sum of regional amyloid PET SUVRs from 10 FreeSurfer Desikan-Killiany ROIs, selected by ElasticNet regression with bootstrap stability assessment. It outperforms global Centiloid in predicting cognitive decline (CDR-SB slope), particularly in the intermediate amyloid burden range (CL 20–60).

## Key Results

| Metric | RAPS | Centiloid | P |
|--------|------|-----------|---|
| Correlation with CDR-SB slope (ADNI) | r = 0.329 | r = 0.222 | .007 (Steiger) |
| AUC for rapid decline >1.5/yr (ADNI) | 0.835 | 0.737 | .007 (DeLong) |
| AUC for rapid decline >1.5/yr (OASIS-3) | 0.894 | 0.807 | .029 (DeLong) |
| CL 20–60 subgroup effect (ADNI) | Cohen's d = 1.01 | — | < .001 |

## Files

| File | Description |
|------|-------------|
| `raps_roi_weights.csv` | ElasticNet coefficients for 11 non-zero ROIs (10 stable + 1 additional) |
| `bootstrap_roi_stability.csv` | Bootstrap selection frequency (1,000 iterations) for all 80 ROIs |
| `raps_construction.py` | Full nested cross-validation pipeline for RAPS construction |
| `requirements.txt` | Python dependencies |

## Computing RAPS for a New Cohort

Given 80 FreeSurfer ROI SUVRs for each participant:

```
RAPS = Σ βᵢ × Zᵢ    (i = 1, ..., 11)
```

where `Zᵢ = (SUVRᵢ − μᵢ) / σᵢ` (z-score standardized using training data parameters).

The 10 stable ROIs (bootstrap selection ≥ 80%) and their coefficients are provided in `raps_roi_weights.csv`. The left hippocampus has the largest weight (β = −0.140, 100% bootstrap stability).

## Requirements

- Python ≥ 3.10
- See `requirements.txt`

## Data Availability

- **ADNI**: Available at [adni.loni.usc.edu](https://adni.loni.usc.edu) upon approval
- **OASIS-3**: Available at [nitrc.org/ir/](https://www.nitrc.org/ir/) upon request

## License

MIT License

## Contact

Takumi Hirose, MD, PhD  
Department of Psychiatry, Juntendo University School of Medicine  
t-hirose@juntendo.ac.jp
