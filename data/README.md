# Data files

## raps_roi_weights.csv
Final 7-ROI RAPS coefficients from ElasticNet refit on the complete ADNI cohort
(N = 515, alpha = 0.010, L1 ratio = 0.10).

| Column | Description |
|---|---|
| `roi` | FreeSurfer Desikan-Killiany ROI name (UC Berkeley pipeline) |
| `coefficient` | ElasticNet coefficient (β) |
| `abs_coefficient` | Absolute value of coefficient |
| `rank` | Rank by absolute coefficient (1 = largest) |

## bootstrap_roi_stability.csv
Bootstrap stability assessment (1,000 iterations) for all 82 FreeSurfer ROIs.

| Column | Description |
|---|---|
| `roi` | ROI name |
| `selection_frequency` | Fraction of bootstrap iterations where ROI was selected (0-1) |
| `coef_mean` | Mean coefficient across iterations |
| `coef_std` | SD of coefficient |
| `coef_p2.5`, `coef_p97.5` | 95% bootstrap CI |
| `selected_80pct` | Whether selection frequency ≥ 0.80 |
| `selected_in_final` | Whether ROI is in final 7-ROI RAPS |

## Note on ROI count (80 → 82)
The initial pipeline (V3) inadvertently excluded left/right caudate due to a substring
matching bug in date-column detection ("DATE" matched "CAUDATE"). After bug fix, the
proper 82-ROI set was analyzed; caudate had β = 0 in ElasticNet and was not selected,
so the conclusions remain unchanged. See V4 release notes.
