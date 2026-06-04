# Data files

## raps_roi_weights.csv
Final 9-ROI RAPS coefficients from ElasticNet refit on the ADNI discovery cohort
(N = 433, alpha = 0.058, L1 ratio = 0.10).

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
| `coef_p2.5`, `coef_p50`, `coef_p97.5` | 2.5th / 50th / 97.5th percentile of the bootstrap coefficient |
| `selected_80pct` | Whether selection frequency ≥ 0.80 |
| `final_coef`, `final_nonzero` | Coefficient and non-zero flag in the final 82-ROI ElasticNet refit |
| `selected_in_final` | Whether ROI is in the final 9-ROI RAPS |
| `intersect_canonical` | Intersection criterion: non-zero final coefficient AND bootstrap stability ≥ 80% |

## Note on ROI count (7 → 9)
The final RAPS is defined as the intersection of non-zero ElasticNet coefficients and
bootstrap selection frequency ≥ 80%. Rebuilding the pipeline on the EJNMMI discovery
cohort of N = 433 amyloid-positive ADNI participants selected 9 ROIs. Relative to the
prior 7-ROI version (built on ADNI N = 515), the left putamen was re-selected and the
left accumbens was newly added. See the main README Version Notes.
