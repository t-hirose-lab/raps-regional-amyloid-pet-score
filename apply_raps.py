"""
apply_raps.py — Apply the trained 9-ROI RAPS model to new amyloid PET data.

Usage:
    python apply_raps.py input.csv output.csv

input.csv must have these columns (FreeSurfer Desikan-Killiany SUVRs from the
UC Berkeley amyloid PET pipeline, or equivalent):
    - LEFT_HIPPOCAMPUS_SUVR
    - CTX_LH_FUSIFORM_SUVR
    - RIGHT_PALLIDUM_SUVR
    - CTX_LH_TRANSVERSETEMPORAL_SUVR
    - CTX_LH_LINGUAL_SUVR
    - CTX_RH_ENTORHINAL_SUVR
    - LEFT_ACCUMBENS_AREA_SUVR
    - LEFT_AMYGDALA_SUVR
    - LEFT_PUTAMEN_SUVR

output.csv will add a 'RAPS' column (z-scored within the input cohort).
"""
import sys
import joblib
import pandas as pd


def main(input_csv: str, output_csv: str, model_path: str = "model/raps_9roi_final_model.joblib") -> None:
    bundle = joblib.load(model_path)
    enet = bundle["enet"]
    scaler = bundle["scaler"]
    roi_names = list(bundle["rois"])

    df = pd.read_csv(input_csv)
    missing = [c for c in roi_names if c not in df.columns]
    if missing:
        sys.exit(f"Missing required columns: {missing}")

    X = df[roi_names].to_numpy(dtype=float)
    X_scaled = scaler.transform(X)
    raps_raw = enet.predict(X_scaled)
    raps_z = (raps_raw - raps_raw.mean()) / raps_raw.std()

    df["RAPS"] = raps_z
    df.to_csv(output_csv, index=False)
    print(f"Wrote {len(df)} participants to {output_csv}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
