import numpy as np
import os
import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest

def label_generation(df: pd.DataFrame, path_to_save: str, split: str = "training") -> None:

    # ============================================================
    # 2. Basic cleanup
    #    Keep the columns needed for label export, but do not use
    #    IDs / coordinates / spectroscopic metadata as anomaly inputs
    # ============================================================
    id_cols = ["object_id", "specz_redshift"]

    # Columns NOT to use for anomaly-label generation
    exclude_from_features = {
        "object_id",
        "coord",
        "ra",
        "dec",
        "specz_ra",
        "specz_dec",
        "specz_name",
        "skymap_id",
        "x_coord",
        "y_coord",
        "specz_redshift",
        "specz_redshift_err",
        "specz_flag_homogeneous",
        "specz_mag_i",
    }

    # ============================================================
    # 3. Feature engineering from existing morphology/photometry
    #    These are only used to CREATE the outlier label
    # ============================================================
    bands = ["g", "r", "i", "z", "y"]

    eps = 1e-8

    # ---- Colors from cmodel magnitudes
    color_pairs = [("g", "r"), ("r", "i"), ("i", "z"), ("z", "y"), ("g", "i")]
    for b1, b2 in color_pairs:
        c1 = f"{b1}_cmodel_mag"
        c2 = f"{b2}_cmodel_mag"
        if c1 in df.columns and c2 in df.columns:
            df[f"{b1}-{b2}_color"] = df[c1] - df[c2]

    # ---- Per-band engineered morphology features
    for b in bands:
        major = f"{b}_major_axis"
        minor = f"{b}_minor_axis"
        pop5 = f"{b}_central_image_pop_5px_rad"
        pop10 = f"{b}_central_image_pop_10px_rad"
        pop15 = f"{b}_central_image_pop_15px_rad"
        area = f"{b}_isophotal_area"
        hlr = f"{b}_half_light_radius"
        peak_sb = f"{b}_peak_surface_brightness"
        mag = f"{b}_cmodel_mag"

        if major in df.columns and minor in df.columns:
            df[f"{b}_axis_ratio"] = df[minor] / (df[major] + eps)
            df[f"{b}_elongation"] = df[major] / (df[minor] + eps)

        if pop5 in df.columns and pop15 in df.columns:
            df[f"{b}_conc_5_15"] = df[pop5] / (df[pop15] + eps)

        if pop10 in df.columns and pop15 in df.columns:
            df[f"{b}_conc_10_15"] = df[pop10] / (df[pop15] + eps)

        if peak_sb in df.columns and mag in df.columns:
            df[f"{b}_peak_minus_mag"] = df[peak_sb] - df[mag]

        if area in df.columns:
            df[f"{b}_log_isophotal_area"] = np.log1p(df[area])

        if hlr in df.columns:
            df[f"{b}_log_half_light_radius"] = np.log1p(df[hlr])

    # ---- Cross-band summary features
    summary_bases = [
        "sersic_index",
        "ellipticity",
        "half_light_radius",
        "petro_rad",
        "peak_surface_brightness",
        "cmodel_mag",
        "axis_ratio",
        "elongation",
        "conc_5_15",
        "conc_10_15",
    ]

    for base in summary_bases:
        cols = [f"{b}_{base}" for b in bands if f"{b}_{base}" in df.columns]
        if len(cols) >= 2:
            df[f"{base}_mean"] = df[cols].mean(axis=1)
            df[f"{base}_std"] = df[cols].std(axis=1)

    # ============================================================
    # 4. Build feature matrix for anomaly labeling
    # ============================================================
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    feature_cols = [
        c for c in numeric_cols
        if c not in exclude_from_features
    ]

    X = df[feature_cols].copy()

    # Optional: replace inf/-inf if any appear from ratios/logs
    X = X.replace([np.inf, -np.inf], np.nan)

    # ============================================================
    # 5. Impute + robust scale
    #    This is only to make anomaly label generation stable
    # ============================================================
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()

    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)

    # ============================================================
    # 6. Create outlier labels with Isolation Forest
    #    contamination controls the fraction labeled as outliers
    #    Example: 0.05 means about 5% outliers
    # ============================================================
    contamination = 0.05

    iso = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )

    iso.fit(X_scaled)

    # sklearn returns:
    #   1  = inlier
    #  -1  = outlier
    raw_pred = iso.predict(X_scaled)
    outlier_label = (raw_pred == -1).astype(int)

    # Higher score below means "more anomalous"
    # score_samples(): larger = more normal
    # so negate it to make larger = more anomalous
    outlier_score = -iso.score_samples(X_scaled)

    # ============================================================
    # 7. Export label table for later image-based model training
    # ============================================================
    labels_df = pd.DataFrame({
        "object_id": df["object_id"],
        "specz_redshift": df["specz_redshift"],
        "outlier_label": outlier_label,
        "outlier_score": outlier_score,
    })

    # Optional: inspect label distribution
    print(labels_df["outlier_label"].value_counts(dropna=False))
    print(labels_df.head())

    labels_df.to_csv(os.path.join(path_to_save,f"galaxy_labels_redshift_outlier_{split}.csv"), index=False)
    print(f"Saved: galaxy_labels_redshift_outlier_{split}.csv")

if __name__ == "__main__":
    data_dir = Path("/home/erskordi/Documents/UNM-files/Spring26/ECE533/Project/Data")
    csv_paths = list(data_dir.glob("5x*.csv"))
    for path in csv_paths:
        if "training" in str(path):
            training_parts = path.parts
        elif "validation" in str(path):
            validation_parts = path.parts
        elif "test" in str(path):
            test_parts = path.parts
    training_csv = pd.read_csv(os.path.join(*training_parts))
    validation_csv = pd.read_csv(os.path.join(*validation_parts))
    test_csv = pd.read_csv(os.path.join(*test_parts))

    label_generation(training_csv, path_to_save=str(data_dir), split="training")
    label_generation(validation_csv, path_to_save=str(data_dir), split="validation")
    label_generation(test_csv, path_to_save=str(data_dir), split="test")

    print("Files saved at:", data_dir)