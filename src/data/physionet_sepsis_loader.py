"""
Aggregates PhysioNet 2019 Sepsis Challenge per-patient .psv time series
into one row per patient, for use alongside the other tabular datasets.

Aggregation choice: last-observed vital signs (most recent snapshot before
end of stay/labeling) plus summary stats (mean, max) for key vitals across
the stay. Label = 1 if SepsisLabel is ever 1 for that patient, else 0.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

RAW_DIR = Path("data/raw/physionet_sepsis/training")
OUT_PATH = Path("data/raw/physionet_sepsis_aggregated.csv")

KEY_VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]
KEY_LABS = ["WBC", "Lactate", "Creatinine", "Platelets"]


def aggregate_patient(filepath: Path) -> dict:
    df = pd.read_csv(filepath, sep="|")
    label = int(df["SepsisLabel"].max())

    row = {"patient_id": filepath.stem, "sepsis": label}

    # Last observed value (ffill then take last row) for key vitals
    last = df[KEY_VITALS].ffill().iloc[-1]
    for col in KEY_VITALS:
        row[f"{col}_last"] = last[col]

    # Mean and max across stay for key vitals + labs
    for col in KEY_VITALS + KEY_LABS:
        if col in df.columns:
            row[f"{col}_mean"] = df[col].mean()
            row[f"{col}_max"] = df[col].max()

    # Static demographics (constant per patient)
    for col in ["Age", "Gender", "HospAdmTime"]:
        row[col] = df[col].iloc[0]

    row["ICU_LOS_hours"] = df["ICULOS"].max()
    return row


def build_aggregated_dataset() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("training_set*/*.psv"))
    rows = []
    for f in tqdm(files, desc="Aggregating patients"):
        try:
            rows.append(aggregate_patient(f))
        except Exception as e:
            print(f"Skipped {f.name}: {e}")
    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    df = build_aggregated_dataset()
    df.to_csv(OUT_PATH, index=False)
    print(df.shape)
    print(df["sepsis"].value_counts())
    print(df["sepsis"].mean())