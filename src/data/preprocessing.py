"""
Unified preprocessing pipeline for all four datasets (synthetic +
3 real). Produces stratified train/val/test splits and fitted
StandardScaler objects, saved to disk for reuse at inference time.
"""

import json
import pickle
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class SplitData:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: list
    scaler: StandardScaler


def make_splits(
    df: pd.DataFrame,
    feature_cols: list,
    label_cols: list,
    dataset_name: str,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> SplitData:
    df = df.dropna(subset=feature_cols + label_cols).reset_index(drop=True)

    X = df[feature_cols].values.astype(np.float32)
    y = df[label_cols].values.astype(np.float32)

    # Stratify on first label column if binary, else no stratification
    strat = y[:, 0] if y.shape[1] == 1 or len(np.unique(y[:, 0])) <= 10 else None

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), random_state=seed, stratify=strat
    )
    strat_temp = y_temp[:, 0] if strat is not None else None
    relative_val = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - relative_val), random_state=seed, stratify=strat_temp
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Persist scaler + splits
    with open(PROCESSED_DIR / f"{dataset_name}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    np.savez(
        PROCESSED_DIR / f"{dataset_name}_splits.npz",
        X_train=X_train_scaled, X_val=X_val_scaled, X_test=X_test_scaled,
        y_train=y_train, y_val=y_val, y_test=y_test,
    )

    meta = {
        "dataset_name": dataset_name,
        "feature_names": feature_cols,
        "label_names": label_cols,
        "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
        "seed": seed,
    }
    with open(PROCESSED_DIR / f"{dataset_name}_split_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return SplitData(X_train_scaled, X_val_scaled, X_test_scaled,
                      y_train, y_val, y_test, feature_cols, scaler)


def preprocess_synthetic():
    from src.data.synthetic_generator import ALL_FEATURES
    df = pd.read_csv("data/processed/maternal_neonatal_dataset.csv")
    return make_splits(df, ALL_FEATURES, ["pph", "sepsis", "hie"], "synthetic")


def preprocess_maternal_health_risk():
    df = pd.read_csv("data/raw/uci_maternal_health_risk.csv")
    df["risk_binary"] = (df["RiskLevel"] != "low risk").astype(int)
    features = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
    return make_splits(df, features, ["risk_binary"], "maternal_health_risk")


def preprocess_cardiotocography():
    df = pd.read_csv("data/raw/uci_cardiotocography.csv")
    df["fetal_risk_binary"] = (df["NSP"] != 1).astype(int)  # 1=Normal -> 0, Suspect/Path -> 1
    features = ["LB", "AC", "FM", "UC", "DL", "DS", "DP", "ASTV", "MSTV",
                "ALTV", "MLTV", "Width", "Min", "Max", "Nmax", "Nzeros",
                "Mode", "Mean", "Median", "Variance", "Tendency"]
    return make_splits(df, features, ["fetal_risk_binary"], "cardiotocography")


def preprocess_physionet_sepsis():
    df = pd.read_csv("data/raw/physionet_sepsis_aggregated.csv")
    feature_cols = [c for c in df.columns if c not in ("patient_id", "sepsis")]
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
    return make_splits(df, feature_cols, ["sepsis"], "physionet_sepsis")

def preprocess_cross_domain(domain: str):
    from src.data.cross_domain_generators import CROSS_DOMAIN_CONFIG
    cfg = CROSS_DOMAIN_CONFIG[domain]
    df = pd.read_csv(f"data/processed/{domain}_dataset.csv")
    return make_splits(df, cfg["features"], cfg["targets"], domain)


if __name__ == "__main__":
    for name, fn in [
        ("synthetic", preprocess_synthetic),
        ("maternal_health_risk", preprocess_maternal_health_risk),
        ("cardiotocography", preprocess_cardiotocography),
        ("physionet_sepsis", preprocess_physionet_sepsis),
    ]:
        split = fn()
        print(f"{name}: train={split.X_train.shape}, val={split.X_val.shape}, test={split.X_test.shape}")
        print(f"  label balance (train): {split.y_train.mean(axis=0)}")
    
    for domain in ["finance", "manufacturing", "environment"]:
        split = preprocess_cross_domain(domain)
        print(f"{domain}: train={split.X_train.shape}, val={split.X_val.shape}, test={split.X_test.shape}")
        print(f"  label balance (train): {split.y_train.mean(axis=0)}")