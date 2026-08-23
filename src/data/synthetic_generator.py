"""
Synthetic maternal-neonatal dyad dataset generator.

Simulates paired mother-baby feature vectors and three simultaneous
clinical risk outcomes: postpartum hemorrhage (PPH), neonatal sepsis,
and hypoxic-ischemic encephalopathy (HIE).

Design principle: each outcome has its own latent risk pathway built
from a mix of shared (general deterioration) and outcome-specific
maternal/fetal features, with controllable correlation strength and
class prevalence — so the generator is tunable and auditable rather
than a black box.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class GeneratorConfig:
    n_samples: int = 8000
    seed: int = 42

    # Target positive prevalence per outcome
    pph_prevalence: float = 0.12
    sepsis_prevalence: float = 0.10
    hie_prevalence: float = 0.015

    # Shared vs outcome-specific weighting (0 = fully specific, 1 = fully shared)
    shared_weight: float = 0.35


MATERNAL_FEATURES = [
    "maternal_age", "gravidity", "parity", "systolic_bp", "diastolic_bp",
    "body_temp_f", "heart_rate", "blood_sugar",
]

FETAL_FEATURES = [
    "fetal_heart_rate", "baseline_variability", "accelerations",
    "uterine_contractions", "light_decelerations", "severe_decelerations",
    "prolonged_decelerations", "abnormal_short_term_variability",
    "mean_long_term_variability", "histogram_width",
]

ALL_FEATURES = MATERNAL_FEATURES + FETAL_FEATURES

# Features that are monotonic risk-increasing (used later for MMNet constraint)
MONOTONIC_INCREASING = [
    "systolic_bp", "diastolic_bp", "body_temp_f", "heart_rate",
    "blood_sugar", "uterine_contractions", "severe_decelerations",
    "prolonged_decelerations", "abnormal_short_term_variability",
]


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_dataset(config: GeneratorConfig = GeneratorConfig()) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(config.seed)
    n = config.n_samples

    # --- Base feature generation (clinically plausible ranges) ---
    df = pd.DataFrame({
        "maternal_age": rng.normal(28, 6, n).clip(15, 48),
        "gravidity": rng.poisson(2, n).clip(0, 8),
        "parity": rng.poisson(1, n).clip(0, 6),
        "systolic_bp": rng.normal(118, 15, n).clip(80, 200),
        "diastolic_bp": rng.normal(76, 10, n).clip(50, 130),
        "body_temp_f": rng.normal(98.4, 0.9, n).clip(96, 104),
        "heart_rate": rng.normal(82, 12, n).clip(50, 160),
        "blood_sugar": rng.normal(95, 20, n).clip(50, 250),
        "fetal_heart_rate": rng.normal(140, 12, n).clip(100, 190),
        "baseline_variability": rng.normal(8, 3, n).clip(0, 25),
        "accelerations": rng.poisson(2, n).clip(0, 10),
        "uterine_contractions": rng.poisson(3, n).clip(0, 12),
        "light_decelerations": rng.poisson(0.5, n).clip(0, 5),
        "severe_decelerations": rng.poisson(0.1, n).clip(0, 4),
        "prolonged_decelerations": rng.poisson(0.05, n).clip(0, 3),
        "abnormal_short_term_variability": rng.normal(35, 15, n).clip(0, 90),
        "mean_long_term_variability": rng.normal(9, 5, n).clip(0, 30),
        "histogram_width": rng.normal(70, 25, n).clip(10, 180),
    })

    # --- Latent risk pathways ---
    def zscore(col):
        return (df[col] - df[col].mean()) / df[col].std()

    shared_risk = (
        0.3 * zscore("body_temp_f") + 0.25 * zscore("heart_rate") +
        0.2 * zscore("systolic_bp") + 0.15 * zscore("uterine_contractions") +
        0.1 * zscore("blood_sugar")
    )

    pph_specific = (
        0.5 * zscore("systolic_bp") + 0.3 * zscore("diastolic_bp") +
        0.2 * zscore("parity")
    )
    sepsis_specific = (
        0.5 * zscore("body_temp_f") + 0.3 * zscore("heart_rate") +
        0.2 * zscore("fetal_heart_rate")
    )
    hie_specific = (
        0.4 * zscore("abnormal_short_term_variability") +
        0.3 * zscore("prolonged_decelerations") +
        0.2 * zscore("severe_decelerations") +
        0.1 * zscore("baseline_variability") * -1
    )

    sw = config.shared_weight
    pph_score = sw * shared_risk + (1 - sw) * pph_specific
    sepsis_score = sw * shared_risk + (1 - sw) * sepsis_specific
    hie_score = sw * shared_risk + (1 - sw) * hie_specific

    def to_binary(score, target_prevalence):
        noisy_score = score + rng.normal(0, 0.15 * score.std(), n)
        threshold = np.quantile(noisy_score, 1 - target_prevalence)
        binary = (noisy_score >= threshold).astype(int)
        return binary, noisy_score

    df["pph"], pph_raw = to_binary(pph_score, config.pph_prevalence)
    df["sepsis"], sepsis_raw = to_binary(sepsis_score, config.sepsis_prevalence)
    df["hie"], hie_raw = to_binary(hie_score, config.hie_prevalence)

    metadata = {
        "n_samples": n,
        "seed": config.seed,
        "features": ALL_FEATURES,
        "monotonic_increasing": MONOTONIC_INCREASING,
        "prevalence": {
            "pph": float(df["pph"].mean()),
            "sepsis": float(df["sepsis"].mean()),
            "hie": float(df["hie"].mean()),
        },
        "feature_outcome_corr_max": {
            "pph": float(df[MATERNAL_FEATURES + FETAL_FEATURES].corrwith(pd.Series(pph_raw)).abs().max()),
            "sepsis": float(df[MATERNAL_FEATURES + FETAL_FEATURES].corrwith(pd.Series(sepsis_raw)).abs().max()),
            "hie": float(df[MATERNAL_FEATURES + FETAL_FEATURES].corrwith(pd.Series(hie_raw)).abs().max()),
        },
    }
    return df, metadata


if __name__ == "__main__":
    import json
    df, meta = generate_dataset()
    df.to_csv("data/processed/maternal_neonatal_dataset.csv", index=False)
    with open("data/processed/dataset_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(df.shape)
    print(meta["prevalence"])
    print(meta["feature_outcome_corr_max"])