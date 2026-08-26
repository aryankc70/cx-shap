"""
Cross-domain synthetic dataset generators: finance, manufacturing,
environment. Adapted from recovered source material (original thesis
benchmark scaffold), restructured to match this repo's conventions.

Each domain has 3 binary outcomes trained jointly (multi-task), tests
whether CX-SHAP generalizes across structurally different domains
with zero code changes to the core pipeline (only dataset/feature
config changes).

Guideline directions for finance/manufacturing/environment are
DOMAIN HEURISTICS, not clinically-sourced guidelines like the
healthcare domain's Component 4 knowledge base - this distinction
is stated explicitly and should be preserved in any write-up.
"""

import numpy as np
import pandas as pd


def generate_finance(n: int, seed: int = 42) -> pd.DataFrame:
    """Synthetic credit card transactions (Kaggle-style: V1-V28 PCA
    components + Amount). Three tasks: fraud, high-value fraud,
    card-not-present fraud."""
    rng = np.random.default_rng(seed)
    data = {f"V{i}": rng.normal(0, 1, n) for i in range(1, 29)}
    data["Amount"] = np.abs(rng.exponential(88, n))
    df = pd.DataFrame(data)

    n_fraud = max(1, int(n * 0.00173))
    fraud_idx = rng.choice(n, n_fraud, replace=False)
    for i in [1, 2, 3, 4, 14, 17]:
        df.loc[fraud_idx, f"V{i}"] += rng.normal(-3, 1, n_fraud)
    df.loc[fraud_idx, "Amount"] *= rng.uniform(0.5, 8, n_fraud)

    fraud = np.zeros(n, dtype=int)
    fraud[fraud_idx] = 1
    df["fraud"] = fraud
    df["hv_fraud"] = ((fraud == 1) & (df["Amount"] > 200)).astype(int)
    df["cnp_fraud"] = ((fraud == 1) & (df["V3"] < -2)).astype(int)
    return df


def generate_manufacturing(n: int, seed: int = 42) -> pd.DataFrame:
    """Synthetic predictive maintenance (AI4I 2020-style). Three
    failure modes: machine failure, tool wear failure, heat failure."""
    rng = np.random.default_rng(seed)
    air_temp = rng.normal(300.0, 2.0, n).clip(295, 305)
    process_temp = air_temp + rng.normal(10.0, 1.0, n)
    rot_speed = rng.normal(1538, 179, n).clip(1168, 2886)
    torque = rng.normal(39.99, 9.97, n).clip(3.8, 76.6)
    tool_wear = rng.uniform(0, 253, n)
    product_type = rng.choice(["L", "M", "H"], n, p=[0.5, 0.3, 0.2])
    type_enc = (product_type == "H").astype(float) + 0.5 * (product_type == "M").astype(float)

    # Tool wear failure (TWF): designed injection, since the natural joint
    # tail (tool_wear>200 AND torque*rot_speed<11000) essentially never
    # occurs by chance with these marginal distributions. Target ~0.5%
    # prevalence, matching the real AI4I 2020 dataset's TWF rate.
    twf = np.zeros(n, dtype=int)
    n_twf = max(1, int(n * 0.005))
    twf_idx = rng.choice(n, n_twf, replace=False)
    tool_wear[twf_idx] = rng.uniform(200, 253, n_twf)
    torque[twf_idx] = rng.uniform(3.8, 10.0, n_twf)
    rot_speed[twf_idx] = rng.uniform(1168, 1300, n_twf)
    twf[twf_idx] = 1

    temp_diff = process_temp - air_temp
    hdf = ((temp_diff < 8.6) & (rot_speed < 1380)).astype(int)
    power = torque * rot_speed * 2 * np.pi / 60
    pwf = ((power < 3500) | (power > 9000)).astype(int)

    machine_fail = np.clip(twf + hdf + pwf + (rng.random(n) < 0.005).astype(int), 0, 1)
    noise_idx = rng.choice(n, int(n * 0.001), replace=False)
    machine_fail[noise_idx] = 1 - machine_fail[noise_idx]

    return pd.DataFrame({
        "air_temperature": air_temp, "process_temperature": process_temp,
        "rotational_speed": rot_speed, "torque": torque, "tool_wear": tool_wear,
        "product_type_enc": type_enc,
        "machine_failure": machine_fail, "tool_wear_failure": twf, "heat_failure": hdf,
    })


def generate_environment(n: int, seed: int = 42) -> pd.DataFrame:
    """Synthetic air quality (Beijing Multi-Site-style). Three WHO
    threshold exceedance tasks: PM2.5, NO2, O3."""
    rng = np.random.default_rng(seed)
    temp = rng.normal(12.0, 12.0, n)
    dewp = temp - rng.uniform(2, 15, n)
    rain = rng.exponential(0.05, n).clip(0, 50)
    wspm = rng.exponential(2.0, n).clip(0, 15)
    hour = rng.integers(0, 24, n)
    month = rng.integers(1, 13, n)

    rush_hour = ((hour >= 7) & (hour <= 9)) | ((hour >= 17) & (hour <= 19))
    winter = (month >= 11) | (month <= 2)

    pm25 = (rng.lognormal(3.5, 0.8, n) + 40 * winter.astype(float) +
            20 * rush_hour.astype(float) - 10 * (wspm > 3).astype(float) +
            rng.normal(0, 15, n)).clip(0, 500)
    no2 = (rng.lognormal(4.0, 0.6, n) + 30 * rush_hour.astype(float) -
           8 * (wspm > 3).astype(float) + rng.normal(0, 20, n)).clip(0, 400)
    o3 = (rng.lognormal(4.2, 0.5, n) + 40 * ((hour >= 12) & (hour <= 16)).astype(float) +
          20 * (temp > 25).astype(float) + rng.normal(0, 25, n)).clip(0, 400)
    co = rng.lognormal(6.0, 0.7, n).clip(0, 5000)
    so2 = rng.lognormal(3.0, 0.8, n).clip(0, 200)

    return pd.DataFrame({
        "pm25": pm25, "no2": no2, "o3": o3, "co": co, "so2": so2,
        "temperature": temp, "pressure": rng.normal(1013, 10, n),
        "dew_point": dewp, "rainfall": rain, "wind_speed": wspm,
        "hour": hour, "month": month,
        "pm25_high": (pm25 > 75).astype(int),
        "no2_high": (no2 > 200).astype(int),
        "o3_high": (o3 > 180).astype(int),
    })


CROSS_DOMAIN_CONFIG = {
    "finance": {
        "generator": generate_finance,
        "features": [f"V{i}" for i in range(1, 29)] + ["Amount"],
        "targets": ["fraud", "hv_fraud", "cnp_fraud"],
        "n_samples": 100000,
        "guideline_dirs_heuristic": {"Amount": +1, "V1": -1, "V2": -1, "V3": +1, "V4": +1, "V14": -1},
    },
    "manufacturing": {
        "generator": generate_manufacturing,
        "features": ["air_temperature", "process_temperature", "rotational_speed",
                     "torque", "tool_wear", "product_type_enc"],
        "targets": ["machine_failure", "tool_wear_failure", "heat_failure"],
        "n_samples": 50000,
        "guideline_dirs_heuristic": {"tool_wear": +1, "torque": +1, "rotational_speed": -1, "air_temperature": +1},
    },
    "environment": {
        "generator": generate_environment,
        "features": ["pm25", "no2", "o3", "co", "so2", "temperature",
                     "pressure", "dew_point", "rainfall", "wind_speed", "hour", "month"],
        "targets": ["pm25_high", "no2_high", "o3_high"],
        "n_samples": 5000,
        "guideline_dirs_heuristic": {"pm25": +1, "no2": +1, "o3": +1, "co": +1, "so2": +1, "wind_speed": -1, "rainfall": -1},
    },
}


if __name__ == "__main__":
    for domain, cfg in CROSS_DOMAIN_CONFIG.items():
        df = cfg["generator"](cfg["n_samples"])
        df.to_csv(f"data/processed/{domain}_dataset.csv", index=False)
        print(f"{domain}: {df.shape}")
        for t in cfg["targets"]:
            print(f"  {t}: {df[t].mean()*100:.2f}% positive")