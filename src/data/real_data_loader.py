"""
Loaders for real clinical datasets used as external validation for
MaternaAI's three outcome pathways:
  - UCI Maternal Health Risk        -> general maternal risk (PPH-adjacent)
  - PhysioNet 2019 Sepsis Challenge -> sepsis
  - UCI Cardiotocography            -> fetal distress (HIE-adjacent)
"""

import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def load_maternal_health_risk() -> pd.DataFrame:
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=863)  # Maternal Health Risk
    df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    df.to_csv(RAW_DIR / "uci_maternal_health_risk.csv", index=False)
    return df


def load_cardiotocography() -> pd.DataFrame:
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=193)  # Cardiotocography
    df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    df.to_csv(RAW_DIR / "uci_cardiotocography.csv", index=False)
    return df


if __name__ == "__main__":
    mhr = load_maternal_health_risk()
    print("Maternal Health Risk:", mhr.shape)
    print(mhr.columns.tolist())
    print(mhr.iloc[:, -1].value_counts())

    ctg = load_cardiotocography()
    print("\nCardiotocography:", ctg.shape)
    print(ctg.columns.tolist())
    print(ctg.iloc[:, -1].value_counts())