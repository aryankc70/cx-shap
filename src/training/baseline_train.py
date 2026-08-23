"""
Baseline training: standard BCE loss, no imbalance handling, no
monotonicity enforcement disabled/enabled toggle here — this script
exists specifically to reproduce and document the F1=0 collapse
caused by class imbalance (HIE prevalence ~1.7%), before O-DIL
(Step 6) fixes it.
"""

import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score

from src.models.mmnet import MMNet
from src.data.synthetic_generator import ALL_FEATURES, MONOTONIC_INCREASING

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def load_splits():
    data = np.load("data/processed/synthetic_splits.npz")
    return data


def get_monotonic_idx():
    return [ALL_FEATURES.index(f) for f in MONOTONIC_INCREASING]


def train_baseline(epochs=50, lr=1e-3, batch_size=64):
    data = load_splits()
    X_train = torch.tensor(data["X_train"])
    y_train = torch.tensor(data["y_train"])
    X_val = torch.tensor(data["X_val"])
    y_val = torch.tensor(data["y_val"])
    X_test = torch.tensor(data["X_test"])
    y_test = torch.tensor(data["y_test"])

    model = MMNet(n_features=X_train.shape[1], monotonic_feature_idx=get_monotonic_idx())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    n = X_train.shape[0]
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_train[idx], y_train[idx]
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            model.clamp_monotonic_weights()
            epoch_loss += loss.item() * len(idx)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}  loss={epoch_loss/n:.4f}")

    model.eval()
    with torch.no_grad():
        test_preds = model(X_test).numpy()
    y_test_np = y_test.numpy()

    results = {}
    task_names = ["pph", "sepsis", "hie"]
    for i, task in enumerate(task_names):
        auc = roc_auc_score(y_test_np[:, i], test_preds[:, i])
        f1 = f1_score(y_test_np[:, i], (test_preds[:, i] >= 0.5).astype(int))
        results[task] = {"auc": auc, "f1_at_0.5": f1}
        print(f"{task}: AUC={auc:.3f}  F1(thresh=0.5)={f1:.3f}")

    with open("results/baseline_bce_results.json", "w") as f:
        json.dump({"param_count": model.param_count(), "results": results}, f, indent=2)

    print(f"\nModel param count: {model.param_count()}")
    return model, results


if __name__ == "__main__":
    train_baseline()