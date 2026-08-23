"""
O-DIL: three-phase imbalanced training procedure.

Phase A: standard BCE, general representation learning.
Phase B: focal loss + weighted oversampling of rare positive cases,
         forces the model to attend to minority classes (esp. HIE).
Phase C: weighted BCE + per-task decision threshold tuning on
         validation set, translating rank-ordering into calibrated
         binary predictions.
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
    return np.load("data/processed/synthetic_splits.npz")


def get_monotonic_idx():
    return [ALL_FEATURES.index(f) for f in MONOTONIC_INCREASING]


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        eps = 1e-7
        preds = preds.clamp(eps, 1 - eps)
        pt = torch.where(targets == 1, preds, 1 - preds)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        loss = -alpha_t * (1 - pt) ** self.gamma * torch.log(pt)
        return loss.mean()


def weighted_sampler_indices(y, n, oversample_factor=5, rng=None):
    """Returns an index array oversampling rows where any task label is 1."""
    rng = rng or np.random.default_rng(SEED)
    any_positive = (y.sum(axis=1) > 0)
    pos_idx = np.where(any_positive)[0]
    neg_idx = np.where(~any_positive)[0]
    oversampled_pos = rng.choice(pos_idx, size=len(pos_idx) * oversample_factor, replace=True)
    combined = np.concatenate([neg_idx, oversampled_pos])
    rng.shuffle(combined)
    return combined


def tune_thresholds(model, X_val, y_val):
    model.eval()
    with torch.no_grad():
        preds = model(X_val).numpy()
    thresholds = {}
    task_names = ["pph", "sepsis", "hie"]
    for i, task in enumerate(task_names):
        best_f1, best_t = 0.0, 0.5
        for t in np.arange(0.10, 0.91, 0.05):
            f1 = f1_score(y_val[:, i], (preds[:, i] >= t).astype(int))
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[task] = float(best_t)
    return thresholds


def train_odil(epochs_a=50, epochs_b=60, epochs_c=40, batch_size=64):
    data = load_splits()
    X_train = torch.tensor(data["X_train"])
    y_train_np = data["y_train"]
    y_train = torch.tensor(y_train_np)
    X_val = torch.tensor(data["X_val"])
    y_val_np = data["y_val"]
    y_val = torch.tensor(y_val_np)
    X_test = torch.tensor(data["X_test"])
    y_test_np = data["y_test"]

    model = MMNet(n_features=X_train.shape[1], monotonic_feature_idx=get_monotonic_idx())
    n = X_train.shape[0]

    # --- Phase A: representation learning (standard BCE) ---
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCELoss()
    for epoch in range(epochs_a):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            loss = bce(model(X_train[idx]), y_train[idx])
            loss.backward()
            optimizer.step()
            model.clamp_monotonic_weights()
    print("Phase A done.")

    # --- Phase B: focal loss + oversampling ---
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    focal = FocalLoss(alpha=0.75, gamma=2.0)
    rng = np.random.default_rng(SEED)
    for epoch in range(epochs_b):
        model.train()
        idx_pool = weighted_sampler_indices(y_train_np, n, oversample_factor=5, rng=rng)
        rng.shuffle(idx_pool)
        for i in range(0, len(idx_pool), batch_size):
            idx = idx_pool[i:i + batch_size]
            optimizer.zero_grad()
            loss = focal(model(X_train[idx]), y_train[idx])
            loss.backward()
            optimizer.step()
            model.clamp_monotonic_weights()
    print("Phase B done.")

    # --- Phase C: weighted BCE + threshold tuning ---
    pos_weights = torch.tensor([
        (y_train_np[:, i] == 0).sum() / max((y_train_np[:, i] == 1).sum(), 1)
        for i in range(3)
    ], dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(epochs_c):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            preds = model(X_train[idx]).clamp(1e-7, 1 - 1e-7)
            yb = y_train[idx]
            weights = torch.where(yb == 1, pos_weights, torch.ones_like(pos_weights))
            loss = -(weights * (yb * torch.log(preds) + (1 - yb) * torch.log(1 - preds))).mean()
            loss.backward()
            optimizer.step()
            model.clamp_monotonic_weights()
    print("Phase C done.")

    thresholds = tune_thresholds(model, X_val, y_val)
    print("Tuned thresholds:", thresholds)

    model.eval()
    with torch.no_grad():
        test_preds = model(torch.tensor(data["X_test"])).numpy()

    results = {}
    task_names = ["pph", "sepsis", "hie"]
    for i, task in enumerate(task_names):
        auc = roc_auc_score(y_test_np[:, i], test_preds[:, i])
        f1 = f1_score(y_test_np[:, i], (test_preds[:, i] >= thresholds[task]).astype(int))
        results[task] = {"auc": auc, "f1_tuned": f1, "threshold": thresholds[task]}
        print(f"{task}: AUC={auc:.3f}  F1(tuned)={f1:.3f}  threshold={thresholds[task]}")

    with open("results/odil_results.json", "w") as f:
        json.dump(results, f, indent=2)

    torch.save(model.state_dict(), "results/mmnet_odil_weights.pt")
    return model, results


if __name__ == "__main__":
    train_odil()