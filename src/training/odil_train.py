"""
O-DIL: three-phase imbalanced training procedure.

Now dataset-agnostic: dataset_name selects which preprocessed splits
to load (synthetic / maternal_health_risk / cardiotocography /
physionet_sepsis). Monotonic feature enforcement only applies when
a monotonic_idx list is explicitly passed - real datasets here train
without it, since we haven't defined clinically-validated monotonic
directions for their feature sets (documented limitation, not an
oversight).
"""

import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score

from src.models.mmnet import MMNet
from src.data.synthetic_generator import ALL_FEATURES, MONOTONIC_INCREASING

SEED = 42


def load_splits(dataset_name="synthetic"):
    return np.load(f"data/processed/{dataset_name}_splits.npz")


def get_synthetic_monotonic_idx():
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


def weighted_sampler_indices(y, oversample_factor=5, rng=None):
    rng = rng or np.random.default_rng(SEED)
    any_positive = (y.sum(axis=1) > 0)
    pos_idx = np.where(any_positive)[0]
    neg_idx = np.where(~any_positive)[0]
    if len(pos_idx) == 0:
        return np.arange(len(y))
    oversampled_pos = rng.choice(pos_idx, size=len(pos_idx) * oversample_factor, replace=True)
    combined = np.concatenate([neg_idx, oversampled_pos])
    rng.shuffle(combined)
    return combined


def tune_thresholds(model, X_val, y_val, task_names):
    model.eval()
    with torch.no_grad():
        preds = model(X_val).numpy()
    thresholds = {}
    for i, task in enumerate(task_names):
        best_f1, best_t = 0.0, 0.5
        for t in np.arange(0.10, 0.91, 0.05):
            f1 = f1_score(y_val[:, i], (preds[:, i] >= t).astype(int))
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[task] = float(best_t)
    return thresholds


def train_odil(
    dataset_name="synthetic",
    task_names=None,
    epochs_a=50, epochs_b=60, epochs_c=40, batch_size=64,
    monotonic_idx=None, run_phase_b=True, run_phase_c=True,
    seed=SEED,
):
    """task_names: list of task label names matching the dataset's label
    columns in order (e.g. ["pph","sepsis","hie"] or ["risk_binary"]).
    monotonic_idx: list of feature indices to constrain monotonic-increasing,
    or None to skip the constraint entirely."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    data = load_splits(dataset_name)
    X_train = torch.tensor(data["X_train"])
    y_train_np = data["y_train"]
    X_val = torch.tensor(data["X_val"])
    y_val_np = data["y_val"]
    X_test = torch.tensor(data["X_test"])
    y_test_np = data["y_test"]

    n_tasks = y_train_np.shape[1]
    if task_names is None:
        task_names = [f"task_{i}" for i in range(n_tasks)]

    y_train = torch.tensor(y_train_np)
    y_val = torch.tensor(y_val_np)

    model = MMNet(n_features=X_train.shape[1], monotonic_feature_idx=(monotonic_idx or []))
    n = X_train.shape[0]

    def clamp_if_monotonic():
        if monotonic_idx:
            model.clamp_monotonic_weights()

    # Head selection: MMNet always has 3 heads (pph/sepsis/hie by name,
    # but architecturally generic). We use the first n_tasks heads.
    head_cols = list(range(n_tasks))

    # --- Phase A ---
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCELoss()
    for epoch in range(epochs_a):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            preds = model(X_train[idx])[:, head_cols]
            loss = bce(preds, y_train[idx])
            loss.backward()
            optimizer.step()
            clamp_if_monotonic()

    # --- Phase B ---
    if run_phase_b:
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        focal = FocalLoss(alpha=0.75, gamma=2.0)
        rng = np.random.default_rng(seed)
        for epoch in range(epochs_b):
            model.train()
            idx_pool = weighted_sampler_indices(y_train_np, oversample_factor=5, rng=rng)
            rng.shuffle(idx_pool)
            for i in range(0, len(idx_pool), batch_size):
                idx = idx_pool[i:i + batch_size]
                optimizer.zero_grad()
                preds = model(X_train[idx])[:, head_cols]
                loss = focal(preds, y_train[idx])
                loss.backward()
                optimizer.step()
                clamp_if_monotonic()

    # --- Phase C ---
    if run_phase_c:
        pos_weights = torch.tensor([
            (y_train_np[:, i] == 0).sum() / max((y_train_np[:, i] == 1).sum(), 1)
            for i in range(n_tasks)
        ], dtype=torch.float32)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        for epoch in range(epochs_c):
            model.train()
            perm = torch.randperm(n)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                optimizer.zero_grad()
                preds = model(X_train[idx])[:, head_cols].clamp(1e-7, 1 - 1e-7)
                yb = y_train[idx]
                weights = torch.where(yb == 1, pos_weights, torch.ones_like(pos_weights))
                loss = -(weights * (yb * torch.log(preds) + (1 - yb) * torch.log(1 - preds))).mean()
                loss.backward()
                optimizer.step()
                clamp_if_monotonic()
        thresholds = tune_thresholds(model, X_val, y_val, task_names)
    else:
        thresholds = {task: 0.5 for task in task_names}

    model.eval()
    with torch.no_grad():
        test_preds_full = model(X_test).numpy()
    test_preds = test_preds_full[:, head_cols]

    results = {}
    for i, task in enumerate(task_names):
        auc = roc_auc_score(y_test_np[:, i], test_preds[:, i])
        f1 = f1_score(y_test_np[:, i], (test_preds[:, i] >= thresholds[task]).astype(int))
        results[task] = {"auc": round(float(auc), 4), "f1": round(float(f1), 4), "threshold": thresholds[task]}

    return model, results


if __name__ == "__main__":
    model, results = train_odil(
        dataset_name="synthetic",
        task_names=["pph", "sepsis", "hie"],
        monotonic_idx=get_synthetic_monotonic_idx(),
    )
    print("Full O-DIL results (synthetic):")
    for task, r in results.items():
        print(f"  {task}: AUC={r['auc']}  F1={r['f1']}  threshold={r['threshold']}")
    with open("results/odil_results.json", "w") as f:
        json.dump(results, f, indent=2)
    torch.save(model.state_dict(), "results/mmnet_odil_weights.pt")