"""
CX-SHAP Component 1: Attribution decomposition.

Decomposes per-task SHAP attributions into a shared component
(general risk signal common across tasks) and a task-specific
residual (what's uniquely informative for that one outcome).

phi_j^shared = (1/K) * sum_k |phi_j^k|
phi_j^task_k = phi_j^k - phi_j^shared

Decomposition satisfies: phi^shared + phi^task_k = phi^k (by construction).
"""

import json
import numpy as np
import torch
import shap

from src.models.mmnet import MMNet
from src.training.odil_train import load_splits, get_synthetic_monotonic_idx
from src.data.synthetic_generator import ALL_FEATURES

TASK_NAMES = ["pph", "sepsis", "hie"]


def load_trained_model():
    data = load_splits("synthetic")
    n_features = data["X_train"].shape[1]
    model = MMNet(n_features=n_features, monotonic_feature_idx=get_synthetic_monotonic_idx())
    model.load_state_dict(torch.load("results/mmnet_odil_weights.pt"))
    model.eval()
    return model, data


def make_task_predict_fn(model, task_idx):
    def predict_fn(x_numpy):
        with torch.no_grad():
            x_t = torch.tensor(x_numpy, dtype=torch.float32)
            preds = model(x_t).numpy()
        return preds[:, task_idx]
    return predict_fn


def compute_decomposition(n_explain=30, n_background=30, seed=42):
    model, data = load_trained_model()
    rng = np.random.default_rng(seed)

    X_test = data["X_test"]
    X_train = data["X_train"]

    background_idx = rng.choice(len(X_train), size=n_background, replace=False)
    background = X_train[background_idx]

    explain_idx = rng.choice(len(X_test), size=n_explain, replace=False)
    X_explain = X_test[explain_idx]

    # Per-task SHAP values, shape (n_explain, n_features) each
    phi_per_task = {}
    for k, task in enumerate(TASK_NAMES):
        predict_fn = make_task_predict_fn(model, k)
        explainer = shap.PermutationExplainer(predict_fn, background, seed=seed)
        sv = explainer(X_explain, max_evals=(2 * X_explain.shape[1] + 1))
        phi_per_task[task] = sv.values  # (n_explain, n_features)

    # Stack: shape (K, n_explain, n_features)
    phi_stack = np.stack([phi_per_task[t] for t in TASK_NAMES], axis=0)

    # phi_shared per feature per sample = mean over tasks of |phi|
    phi_shared = np.mean(np.abs(phi_stack), axis=0)  # (n_explain, n_features)

    phi_task_specific = {}
    for k, task in enumerate(TASK_NAMES):
        phi_task_specific[task] = phi_per_task[task] - phi_shared

    # Verify decomposition identity holds
    for k, task in enumerate(TASK_NAMES):
        reconstructed = phi_shared + phi_task_specific[task]
        max_err = np.max(np.abs(reconstructed - phi_per_task[task]))
        assert max_err < 1e-6, f"Decomposition identity violated for {task}: max_err={max_err}"
    print("Decomposition identity verified: phi_shared + phi_task = phi_k for all tasks.")

    # Aggregate: mean |shared| and mean |task-specific| per feature, across samples
    summary = {}
    for i, feat in enumerate(ALL_FEATURES):
        summary[feat] = {
            "mean_shared": float(np.mean(phi_shared[:, i])),
        }
        for task in TASK_NAMES:
            summary[feat][f"mean_{task}_specific"] = float(np.mean(np.abs(phi_task_specific[task][:, i])))

    with open("results/attribution_decomposition_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print top 5 features by shared attribution, and top 5 by each task-specific residual
    print("\nTop 5 features by shared (general risk) attribution:")
    ranked = sorted(summary.items(), key=lambda kv: -kv[1]["mean_shared"])[:5]
    for feat, vals in ranked:
        print(f"  {feat}: shared={vals['mean_shared']:.4f}")

    for task in TASK_NAMES:
        print(f"\nTop 5 features by {task}-specific residual:")
        ranked = sorted(summary.items(), key=lambda kv: -kv[1][f"mean_{task}_specific"])[:5]
        for feat, vals in ranked:
            print(f"  {feat}: {task}_specific={vals[f'mean_{task}_specific']:.4f}")

    return phi_per_task, phi_shared, phi_task_specific, summary


if __name__ == "__main__":
    compute_decomposition()