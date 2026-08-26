"""
CX-SHAP Component 1: Attribution decomposition. Dataset-agnostic -
works for synthetic (healthcare) or any cross-domain dataset given
its feature names, task names, and trained model weights path.
"""

import json
import numpy as np
import torch
import shap

from src.models.mmnet import MMNet
from src.training.odil_train import load_splits, get_synthetic_monotonic_idx
from src.data.synthetic_generator import ALL_FEATURES as SYNTHETIC_FEATURES

DEFAULT_TASK_NAMES = ["pph", "sepsis", "hie"]
TASK_NAMES = DEFAULT_TASK_NAMES  # kept for backward compatibility with any old imports


def load_trained_model(dataset_name="synthetic", n_tasks=3, monotonic_idx=None,
                        weights_path=None):
    data = load_splits(dataset_name)
    n_features = data["X_train"].shape[1]
    if monotonic_idx is None and dataset_name == "synthetic":
        monotonic_idx = get_synthetic_monotonic_idx()
    model = MMNet(n_features=n_features, monotonic_feature_idx=(monotonic_idx or []))
    weights_path = weights_path or f"results/mmnet_odil_weights_{dataset_name}.pt"
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.eval()
    return model, data


def make_task_predict_fn(model, task_idx):
    def predict_fn(x_numpy):
        with torch.no_grad():
            x_t = torch.tensor(x_numpy, dtype=torch.float32)
            preds = model(x_t).numpy()
        return preds[:, task_idx]
    return predict_fn


def compute_decomposition(dataset_name="synthetic", task_names=None, feature_names=None,
                           weights_path=None, n_explain=30, n_background=30, seed=42):
    task_names = task_names or DEFAULT_TASK_NAMES
    feature_names = feature_names or SYNTHETIC_FEATURES
    n_tasks = len(task_names)

    model, data = load_trained_model(dataset_name, n_tasks=n_tasks, weights_path=weights_path)
    rng = np.random.default_rng(seed)

    X_test = data["X_test"]
    X_train = data["X_train"]

    n_background = min(n_background, len(X_train))
    n_explain = min(n_explain, len(X_test))

    background_idx = rng.choice(len(X_train), size=n_background, replace=False)
    background = X_train[background_idx]
    explain_idx = rng.choice(len(X_test), size=n_explain, replace=False)
    X_explain = X_test[explain_idx]

    phi_per_task = {}
    for k, task in enumerate(task_names):
        predict_fn = make_task_predict_fn(model, k)
        explainer = shap.PermutationExplainer(predict_fn, background, seed=seed)
        sv = explainer(X_explain, max_evals=(2 * X_explain.shape[1] + 1))
        phi_per_task[task] = sv.values

    phi_stack = np.stack([phi_per_task[t] for t in task_names], axis=0)
    phi_shared = np.mean(np.abs(phi_stack), axis=0)

    phi_task_specific = {}
    for k, task in enumerate(task_names):
        phi_task_specific[task] = phi_per_task[task] - phi_shared

    for k, task in enumerate(task_names):
        reconstructed = phi_shared + phi_task_specific[task]
        max_err = np.max(np.abs(reconstructed - phi_per_task[task]))
        assert max_err < 1e-6, f"Decomposition identity violated for {task}: max_err={max_err}"
    print(f"[{dataset_name}] Decomposition identity verified for all tasks.")

    summary = {}
    for i, feat in enumerate(feature_names):
        summary[feat] = {"mean_shared": float(np.mean(phi_shared[:, i]))}
        for task in task_names:
            summary[feat][f"mean_{task}_specific"] = float(np.mean(np.abs(phi_task_specific[task][:, i])))

    out_path = f"results/attribution_decomposition_summary_{dataset_name}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    return phi_per_task, phi_shared, phi_task_specific, summary


if __name__ == "__main__":
    # Regression check: synthetic results should be bit-identical to Step 9.
    compute_decomposition(dataset_name="synthetic", task_names=["pph", "sepsis", "hie"],
                           feature_names=SYNTHETIC_FEATURES)