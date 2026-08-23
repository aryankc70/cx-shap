"""
CX-SHAP Component 2: Cross-method concordance scoring.

Computes LIME attributions per task and measures rank agreement with
SHAP via Spearman correlation. This is the "trust signal" component -
a model can be highly accurate (high AUC) while its two independent
explanation methods disagree on WHY, which this component surfaces.
"""

import json
import numpy as np
import torch
from scipy.stats import spearmanr
from lime.lime_tabular import LimeTabularExplainer

from src.explainability.attribution_decomposition import (
    load_trained_model, make_task_predict_fn, TASK_NAMES
)
from src.data.synthetic_generator import ALL_FEATURES

TRUST_THRESHOLDS = [
    (0.70, "high"),
    (0.40, "moderate"),
    (0.00, "low"),
    (-1.01, "unreliable"),
]


def trust_level(rho):
    for threshold, label in TRUST_THRESHOLDS:
        if rho >= threshold:
            return label
    return "unreliable"


def compute_lime_attributions(model, X_train, X_explain, task_idx, n_perturbations=80, seed=42):
    explainer = LimeTabularExplainer(
        X_train, feature_names=ALL_FEATURES, mode="regression",
        random_state=seed, discretize_continuous=False,
    )
    predict_fn = make_task_predict_fn(model, task_idx)

    lime_values = np.zeros((len(X_explain), len(ALL_FEATURES)))
    for i, x in enumerate(X_explain):
        exp = explainer.explain_instance(
            x, predict_fn, num_features=len(ALL_FEATURES), num_samples=n_perturbations
        )
        feat_weights = dict(exp.as_list())
        # LIME's as_list() keys are strings possibly with inequality conditions
        # since discretize_continuous=False, keys should just be feature names
        for j, feat in enumerate(ALL_FEATURES):
            lime_values[i, j] = feat_weights.get(feat, 0.0)
    return lime_values


def compute_concordance(phi_per_task, X_train, X_explain, model, n_perturbations=80, seed=42):
    concordance_results = {}
    lime_per_task = {}

    for k, task in enumerate(TASK_NAMES):
        print(f"Computing LIME for {task} ...")
        lime_vals = compute_lime_attributions(model, X_train, X_explain, k, n_perturbations, seed)
        lime_per_task[task] = lime_vals

        shap_vals = phi_per_task[task]  # (n_explain, n_features)

        # Per-sample Spearman rho on |attribution| rankings, then average
        rhos = []
        for i in range(len(X_explain)):
            rho, _ = spearmanr(np.abs(shap_vals[i]), np.abs(lime_vals[i]))
            if not np.isnan(rho):
                rhos.append(rho)

        mean_rho = float(np.mean(rhos))
        std_rho = float(np.std(rhos))
        concordance_results[task] = {
            "mean_rho": round(mean_rho, 4),
            "std_rho": round(std_rho, 4),
            "trust_level": trust_level(mean_rho),
            "n_samples": len(rhos),
        }
        print(f"  {task}: mean_rho={mean_rho:.4f}  std={std_rho:.4f}  trust={trust_level(mean_rho)}")

    return concordance_results, lime_per_task


if __name__ == "__main__":
    from src.explainability.attribution_decomposition import compute_decomposition
    phi_per_task, phi_shared, phi_task_specific, summary = compute_decomposition()

    model, data = load_trained_model()
    X_train = data["X_train"]
    rng = np.random.default_rng(42)
    explain_idx = rng.choice(len(data["X_test"]), size=30, replace=False)
    X_explain = data["X_test"][explain_idx]

    concordance_results, lime_per_task = compute_concordance(phi_per_task, X_train, X_explain, model)

    with open("results/concordance_results.json", "w") as f:
        json.dump(concordance_results, f, indent=2)