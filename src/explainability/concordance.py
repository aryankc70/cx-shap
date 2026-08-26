"""
CX-SHAP Component 2: Cross-method concordance scoring. Dataset-agnostic.
"""

import json
import numpy as np
from scipy.stats import spearmanr
from lime.lime_tabular import LimeTabularExplainer

from src.explainability.attribution_decomposition import make_task_predict_fn

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


def compute_lime_attributions(model, X_train, X_explain, task_idx, feature_names,
                               n_perturbations=80, seed=42):
    explainer = LimeTabularExplainer(
        X_train, feature_names=feature_names, mode="regression",
        random_state=seed, discretize_continuous=False,
    )
    predict_fn = make_task_predict_fn(model, task_idx)

    lime_values = np.zeros((len(X_explain), len(feature_names)))
    for i, x in enumerate(X_explain):
        exp = explainer.explain_instance(
            x, predict_fn, num_features=len(feature_names), num_samples=n_perturbations
        )
        feat_weights = dict(exp.as_list())
        for j, feat in enumerate(feature_names):
            lime_values[i, j] = feat_weights.get(feat, 0.0)
    return lime_values


def compute_concordance(phi_per_task, X_train, X_explain, model, task_names, feature_names,
                         n_perturbations=80, seed=42):
    concordance_results = {}
    lime_per_task = {}

    for k, task in enumerate(task_names):
        print(f"Computing LIME for {task} ...")
        lime_vals = compute_lime_attributions(model, X_train, X_explain, k, feature_names,
                                               n_perturbations, seed)
        lime_per_task[task] = lime_vals
        shap_vals = phi_per_task[task]

        rhos = []
        for i in range(len(X_explain)):
            rho, _ = spearmanr(np.abs(shap_vals[i]), np.abs(lime_vals[i]))
            if not np.isnan(rho):
                rhos.append(rho)

        mean_rho = float(np.mean(rhos)) if rhos else 0.0
        std_rho = float(np.std(rhos)) if rhos else 0.0
        concordance_results[task] = {
            "mean_rho": round(mean_rho, 4),
            "std_rho": round(std_rho, 4),
            "trust_level": trust_level(mean_rho),
            "n_samples": len(rhos),
        }
        print(f"  {task}: mean_rho={mean_rho:.4f}  std={std_rho:.4f}  trust={trust_level(mean_rho)}")

    return concordance_results, lime_per_task


if __name__ == "__main__":
    from src.explainability.attribution_decomposition import compute_decomposition, load_trained_model
    from src.data.synthetic_generator import ALL_FEATURES

    phi_per_task, phi_shared, phi_task_specific, summary = compute_decomposition(
        dataset_name="synthetic", task_names=["pph", "sepsis", "hie"], feature_names=ALL_FEATURES
    )

    model, data = load_trained_model("synthetic")
    X_train = data["X_train"]
    rng = np.random.default_rng(42)
    explain_idx = rng.choice(len(data["X_test"]), size=30, replace=False)
    X_explain = data["X_test"][explain_idx]

    concordance_results, lime_per_task = compute_concordance(
        phi_per_task, X_train, X_explain, model, ["pph", "sepsis", "hie"], ALL_FEATURES
    )

    with open("results/concordance_results_synthetic.json", "w") as f:
        json.dump(concordance_results, f, indent=2)