"""
Full CX-SHAP pipeline: runs all 4 components on a single patient and
produces the complete explanation tuple E_CX(x,k) = (phi_shared,
phi_task_k, rho_k, alpha_k) for each task k, plus a natural-language
trace for one worked example patient - the key illustrative figure
for the paper.
"""

import json
import numpy as np
import torch

from src.explainability.attribution_decomposition import (
    load_trained_model, make_task_predict_fn, TASK_NAMES
)
from src.explainability.concordance import compute_lime_attributions, trust_level
from src.explainability.guideline_alignment import compute_guideline_alignment, GUIDELINE_KB
from src.data.synthetic_generator import ALL_FEATURES

import shap
from scipy.stats import spearmanr


def explain_patient(patient_idx, n_background=30, n_shap_evals=None, n_lime_perturbations=80, seed=42):
    model, data = load_trained_model()
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    rng = np.random.default_rng(seed)
    background_idx = rng.choice(len(X_train), size=n_background, replace=False)
    background = X_train[background_idx]

    x_patient = X_test[patient_idx:patient_idx + 1]
    true_labels = y_test[patient_idx]

    with torch.no_grad():
        pred_probs = model(torch.tensor(x_patient, dtype=torch.float32)).numpy()[0]

    n_features = len(ALL_FEATURES)
    n_evals = n_shap_evals or (2 * n_features + 1)

    phi_per_task = {}
    for k, task in enumerate(TASK_NAMES):
        predict_fn = make_task_predict_fn(model, k)
        explainer = shap.PermutationExplainer(predict_fn, background, seed=seed)
        sv = explainer(x_patient, max_evals=n_evals)
        phi_per_task[task] = sv.values[0]  # (n_features,)

    phi_stack = np.stack([phi_per_task[t] for t in TASK_NAMES], axis=0)
    phi_shared = np.mean(np.abs(phi_stack), axis=0)
    phi_task_specific = {t: phi_per_task[t] - phi_shared for t in TASK_NAMES}

    lime_per_task = {}
    concordance = {}
    for k, task in enumerate(TASK_NAMES):
        lime_vals = compute_lime_attributions(
            model, X_train, x_patient, k, n_perturbations=n_lime_perturbations, seed=seed
        )[0]
        lime_per_task[task] = lime_vals
        rho, _ = spearmanr(np.abs(phi_per_task[task]), np.abs(lime_vals))
        concordance[task] = {"rho": round(float(rho), 4), "trust_level": trust_level(rho)}

    alignment = compute_guideline_alignment({t: phi_per_task[t][None, :] for t in TASK_NAMES})

    result = {
        "patient_idx": int(patient_idx),
        "true_labels": {t: int(true_labels[k]) for k, t in enumerate(TASK_NAMES)},
        "predicted_probs": {t: round(float(pred_probs[k]), 4) for k, t in enumerate(TASK_NAMES)},
        "shared_attribution": {f: round(float(phi_shared[i]), 4) for i, f in enumerate(ALL_FEATURES)},
        "task_specific_residual": {
            t: {f: round(float(phi_task_specific[t][i]), 4) for i, f in enumerate(ALL_FEATURES)}
            for t in TASK_NAMES
        },
        "concordance": concordance,
        "guideline_alignment": alignment,
    }
    return result


def print_trace(result):
    print(f"\n=== Patient #{result['patient_idx']} ===")
    print(f"True labels: {result['true_labels']}")
    print(f"Predicted probabilities: {result['predicted_probs']}\n")

    shared_sorted = sorted(result["shared_attribution"].items(), key=lambda kv: -abs(kv[1]))[:3]
    print("Top 3 shared (general risk) features:")
    for feat, val in shared_sorted:
        print(f"  {feat}: {val}")

    for task in TASK_NAMES:
        print(f"\n--- {task.upper()} ---")
        residuals = result["task_specific_residual"][task]
        top = sorted(residuals.items(), key=lambda kv: -abs(kv[1]))[:3]
        print(f"  Top task-specific residuals: {top}")
        c = result["concordance"][task]
        print(f"  Concordance: rho={c['rho']} ({c['trust_level']} trust)")
        a = result["guideline_alignment"][task]
        print(f"  Guideline alignment: {a['alignment']} (checked: {a.get('features_checked')})")


if __name__ == "__main__":
    # Pick a patient with at least one positive label for a meaningful trace
    model, data = load_trained_model()
    y_test = data["y_test"]
    positive_idx = np.where(y_test.sum(axis=1) > 0)[0]
    chosen_idx = int(positive_idx[0]) if len(positive_idx) > 0 else 0

    result = explain_patient(chosen_idx)
    print_trace(result)

    with open("results/worked_example_patient.json", "w") as f:
        json.dump(result, f, indent=2)