"""
Step 22: Train + explain across all cross-domain datasets (finance,
manufacturing, environment), reusing the exact same O-DIL training
and CX-SHAP explanation pipeline as healthcare (zero code changes -
only dataset/feature/task config differs). Directly tests RQ3: does
CX-SHAP generalize across structurally different domains without
code modification?
"""

import json
import numpy as np
import torch

from src.training.odil_train import train_odil
from src.explainability.attribution_decomposition import compute_decomposition, load_trained_model
from src.explainability.concordance import compute_concordance
from src.explainability.guideline_alignment import compute_guideline_alignment, build_domain_kb
from src.data.cross_domain_generators import CROSS_DOMAIN_CONFIG


def run_domain(domain_name, seed=42):
    cfg = CROSS_DOMAIN_CONFIG[domain_name]
    task_names = cfg["targets"]
    feature_names = cfg["features"]

    print(f"\n=== Training on {domain_name} ===")
    model, results = train_odil(
        dataset_name=domain_name, task_names=task_names, monotonic_idx=None, seed=seed,
    )
    for task, r in results.items():
        print(f"  {task}: AUC={r['auc']}  F1={r['f1']}")

    weights_path = f"results/mmnet_odil_weights_{domain_name}.pt"
    torch.save(model.state_dict(), weights_path)

    print(f"Explaining {domain_name} ...")
    phi_per_task, phi_shared, phi_task_specific, summary = compute_decomposition(
        dataset_name=domain_name, task_names=task_names, feature_names=feature_names,
        weights_path=weights_path, n_explain=30, seed=seed,
    )

    _, data = load_trained_model(domain_name, weights_path=weights_path)
    X_train = data["X_train"]
    rng = np.random.default_rng(seed)
    explain_idx = rng.choice(len(data["X_test"]), size=min(30, len(data["X_test"])), replace=False)
    X_explain = data["X_test"][explain_idx]

    concordance_results, _ = compute_concordance(
        phi_per_task, X_train, X_explain, model, task_names, feature_names, seed=seed
    )

    domain_kb = build_domain_kb(cfg["guideline_dirs_heuristic"], task_names)
    alignment_results = compute_guideline_alignment(phi_per_task, domain_kb, feature_names)

    return {
        "training_results": results,
        "concordance": concordance_results,
        "guideline_alignment": alignment_results,
    }


def run_cross_domain_validation():
    all_results = {}
    for domain in ["finance", "manufacturing", "environment"]:
        all_results[domain] = run_domain(domain)

    with open("results/cross_domain_validation.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n=== Cross-Domain Summary ===")
    for domain, r in all_results.items():
        aucs = [t["auc"] for t in r["training_results"].values()]
        rhos = [c["mean_rho"] for c in r["concordance"].values()]
        alphas = [a["alignment"] for a in r["guideline_alignment"].values() if a["alignment"] is not None]
        mean_alpha = f"{np.mean(alphas):.3f}" if alphas else "N/A"
        print(f"{domain}: best_auc={max(aucs):.3f}  mean_rho={np.mean(rhos):.3f}  mean_alpha={mean_alpha}")

    return all_results


if __name__ == "__main__":
    run_cross_domain_validation()