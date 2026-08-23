"""
Ablation study: isolates the contribution of each O-DIL component
and the monotonicity constraint, plus single-task baselines.
"""

import json
from src.training.odil_train import train_odil

CONFIGS = {
    "full_model":              dict(use_monotonic=True,  run_phase_b=True,  run_phase_c=True),
    "remove_phase_b":          dict(use_monotonic=True,  run_phase_b=False, run_phase_c=True),
    "remove_phase_c":          dict(use_monotonic=True,  run_phase_b=True,  run_phase_c=False),
    "remove_monotonicity":     dict(use_monotonic=False, run_phase_b=True,  run_phase_c=True),
}


def run_ablation():
    all_results = {}

    for name, cfg in CONFIGS.items():
        print(f"\nRunning: {name} ...")
        _, results = train_odil(**cfg)
        all_results[name] = results
        for task, r in results.items():
            print(f"  {task}: AUC={r['auc']}  F1={r['f1']}")

    # Single-task models: one MMNet-style model per task, full O-DIL recipe each
    print("\nRunning: single_task_models ...")
    single_task_results = {}
    for i, task in enumerate(["pph", "sepsis", "hie"]):
        _, r = train_odil(use_monotonic=True, run_phase_b=True, run_phase_c=True, label_cols=[i])
        single_task_results[task] = r[task]
        print(f"  {task}: AUC={r[task]['auc']}  F1={r[task]['f1']}")
    all_results["single_task_models"] = single_task_results

    with open("results/ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results


if __name__ == "__main__":
    run_ablation()