"""
Runs the same MMNet + O-DIL recipe on each real dataset independently
(single-task, since each real dataset has one outcome). Reports honest
comparison against the synthetic multi-task results - this is NOT a
weight-transfer experiment (feature spaces differ), it's validation
that the architecture and training procedure generalize to genuine
clinical data end-to-end.
"""

import json
from src.training.odil_train import train_odil

REAL_DATASETS = {
    "maternal_health_risk": ["risk_binary"],
    "cardiotocography": ["fetal_risk_binary"],
    "physionet_sepsis": ["sepsis"],
}


def run_real_data_validation():
    all_results = {}
    for dataset_name, task_names in REAL_DATASETS.items():
        print(f"\nTraining on: {dataset_name} ...")
        _, results = train_odil(
            dataset_name=dataset_name,
            task_names=task_names,
            monotonic_idx=None,  # no defined monotonic direction for real feature sets
        )
        all_results[dataset_name] = results
        for task, r in results.items():
            print(f"  {task}: AUC={r['auc']}  F1={r['f1']}  threshold={r['threshold']}")

    with open("results/real_data_validation.json", "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results


if __name__ == "__main__":
    run_real_data_validation()