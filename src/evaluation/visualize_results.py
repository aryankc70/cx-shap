import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

FIGURES_DIR = Path("results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 10


def plot_ablation_comparison():
    with open("results/ablation_results.json") as f:
        ablation = json.load(f)

    configs = ["full_model", "remove_phase_b", "remove_phase_c", "remove_monotonicity"]
    config_labels = ["Full O-DIL", "No Phase B\n(focal+oversample)", "No Phase C\n(threshold tuning)", "No monotonicity"]
    tasks = ["pph", "sepsis", "hie"]
    task_labels = ["PPH", "Sepsis", "HIE"]

    x = np.arange(len(configs))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, task in enumerate(tasks):
        f1_vals = [ablation[cfg][task]["f1"] for cfg in configs]
        ax.bar(x + i * width, f1_vals, width, label=task_labels[i])

    ax.set_xlabel("Configuration")
    ax.set_ylabel("F1 Score")
    ax.set_title("Ablation Study: F1 Score by Configuration and Task")
    ax.set_xticks(x + width)
    ax.set_xticklabels(config_labels)
    ax.legend(title="Task")
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ablation_comparison.png")
    plt.close()
    print("Saved ablation_comparison.png")


def plot_concordance_by_task():
    with open("results/concordance_results.json") as f:
        concordance = json.load(f)

    tasks = list(concordance.keys())
    task_labels = [t.upper() for t in tasks]
    rhos = [concordance[t]["mean_rho"] for t in tasks]
    stds = [concordance[t]["std_rho"] for t in tasks]

    trust_colors = {"high": "#2ca02c", "moderate": "#ff7f0e", "low": "#d62728", "unreliable": "#8c0000"}
    colors = [trust_colors[concordance[t]["trust_level"]] for t in tasks]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(task_labels, rhos, yerr=stds, capsize=5, color=colors)

    for threshold, label in [(0.70, "High"), (0.40, "Moderate"), (0.00, "Low")]:
        ax.axhline(threshold, linestyle="--", color="gray", linewidth=0.8)
        ax.text(len(tasks) - 0.4, threshold + 0.01, label, fontsize=8, color="gray")

    ax.set_ylabel("Spearman ρ (SHAP vs LIME)")
    ax.set_title("Cross-Method Concordance by Task")
    ax.set_ylim(-0.3, 1.0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "concordance_by_task.png")
    plt.close()
    print("Saved concordance_by_task.png")


def plot_attribution_decomposition():
    with open("results/attribution_decomposition_summary.json") as f:
        summary = json.load(f)

    features = list(summary.keys())
    shared_vals = [summary[f]["mean_shared"] for f in features]

    # Sort by shared attribution, take top 8 for readability
    sorted_idx = np.argsort(shared_vals)[::-1][:8]
    top_features = [features[i] for i in sorted_idx]

    tasks = ["pph", "sepsis", "hie"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, task in zip(axes, tasks):
        shared = [summary[f]["mean_shared"] for f in top_features]
        specific = [summary[f][f"mean_{task}_specific"] for f in top_features]

        y = np.arange(len(top_features))
        ax.barh(y, shared, label="Shared", color="#4C72B0")
        ax.barh(y, specific, left=shared, label=f"{task.upper()}-specific", color="#DD8452")
        ax.set_yticks(y)
        ax.set_yticklabels(top_features, fontsize=8)
        ax.set_title(task.upper())
        ax.set_xlabel("Mean |attribution|")
        ax.legend(fontsize=8)

    axes[0].invert_yaxis()
    plt.suptitle("Attribution Decomposition: Shared vs Task-Specific")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "attribution_decomposition.png")
    plt.close()
    print("Saved attribution_decomposition.png")


if __name__ == "__main__":
    plot_ablation_comparison()
    plot_concordance_by_task()
    plot_attribution_decomposition()