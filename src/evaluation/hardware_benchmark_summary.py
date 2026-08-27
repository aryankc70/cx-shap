"""
Step 23e: Aggregates all 6 hardware benchmark runs (Mac CPU / Colab T4
GPU x batch=64 / batch=2048 / full-batch) into a single comparison
table and figure - the core evidence for the "GPU paradox depends on
batch size" finding.
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES_DIR = Path("results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RUNS = {
    ("Mac CPU", "batch=64"): "results/hardware_benchmark_mac_cpu.json",
    ("Mac CPU", "batch=2048"): "results/hardware_benchmark_mac_cpu_largebatch.json",
    ("Mac CPU", "full-batch"): "results/hardware_benchmark_mac_cpu_fullbatch.json",
    ("Colab T4 GPU", "batch=64"): "results/hardware_benchmark_colab_t4.json",
    ("Colab T4 GPU", "batch=2048"): "results/hardware_benchmark_colab_t4_largebatch.json",
    ("Colab T4 GPU", "full-batch"): "results/hardware_benchmark_colab_t4_fullbatch.json",
}


def load_all():
    data = {}
    for key, path in RUNS.items():
        with open(path) as f:
            data[key] = json.load(f)["results"]
    return data


def print_summary_table(data):
    sizes = [r["n_samples"] for r in data[("Mac CPU", "batch=64")]]
    print(f"{'n':>10} | {'CPU b=64':>10} {'CPU b=2048':>11} {'CPU full':>9} | {'GPU b=64':>10} {'GPU b=2048':>11} {'GPU full':>9}")
    print("-" * 90)
    for i, n in enumerate(sizes):
        cpu64 = data[("Mac CPU", "batch=64")][i]["train_time_seconds"]
        cpu2048 = data[("Mac CPU", "batch=2048")][i]["train_time_seconds"]
        cpufull = data[("Mac CPU", "full-batch")][i]["train_time_seconds"]
        gpu64 = data[("Colab T4 GPU", "batch=64")][i]["train_time_seconds"]
        gpu2048 = data[("Colab T4 GPU", "batch=2048")][i]["train_time_seconds"]
        gpufull = data[("Colab T4 GPU", "full-batch")][i]["train_time_seconds"]
        print(f"{n:>10} | {cpu64:>10.3f} {cpu2048:>11.3f} {cpufull:>9.3f} | {gpu64:>10.3f} {gpu2048:>11.3f} {gpufull:>9.3f}")

def plot_train_time_comparison(data):
    sizes = [r["n_samples"] for r in data[("Mac CPU", "batch=64")]]
    modes = ["batch=64", "batch=2048", "full-batch"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    for ax, mode in zip(axes, modes):
        cpu_times = [r["train_time_seconds"] for r in data[("Mac CPU", mode)]]
        gpu_times = [r["train_time_seconds"] for r in data[("Colab T4 GPU", mode)]]
        ax.plot(sizes, cpu_times, marker="o", label="Mac CPU")
        ax.plot(sizes, gpu_times, marker="s", label="Colab T4 GPU")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Training samples")
        ax.set_ylabel("Train time (s)")
        ax.set_title(f"Training time: {mode}")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)

    plt.suptitle("GPU vs CPU Training Time by Batch Mode")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "gpu_paradox_training_by_batch.png")
    plt.close()
    print("Saved gpu_paradox_training_by_batch.png")


def plot_inference_latency_comparison(data):
    sizes = [r["n_samples"] for r in data[("Mac CPU", "batch=64")]]

    fig, ax = plt.subplots(figsize=(8, 5))
    for platform, marker in [("Mac CPU", "o"), ("Colab T4 GPU", "s")]:
        # Inference latency should be ~independent of batch mode - use batch=64 run as representative
        latencies = [r["inference_latency_ms"] for r in data[(platform, "batch=64")]]
        ax.plot(sizes, latencies, marker=marker, label=platform)

    ax.set_xscale("log")
    ax.set_xlabel("Training samples (model trained on)")
    ax.set_ylabel("Single-sample inference latency (ms)")
    ax.set_title("Inference Latency: Never Crosses Over Regardless of Training Batch Mode")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "gpu_paradox_inference_latency.png")
    plt.close()
    print("Saved gpu_paradox_inference_latency.png")


if __name__ == "__main__":
    data = load_all()
    print_summary_table(data)
    plot_train_time_comparison(data)
    plot_inference_latency_comparison(data)