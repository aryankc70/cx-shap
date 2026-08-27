"""
Step 23d: Full-batch variant - isolates whether GPU ever crosses over
to beat CPU for training this model, at the largest batch size
possible (one batch per epoch).
"""

import json
import platform
import time
import numpy as np
import torch
import torch.nn as nn

from src.models.mmnet import MMNet
from src.data.synthetic_generator import GeneratorConfig, generate_dataset, ALL_FEATURES, MONOTONIC_INCREASING
from sklearn.preprocessing import StandardScaler

DATA_SIZES = [1000, 10000, 100000, 500000, 1000000]
BENCHMARK_EPOCHS = 5


def get_platform_tag():
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "batch_mode": "full_batch",
    }


def benchmark_one_size(n_samples, seed=42):
    config = GeneratorConfig(n_samples=n_samples, seed=seed)
    df, _ = generate_dataset(config)

    X = df[ALL_FEATURES].values.astype(np.float32)
    y = df[["pph", "sepsis", "hie"]].values.astype(np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    X_t = torch.tensor(X)
    y_t = torch.tensor(y)

    monotonic_idx = [ALL_FEATURES.index(f) for f in MONOTONIC_INCREASING]
    model = MMNet(n_features=X_t.shape[1], monotonic_feature_idx=monotonic_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCELoss()

    train_start = time.perf_counter()
    for epoch in range(BENCHMARK_EPOCHS):
        model.train()
        optimizer.zero_grad()
        loss = bce(model(X_t), y_t)
        loss.backward()
        optimizer.step()
        model.clamp_monotonic_weights()
    train_time = time.perf_counter() - train_start

    model.eval()
    single_sample = X_t[0:1]
    latencies = []
    with torch.no_grad():
        for _ in range(110):
            start = time.perf_counter()
            _ = model(single_sample)
            latencies.append(time.perf_counter() - start)
    latencies = latencies[10:]
    median_latency_ms = float(np.median(latencies)) * 1000

    return {
        "n_samples": n_samples,
        "train_time_seconds": round(train_time, 4),
        "inference_latency_ms": round(median_latency_ms, 4),
        "epochs": BENCHMARK_EPOCHS,
        "param_count": model.param_count(),
    }


def run_benchmark():
    results = {"platform": get_platform_tag(), "results": []}
    for n in DATA_SIZES:
        print(f"Benchmarking n={n} (full-batch) ...")
        r = benchmark_one_size(n)
        results["results"].append(r)
        print(f"  train_time={r['train_time_seconds']}s  inference_latency={r['inference_latency_ms']}ms")

    with open("results/hardware_benchmark_mac_cpu_fullbatch.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    run_benchmark()