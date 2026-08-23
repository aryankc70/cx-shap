"""
CX-SHAP Component 3: Temporal smoothing.

For monitored patients with repeated readings over time, smooths
attributions across the last w readings to prevent explanations
from flickering reading-to-reading. Off by default - only relevant
for longitudinal monitoring, not single-encounter triage.

phi_smooth(t) = (1-lambda)*phi(t) + lambda * mean(phi(t-w+1 : t))
"""

import numpy as np


def smooth_attributions(phi_sequence: np.ndarray, lam: float = 0.3, window: int = 3) -> np.ndarray:
    """phi_sequence: shape (T, n_features) - attributions at each of T
    timesteps for one patient (one task). Returns smoothed sequence,
    same shape."""
    T, n_features = phi_sequence.shape
    smoothed = np.zeros_like(phi_sequence)

    for t in range(T):
        start = max(0, t - window + 1)
        moving_avg = phi_sequence[start:t + 1].mean(axis=0)
        smoothed[t] = (1 - lam) * phi_sequence[t] + lam * moving_avg

    return smoothed


def demo_smoothing_effect(seed=42):
    """Synthetic demo: simulate a noisy attribution sequence and show
    smoothing reduces reading-to-reading flicker (variance of
    consecutive differences) while preserving the underlying trend."""
    rng = np.random.default_rng(seed)
    T, n_features = 12, 5

    # Simulate a trending signal + noise (like a deteriorating patient)
    trend = np.linspace(0, 1, T)[:, None] * np.ones((1, n_features))
    noise = rng.normal(0, 0.15, size=(T, n_features))
    raw = trend + noise

    smoothed = smooth_attributions(raw, lam=0.3, window=3)

    raw_flicker = np.mean(np.abs(np.diff(raw, axis=0)))
    smoothed_flicker = np.mean(np.abs(np.diff(smoothed, axis=0)))

    print(f"Raw reading-to-reading flicker (mean abs diff): {raw_flicker:.4f}")
    print(f"Smoothed reading-to-reading flicker: {smoothed_flicker:.4f}")
    print(f"Flicker reduction: {(1 - smoothed_flicker/raw_flicker)*100:.1f}%")

    # Confirm trend is preserved (correlation between smoothed and true trend)
    trend_corr = np.corrcoef(smoothed[:, 0], trend[:, 0])[0, 1]
    print(f"Correlation between smoothed signal and true trend: {trend_corr:.4f}")

    return raw, smoothed


if __name__ == "__main__":
    demo_smoothing_effect()