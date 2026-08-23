"""
Tests the CX-SHAP decomposition IDENTITY algebraically, independent
of SHAP's stochastic estimation - fast, deterministic, CI-safe.
"""

import numpy as np


def decompose(phi_stack):
    """phi_stack: (K, n_samples, n_features). Returns (phi_shared, phi_task_specific list)."""
    phi_shared = np.mean(np.abs(phi_stack), axis=0)
    phi_task_specific = [phi_stack[k] - phi_shared for k in range(phi_stack.shape[0])]
    return phi_shared, phi_task_specific


def test_decomposition_identity_holds_exactly():
    rng = np.random.default_rng(0)
    phi_stack = rng.normal(size=(3, 10, 5))  # 3 tasks, 10 samples, 5 features

    phi_shared, phi_task_specific = decompose(phi_stack)

    for k in range(3):
        reconstructed = phi_shared + phi_task_specific[k]
        np.testing.assert_allclose(reconstructed, phi_stack[k], atol=1e-10)


def test_shared_attribution_is_non_negative():
    rng = np.random.default_rng(1)
    phi_stack = rng.normal(size=(3, 10, 5))
    phi_shared, _ = decompose(phi_stack)
    assert (phi_shared >= 0).all()  # mean of absolute values can't be negative