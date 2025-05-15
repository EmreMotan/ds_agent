import numpy as np
from typing import Dict


def bootstrap_mean_diff(
    a: np.ndarray, b: np.ndarray, n: int = 5000, ci: float = 0.95, two_tailed: bool = True
) -> dict:
    """
    Bootstrap the mean difference between two samples, compute CI and p-value.
    Returns dict with means, diff, CI bounds, and p-value.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Inputs must be 1D arrays")
    rng = np.random.default_rng(42)
    boot_diffs = []
    for _ in range(n):
        a_samp = rng.choice(a, size=a.size, replace=True)
        b_samp = rng.choice(b, size=b.size, replace=True)
        boot_diffs.append(a_samp.mean() - b_samp.mean())
    boot_diffs = np.array(boot_diffs)
    mean_a = a.mean()
    mean_b = b.mean()
    diff = mean_a - mean_b
    alpha = 1 - ci
    lower = np.percentile(boot_diffs, 100 * (alpha / 2))
    upper = np.percentile(boot_diffs, 100 * (1 - alpha / 2))
    # p-value: fraction of bootstraps crossing zero (two-tailed if specified)
    if two_tailed:
        p = 2 * min((boot_diffs < 0).mean(), (boot_diffs > 0).mean())
    else:
        p = (boot_diffs < 0).mean() if diff > 0 else (boot_diffs > 0).mean()
    return {
        "mean_a": mean_a,
        "mean_b": mean_b,
        "diff": diff,
        "ci_lower": lower,
        "ci_upper": upper,
        "p_value": p,
        "n_boot": n,
    }
