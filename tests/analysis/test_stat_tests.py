import numpy as np
from ds_agent.analysis.stat_tests import bootstrap_mean_diff


def test_bootstrap_mean_diff_basic():
    np.random.seed(0)
    a = np.random.normal(10, 2, 100)
    b = np.random.normal(12, 2, 100)
    result = bootstrap_mean_diff(a, b, n=1000, ci=0.95)
    # Check keys
    for key in ["mean_a", "mean_b", "diff", "ci_lower", "ci_upper", "p_value", "n_boot"]:
        assert key in result
    # Check value ranges
    assert abs(result["mean_a"] - 10) < 1
    assert abs(result["mean_b"] - 12) < 1
    assert result["ci_lower"] < result["ci_upper"]
    assert 0 <= result["p_value"] <= 1
    # Check that CI contains the observed diff
    assert result["ci_lower"] <= result["diff"] <= result["ci_upper"]
