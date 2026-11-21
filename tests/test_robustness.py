"""Checks for stage 4.  Run with:  python tests/test_robustness.py   (or: pytest)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from robustness import (backtest_sharpe, block_bootstrap_sharpe, permutation_null,  # noqa: E402
                        permuted_pair, sensitivity_grid, sharpe)


def make_pair(n=800, seed=0):
    rng = np.random.default_rng(seed)
    common = rng.normal(0, 0.01, n)
    log_b = np.cumsum(common + rng.normal(0, 0.005, n)) + 4
    log_a = np.cumsum(common + rng.normal(0, 0.005, n)) + 5
    return pd.DataFrame({"A": np.exp(log_a), "B": np.exp(log_b)},
                        index=pd.date_range("2020-01-01", periods=n))


def make_mean_reverting_pair(n=3000, seed=5):
    rng = np.random.default_rng(seed)
    log_b = np.cumsum(rng.normal(0, 0.01, n)) + 4
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = 0.9 * s[t - 1] + rng.normal(0, 0.01)
    return pd.DataFrame({"A": np.exp(0.8 * log_b + s), "B": np.exp(log_b)},
                        index=pd.date_range("2010-01-01", periods=n))


def test_sharpe_formula():
    x = np.array([0.02, -0.01, 0.01])
    assert np.isclose(sharpe(x, 252), x.mean() / x.std(ddof=1) * np.sqrt(252))


def test_permuted_pair_keeps_returns_and_correlation_but_not_the_order():
    pair = make_pair()
    fake = permuted_pair(pair, np.random.default_rng(1))
    r1, r2 = np.log(pair).diff().dropna(), np.log(fake).diff().dropna()
    assert np.allclose(pair.iloc[0], fake.iloc[0])                          # same start
    assert np.allclose(np.sort(r1["A"]), np.sort(r2["A"]))                  # same returns
    assert np.isclose(r1.corr().iloc[0, 1], r2.corr().iloc[0, 1])           # same correlation
    assert not np.allclose(r1.to_numpy(), r2.to_numpy())                    # different order


def test_bootstrap_centres_on_the_observed_sharpe():
    rng = np.random.default_rng(0)
    net = rng.normal(0.001, 0.01, 2000)
    boot = block_bootstrap_sharpe(net, 252, block=20, n_boot=1000)
    assert abs(np.nanmean(boot) - sharpe(net, 252)) < 0.3
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    assert lo < sharpe(net, 252) < hi


def test_sensitivity_grid_has_one_row_per_combination():
    pair = make_pair()
    grid = sensitivity_grid(pair, [60, 120], [1.5, 2.0, 2.5], 0.5, 0.0005, 1, 252)
    assert len(grid) == 6 and set(grid["window"]) == {60, 120}


def test_real_mean_reversion_beats_the_permutation_null():
    pair = make_mean_reverting_pair()
    observed = backtest_sharpe(pair, 252, 2.0, 0.5, 0.0005, 1, 252)[0]
    null = permutation_null(pair, [(252, 2.0, 0.5)], 0.0005, 1, 252, n_perm=40)[:, 0]
    assert observed > np.percentile(null, 95)


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"\n{len(tests)} checks passed")
