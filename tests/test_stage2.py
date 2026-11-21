"""Checks for stage 2.  Run with:  python tests/test_stage2.py   (or: pytest)"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data import gap_report, get_pair, load_prices, log_returns  # noqa: E402
from pairs import half_life, hedge_ratio, spread  # noqa: E402


def test_load_prices_keeps_only_the_window():
    csv = ("date,BTC,ETH,GOLD,SILVER\n"
           "2018-01-01,1,1,,\n"
           "2018-01-02,2,2,10,5\n"
           "2018-01-03,3,3,11,6\n"
           "2018-01-04,4,4,12,7\n")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "prices.csv"
        p.write_text(csv)
        out = load_prices(p, start="2018-01-02", end="2018-01-03")
    assert list(out.index.strftime("%Y-%m-%d")) == ["2018-01-02", "2018-01-03"]


def test_get_pair_keeps_only_days_with_both_prices():
    idx = pd.date_range("2020-01-01", periods=5)
    prices = pd.DataFrame({"A": [1, 2, np.nan, 4, 5.0], "B": [1, np.nan, 3, 4, 5.0]}, index=idx)
    pair = get_pair(prices, "A", "B")
    assert list(pair.index.day) == [1, 4, 5]


def test_log_returns():
    pair = pd.DataFrame({"A": [100.0, 110.0], "B": [50.0, 50.0]}, index=pd.date_range("2020-01-01", periods=2))
    r = log_returns(pair)
    assert abs(r["A"].iloc[0] - np.log(1.1)) < 1e-12 and r["B"].iloc[0] == 0


def test_gap_report():
    idx = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-08", "2020-01-09"])
    n, longest = gap_report(pd.DataFrame({"A": range(4)}, index=idx), 3)
    assert n == 1 and longest == 6


def test_hedge_ratio_recovers_the_true_values():
    rng = np.random.default_rng(0)
    log_b = np.cumsum(rng.normal(0, 0.02, 3000))
    log_a = 0.3 + 1.7 * log_b + rng.normal(0, 0.01, 3000)
    alpha, beta = hedge_ratio(log_a, log_b)
    assert abs(beta - 1.7) < 0.02 and abs(alpha - 0.3) < 0.05


def test_spread_is_what_is_left_after_removing_b():
    log_a, log_b = np.array([3.0, 4.0]), np.array([1.0, 2.0])
    assert np.allclose(spread(log_a, log_b, beta=2.0, alpha=0.5), [0.5, -0.5])


def test_half_life_recovers_a_known_value():
    phi = 0.95                      # true half-life = ln(2) / -ln(0.95) = 13.5 days
    rng = np.random.default_rng(1)
    s = np.zeros(20000)
    for t in range(1, len(s)):
        s[t] = phi * s[t - 1] + rng.normal()
    assert abs(half_life(s) - np.log(2) / -np.log(phi)) < 1.0


def test_half_life_of_a_random_walk_is_huge_or_nan():
    # A random walk never comes back to a mean.  The estimate is either NaN or a
    # huge number of days (in 200 simulated walks the shortest typical value
    # was about 240 days), never a few days or weeks.
    rng = np.random.default_rng(2)
    h = half_life(np.cumsum(rng.normal(size=5000)))
    assert np.isnan(h) or h > 100


def test_statsmodels_tests_separate_stationary_from_random_walk():
    try:
        import statsmodels  # noqa: F401
    except ImportError:
        print("skip (statsmodels is not installed)", end=" ")
        return
    import stats_tests
    rng = np.random.default_rng(3)
    noise = rng.normal(size=2000)
    walk = np.cumsum(noise)
    stationary = np.zeros(2000)
    for t in range(1, 2000):
        stationary[t] = 0.5 * stationary[t - 1] + noise[t]
    assert stats_tests.adf_test(stationary)["pvalue"] < 0.01
    assert stats_tests.adf_test(walk)["pvalue"] > 0.01
    # cointegrated pair: B is a random walk, A = 2 B + small stationary noise
    other_walk = np.cumsum(rng.normal(size=2000))
    a = 2 * walk + rng.normal(0, 0.5, 2000)
    assert stats_tests.coint_test(a, walk)["pvalue"] < 0.01
    assert stats_tests.coint_test(other_walk, walk)["pvalue"] > 0.01


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"\n{len(tests)} checks passed")
