"""Checks for stage 3.  Run with:  python tests/test_backtest.py   (or: pytest)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from backtest import (execute, performance, positions_from_z,  # noqa: E402
                      rolling_inputs, run_backtest)
from pairs import hedge_ratio  # noqa: E402


def make_pair(n=600, seed=0):
    rng = np.random.default_rng(seed)
    log_b = np.cumsum(rng.normal(0, 0.01, n)) + 4
    log_a = 0.5 * log_b + np.cumsum(rng.normal(0, 0.01, n)) + 1
    idx = pd.date_range("2020-01-01", periods=n)
    return pd.DataFrame({"A": np.exp(log_a), "B": np.exp(log_b)}, index=idx)


def test_positions_from_z_follow_the_rules():
    z = [0, 1, 2.5, 1.5, 0.4, 0.2, -2.5, -1, -0.3]
    #     flat, flat, enter short, still short, exit, flat, enter long, still long, exit
    assert list(positions_from_z(z, 2.0, 0.5)) == [0, 0, -1, -1, 0, 0, 1, 1, 0]


def test_positions_are_flat_while_z_is_missing():
    assert list(positions_from_z([np.nan, np.nan, 3.0])) == [0, 0, -1]


def test_rolling_inputs_match_a_direct_regression():
    pair = make_pair()
    window, t = 100, 350
    r = rolling_inputs(pair, window).iloc[t]
    la = np.log(pair["A"].iloc[t - window + 1: t + 1]).to_numpy()
    lb = np.log(pair["B"].iloc[t - window + 1: t + 1]).to_numpy()
    alpha, beta = hedge_ratio(la, lb)
    resid = la - alpha - beta * lb
    assert abs(r["beta"] - beta) < 1e-9 and abs(r["alpha"] - alpha) < 1e-9
    assert abs(r["resid_std"] - resid.std(ddof=1)) < 1e-9
    assert abs(r["z"] - resid[-1] / resid.std(ddof=1)) < 1e-9


def test_no_lookahead_changing_the_future_does_not_change_the_past():
    pair = make_pair()
    changed = pair.copy()
    changed.iloc[401:] = changed.iloc[401:] * 3.0 + 50.0          # wreck the future
    a = rolling_inputs(pair, 100)["z"].iloc[:401].to_numpy()
    b = rolling_inputs(changed, 100)["z"].iloc[:401].to_numpy()
    assert np.allclose(a, b, equal_nan=True)
    d1, _ = run_backtest(pair, 100, 2.0, 0.5, 0.001)
    d2, _ = run_backtest(changed, 100, 2.0, 0.5, 0.001)
    assert (d1["held"].iloc[:402].to_numpy() == d2["held"].iloc[:402].to_numpy()).all()
    # profit of day 401 itself uses the (changed) price of day 401, so compare up to day 400
    assert np.allclose(d1["pnl_net"].iloc[:401], d2["pnl_net"].iloc[:401])


def test_execute_hand_calculation_without_delay():
    # A rises 10 % on day 2, B is flat; beta = 1 -> 0.5 in A long, 0.5 in B short
    a, b = [100.0, 110.0, 121.0, 121.0], [50.0] * 4
    daily, trades = execute(a, b, states=[0, 1, 0, 0], betas=[1.0] * 4, cost_rate=0.001, delay=0)
    assert np.isclose(daily["pnl_gross"].sum(), 0.05)             # 0.5/110 units * 11
    assert np.isclose(daily["cost"].sum(), 0.001 + 0.00105)       # entry 1.0, exit 0.55 + 0.5
    assert np.isclose(daily["pnl_net"].sum(), 0.05 - 0.00205)
    assert len(trades) == 1 and np.isclose(trades["net_pnl"].iloc[0], 0.04795)


def test_execute_with_one_day_delay_misses_the_move():
    a, b = [100.0, 110.0, 121.0, 121.0], [50.0] * 4
    daily, _ = execute(a, b, states=[0, 1, 0, 0], betas=[1.0] * 4, cost_rate=0.001, delay=1)
    assert np.isclose(daily["pnl_gross"].sum(), 0.0)
    assert np.isclose(daily["cost"].sum(), 0.002)


def test_zero_cost_means_net_equals_gross():
    pair = make_pair(seed=3)
    daily, _ = run_backtest(pair, 100, 2.0, 0.5, 0.0)
    assert np.allclose(daily["pnl_net"], daily["pnl_gross"])


def test_performance_numbers():
    daily = pd.DataFrame({"pnl_gross": [0.02, -0.01, 0.01], "cost": 0.0,
                          "pnl_net": [0.02, -0.01, 0.01], "traded": 0.0, "held": 0})
    trades = pd.DataFrame(columns=["days", "net_pnl", "open"])
    p = performance(daily, trades, 252)
    assert np.isclose(p["total_pnl_net"], 0.02) and np.isclose(p["max_drawdown"], -0.01)


def test_strategy_profits_when_the_spread_really_mean_reverts():
    rng = np.random.default_rng(5)
    n = 3000
    log_b = np.cumsum(rng.normal(0, 0.01, n)) + 4
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = 0.9 * s[t - 1] + rng.normal(0, 0.01)               # half-life about 6.6 days
    pair = pd.DataFrame({"A": np.exp(0.8 * log_b + s), "B": np.exp(log_b)},
                        index=pd.date_range("2010-01-01", periods=n))
    daily, trades = run_backtest(pair, 252, 2.0, 0.5, 0.0)
    assert daily["pnl_gross"].iloc[252:].sum() > 0.2 and len(trades) > 20


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"\n{len(tests)} checks passed")
