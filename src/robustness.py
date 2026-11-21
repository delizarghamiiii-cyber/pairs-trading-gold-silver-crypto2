"""Robustness checks for the backtest (stage 4).

A backtest gives ONE number (for example a Sharpe ratio of 0.08).  Three questions:
  1. Does the result depend on the exact settings?          -> sensitivity_grid
  2. How uncertain is the number?                           -> block_bootstrap_sharpe
  3. Could a strategy with no real edge get a number this
     good by pure luck?                                     -> permutation_null
"""
import numpy as np
import pandas as pd

from backtest import run_backtest


def sharpe(net_pnl, periods_per_year):
    """Annualised Sharpe ratio of a series of daily profits."""
    net_pnl = np.asarray(net_pnl, float)
    std = net_pnl.std(ddof=1)
    return net_pnl.mean() / std * np.sqrt(periods_per_year) if std > 0 else np.nan


def backtest_sharpe(pair, window, entry_z, exit_z, cost_rate, delay, periods_per_year):
    daily, trades = run_backtest(pair, window, entry_z, exit_z, cost_rate, delay)
    d = daily.iloc[window:]
    return sharpe(d["pnl_net"], periods_per_year), d, trades


def sensitivity_grid(pair, windows, entries, exit_z, cost_rate, delay, periods_per_year):
    """Run the strategy for EVERY combination of window and entry level."""
    rows = []
    for w in windows:
        for e in entries:
            s, d, trades = backtest_sharpe(pair, w, e, exit_z, cost_rate, delay, periods_per_year)
            t = trades[trades["entry_index"] >= w] if len(trades) else trades
            equity = np.r_[0.0, d["pnl_net"].cumsum().to_numpy()]
            rows.append({"window": w, "entry_z": e, "sharpe": s,
                         "total_pnl_net": d["pnl_net"].sum(),
                         "max_drawdown": (equity - np.maximum.accumulate(equity)).min(),
                         "n_trades": len(t)})
    return pd.DataFrame(rows)


def block_bootstrap_sharpe(net_pnl, periods_per_year, block=20, n_boot=2000, seed=42):
    """Sharpe ratios of many re-sampled versions of the profit series.

    Profits on neighbouring days are not independent (a trade lasts weeks), so we
    re-sample whole BLOCKS of `block` consecutive days instead of single days
    (circular block bootstrap).
    """
    net = np.asarray(net_pnl, float)
    n = len(net)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)) % n
    samples = net[idx.reshape(n_boot, -1)[:, :n]]
    std = samples.std(axis=1, ddof=1)
    out = np.full(n_boot, np.nan)
    ok = std > 0
    out[ok] = samples[ok].mean(axis=1) / std[ok] * np.sqrt(periods_per_year)
    return out


def permuted_pair(pair, rng):
    """A fake pair made from the SAME daily returns in a RANDOM order.

    Each day's pair of returns is kept together, so every asset keeps its own
    distribution of returns and the two keep their correlation.  What is destroyed
    is any structure over time, such as the spread coming back to its mean.
    """
    logp = np.log(pair)
    first = logp.iloc[0].to_numpy()
    rets = logp.diff().dropna().to_numpy()
    shuffled = rets[rng.permutation(len(rets))]
    path = np.vstack([first, first + np.cumsum(shuffled, axis=0)])
    return pd.DataFrame(np.exp(path), index=pair.index, columns=pair.columns)


def permutation_null(pair, configs, cost_rate, delay, periods_per_year, n_perm=500, seed=42):
    """Sharpe ratios of the strategy on fake (permuted) pairs.

    configs: list of (window, entry_z, exit_z).  Returns an array of shape
    (n_perm, len(configs)): one Sharpe ratio per fake pair and setting.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros((n_perm, len(configs)))
    for i in range(n_perm):
        fake = permuted_pair(pair, rng)
        for j, (w, e, x) in enumerate(configs):
            out[i, j] = backtest_sharpe(fake, w, e, x, cost_rate, delay, periods_per_year)[0]
    return out
