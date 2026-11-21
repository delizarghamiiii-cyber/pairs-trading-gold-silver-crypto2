"""Loading and cleaning the price data (stage 2).

The analysis window is fixed so that the results never change by accident:
    START  the first day on which gold and silver both have a price
    END    the last COMPLETE day in the downloaded file.  The file was
           downloaded on 2026-11-14, and the last rows (13 and 14 November)
           are incomplete: crypto prices of the 14th were still moving, and the
           13th is missing for BTC and ETH.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
START = "2018-01-02"
END = "2026-11-12"


def load_prices(path=None, start=START, end=END):
    """Read data/prices.csv and keep only the days from `start` to `end`."""
    path = Path(path) if path else ROOT / "data" / "prices.csv"
    prices = pd.read_csv(path, index_col="date", parse_dates=True).sort_index()
    return prices.loc[start:end]


def get_pair(prices, a, b):
    """Prices of two assets, keeping only the days on which BOTH have a price.

    Crypto trades 7 days a week but gold and silver only on weekdays, so every
    pair is cleaned on its own: for gold/silver the weekends are dropped,
    for BTC/ETH nothing is dropped.
    """
    return prices[[a, b]].dropna()


def log_returns(pair):
    """Daily log returns: log(price today / price yesterday)."""
    return np.log(pair).diff().dropna()


def gap_report(pair, max_normal_gap_days):
    """How many times is there a longer pause between two prices than normal,
    and what is the longest pause in days?"""
    gaps = pair.index.to_series().diff().dt.days.dropna()
    return int((gaps > max_normal_gap_days).sum()), int(gaps.max())
