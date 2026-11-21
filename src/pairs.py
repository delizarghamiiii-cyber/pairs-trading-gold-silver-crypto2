"""Building blocks for pairs trading (stage 2).

Idea: if two assets A and B move together in the long run, then
    spread = log(A) - beta * log(B) - alpha
should keep coming back to zero.  `beta` is the hedge ratio: how many units of
log(B) we hold against one unit of log(A).
"""
import numpy as np


def hedge_ratio(log_a, log_b):
    """Fit  log(A) = alpha + beta * log(B) + error  by least squares."""
    log_a, log_b = np.asarray(log_a, float), np.asarray(log_b, float)
    X = np.column_stack([np.ones(len(log_b)), log_b])
    (alpha, beta), *_ = np.linalg.lstsq(X, log_a, rcond=None)
    return float(alpha), float(beta)


def spread(log_a, log_b, beta, alpha=0.0):
    """The spread (what is left of A after removing B)."""
    return log_a - beta * log_b - alpha


def half_life(s):
    """Typical time (in days) for the spread to close half of its gap to the mean.

    Fit  s[t] - s[t-1] = c + b * s[t-1] + noise.  With phi = 1 + b we have
    s[t] - mean = phi * (s[t-1] - mean) + noise, so the gap shrinks by the
    factor phi every day and the half-life is -ln(2) / ln(phi).
    If phi is not between 0 and 1 the spread does not come back to its mean
    and we return NaN.  A spread with no real mean reversion (like a random
    walk) gives a huge half-life, hundreds or thousands of days, or NaN.
    """
    s = np.asarray(s, float)
    lagged, change = s[:-1], np.diff(s)
    X = np.column_stack([np.ones(len(lagged)), lagged])
    (_, b), *_ = np.linalg.lstsq(X, change, rcond=None)
    phi = 1.0 + b
    if 0.0 < phi < 1.0:
        return float(-np.log(2.0) / np.log(phi))
    return float("nan")
