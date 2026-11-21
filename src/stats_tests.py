"""Statistical tests used in stage 2 (thin wrappers around statsmodels).

ADF test (Augmented Dickey-Fuller)
    Null hypothesis: the series has a unit root, i.e. it wanders like a random
    walk and does NOT come back to a mean.  A small p-value (below 0.05) means
    "it does come back", i.e. the series is stationary.

Engle-Granger test
    Tests whether two price series are cointegrated: does some combination
    log(A) - beta * log(B) stay stationary even though each price on its own
    wanders?  Null hypothesis: NOT cointegrated.  Small p-value: cointegrated.
"""
import numpy as np


def adf_test(x):
    """ADF test with a constant; the number of lags is chosen by AIC."""
    from statsmodels.tsa.stattools import adfuller

    stat, pvalue, lags, _, crit, _ = adfuller(np.asarray(x, float),
                                              regression="c", autolag="AIC")
    return {"stat": float(stat), "pvalue": float(pvalue), "lags": int(lags),
            "crit_5pct": float(crit["5%"])}


def coint_test(y, x):
    """Engle-Granger cointegration test of the series y and x."""
    from statsmodels.tsa.stattools import coint

    stat, pvalue, crit = coint(np.asarray(y, float), np.asarray(x, float),
                               trend="c", autolag="aic")
    return {"stat": float(stat), "pvalue": float(pvalue),
            "crit_5pct": float(crit[1])}
