"""Walk-forward backtest of a pairs-trading rule (stage 3).

"Walk-forward" means: on every day we only use what was known on that day.

    1. Look at the last `window` days and fit  log(A) = alpha + beta * log(B).
    2. spread today = log(A) - beta * log(B) - alpha
       z-score      = spread today / (typical size of the spread in the window)
    3. z > +entry  -> the spread is unusually high: SHORT A, LONG B
       z < -entry  -> the spread is unusually low : LONG A, SHORT B
       The position is closed when z comes back close to 0 (|z| < exit).
    4. The trade is executed `delay` days AFTER the signal (default 1), at that
       day's closing price, and every trade pays a proportional cost.

Money convention: every trade has gross size 1 (long leg + short leg together),
the "capital" stays at 1, and profits are NOT reinvested.  So a total profit of
0.30 means "30 % of the size of one position".
"""
import numpy as np
import pandas as pd


def rolling_inputs(pair, window):
    """Rolling hedge ratio and z-score.  Row t uses only days t-window+1 ... t."""
    a, b = (np.log(pair[c]) for c in pair.columns)
    cov = a.rolling(window).cov(b)
    var_a = a.rolling(window).var()
    var_b = b.rolling(window).var()
    beta = cov / var_b
    alpha = a.rolling(window).mean() - beta * b.rolling(window).mean()
    # variance of the regression residuals inside the window: var(a) - beta*cov(a,b)
    resid_std = np.sqrt((var_a - beta * cov).clip(lower=0))
    spread = a - beta * b - alpha
    return pd.DataFrame({"beta": beta, "alpha": alpha, "spread": spread,
                         "resid_std": resid_std, "z": spread / resid_std})


def positions_from_z(z, entry_z=2.0, exit_z=0.5):
    """Turn the z-scores into positions: +1 (long spread), -1 (short spread), 0 (flat)."""
    state, out = 0, []
    for v in z:
        if np.isnan(v):
            state = 0
        elif state == 0:
            if v > entry_z:
                state = -1          # spread too high -> short it
            elif v < -entry_z:
                state = 1           # spread too low  -> long it
        elif state == 1:
            if v > -exit_z:
                state = 0           # came back close enough to zero
        else:
            if v < exit_z:
                state = 0
        out.append(state)
    return np.array(out)


def execute(price_a, price_b, states, betas, cost_rate, delay=1):
    """Run the trades day by day and record profit and loss.

    states[t] is the position wanted from the signal of day t, betas[t] the hedge
    ratio on that day.  It is traded at the close of day t + delay.
    Position size: gross 1, i.e. long  1/(1+|beta|) of A and short beta/(1+|beta|) of B
    (for a long spread).  The number of units is kept fixed until the trade is closed.
    cost_rate: cost per unit of money traded (0.001 = 0.1 % = 10 basis points).
    """
    pa, pb = np.asarray(price_a, float), np.asarray(price_b, float)
    n = len(pa)
    pnl = np.zeros(n)
    cost = np.zeros(n)
    traded = np.zeros(n)
    held = np.zeros(n, dtype=int)
    qa = qb = 0.0                       # units of A and B we hold
    cur = 0                             # current position (+1, -1, 0)
    trades, entry_i, trade_net = [], None, 0.0

    for u in range(n):
        held[u] = cur
        if u > 0:                       # profit of the units carried into day u
            pnl[u] = qa * (pa[u] - pa[u - 1]) + qb * (pb[u] - pb[u - 1])
        if cur != 0:
            trade_net += pnl[u]

        j = u - delay
        if j < 0 or states[j] == cur:
            continue
        target = int(states[j])
        if target == 0:
            new_qa = new_qb = 0.0
        else:
            b = betas[j]
            new_qa = target / (1 + abs(b)) / pa[u]
            new_qb = -target * b / (1 + abs(b)) / pb[u]
        traded[u] = abs(new_qa - qa) * pa[u] + abs(new_qb - qb) * pb[u]
        cost[u] = cost_rate * traded[u]

        if cur != 0:                    # close the open trade
            trade_net -= cost[u]
            trades.append({"entry_index": entry_i, "exit_index": u, "side": cur,
                           "days": u - entry_i, "net_pnl": trade_net, "open": False})
        if target != 0:                 # open a new one
            entry_i = u
            trade_net = -cost[u] if cur == 0 else 0.0
        qa, qb, cur = new_qa, new_qb, target

    if cur != 0:                        # still open at the end of the data
        trades.append({"entry_index": entry_i, "exit_index": n - 1, "side": cur,
                       "days": n - 1 - entry_i, "net_pnl": trade_net, "open": True})

    daily = pd.DataFrame({"held": held, "pnl_gross": pnl, "cost": cost,
                          "pnl_net": pnl - cost, "traded": traded})
    return daily, pd.DataFrame(trades)


def run_backtest(pair, window, entry_z, exit_z, cost_rate, delay=1):
    """Signals + execution for one pair.  Returns (daily table, trades table)."""
    inputs = rolling_inputs(pair, window)
    states = positions_from_z(inputs["z"].to_numpy(), entry_z, exit_z)
    daily, trades = execute(pair.iloc[:, 0], pair.iloc[:, 1], states,
                            inputs["beta"].to_numpy(), cost_rate, delay)
    daily.index = pair.index
    daily = daily.join(inputs[["z", "beta"]])
    daily["state"] = states
    if len(trades):
        trades["entry_date"] = pair.index[trades["entry_index"]]
        trades["exit_date"] = pair.index[trades["exit_index"]]
    return daily, trades


def performance(daily, trades, periods_per_year):
    """Summary numbers of one backtest (the warm-up period must already be cut off)."""
    net = daily["pnl_net"].to_numpy()
    equity = np.r_[0.0, np.cumsum(net)]
    drawdown = equity - np.maximum.accumulate(equity)
    years = len(net) / periods_per_year
    std = net.std(ddof=1)
    closed = trades[~trades["open"]] if len(trades) else trades
    return {
        "years": years,
        "total_pnl_gross": daily["pnl_gross"].sum(),
        "total_costs": daily["cost"].sum(),
        "total_pnl_net": net.sum(),
        "annual_return_net": net.mean() * periods_per_year,
        "annual_volatility": std * np.sqrt(periods_per_year),
        "sharpe_net": net.mean() / std * np.sqrt(periods_per_year) if std > 0 else np.nan,
        "max_drawdown": drawdown.min(),
        "n_trades": len(trades),
        "avg_days_in_trade": closed["days"].mean() if len(closed) else np.nan,
        "win_rate": (closed["net_pnl"] > 0).mean() if len(closed) else np.nan,
        "share_of_days_in_position": (daily["held"] != 0).mean(),
        "turnover_per_year": daily["traded"].sum() / years,
    }
