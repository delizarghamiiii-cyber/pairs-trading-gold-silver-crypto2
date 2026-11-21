"""Stage 3: run the walk-forward backtest once, with settings fixed IN ADVANCE.

    python src/run_backtest.py

The settings below were chosen before looking at any backtest result and are not
changed after seeing the numbers (changing them until the result looks good is
"overfitting").  Stage 4 will show what happens for many other settings, all of
them, not only the best one.

Assumptions (all of them make the result look BETTER than reality, not worse):
  * the short leg can be borrowed for free (no borrowing or funding fees)
  * we can always trade at the closing price, with only the proportional cost below
  * for crypto: shorting is possible (futures / margin)
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from backtest import performance, run_backtest
from data import ROOT, get_pair, load_prices

# ---------------------------------------------------------------- fixed settings
ENTRY_Z = 2.0          # enter when the spread is 2 standard deviations from its mean
EXIT_Z = 0.5           # leave when it is back within 0.5 standard deviations
DELAY = 1              # trade at the close of the day AFTER the signal
SETTINGS = {
    # window = about one year of prices; cost in basis points (1 bp = 0.01 %) per trade
    "Gold / Silver (GLD, SLV)": dict(assets=("GOLD", "SILVER"), window=252,
                                     periods_per_year=252, cost_bps=5),
    "BTC / ETH": dict(assets=("BTC", "ETH"), window=365,
                      periods_per_year=365, cost_bps=10),
}
COST_SCENARIOS = {"no costs": 0.0, "base costs": 1.0, "double costs": 2.0}
# ---------------------------------------------------------------------------------


def main():
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    prices = load_prices()

    rows, curves, zs, trade_tables = [], {}, {}, []
    for title, cfg in SETTINGS.items():
        pair = get_pair(prices, *cfg["assets"])
        for scenario, mult in COST_SCENARIOS.items():
            daily, trades = run_backtest(pair, cfg["window"], ENTRY_Z, EXIT_Z,
                                         cfg["cost_bps"] * 1e-4 * mult, DELAY)
            d = daily.iloc[cfg["window"]:]                         # drop the warm-up period
            t = trades[trades["entry_index"] >= cfg["window"]] if len(trades) else trades
            perf = performance(d, t, cfg["periods_per_year"])
            rows.append({"pair": title, "scenario": scenario, **perf})
            curves[(title, scenario)] = d["pnl_net"].cumsum()
            if scenario == "base costs":
                zs[title] = daily["z"]
                if len(t):
                    trade_tables.append(t.assign(pair=title))

    metrics = pd.DataFrame(rows).set_index(["pair", "scenario"])
    metrics.to_csv(out / "stage3_metrics.csv", encoding="utf-8-sig")
    pd.concat(trade_tables).to_csv(out / "stage3_trades.csv", index=False, encoding="utf-8-sig")

    # figure 4: cumulative profit
    fig, axes = plt.subplots(len(SETTINGS), 1, figsize=(11, 4 * len(SETTINGS)))
    for ax, title in zip(axes, SETTINGS):
        for scenario in COST_SCENARIOS:
            curves[(title, scenario)].plot(ax=ax, label=scenario, lw=1.2)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"{title}: cumulative profit (1.0 = the size of one position)")
        ax.set_xlabel("")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "04_equity_curves.png", dpi=150)
    plt.close(fig)

    # figure 5: the z-score signal
    fig, axes = plt.subplots(len(SETTINGS), 1, figsize=(11, 3.6 * len(SETTINGS)))
    for ax, title in zip(axes, SETTINGS):
        zs[title].plot(ax=ax, lw=0.7)
        for lvl, style in ((ENTRY_Z, "r--"), (-ENTRY_Z, "r--"), (EXIT_Z, "g:"), (-EXIT_Z, "g:")):
            ax.axhline(lvl, color=style[0], ls=style[1:], lw=0.9)
        ax.set_title(f"{title}: z-score (red = entry level, green = exit level)")
        ax.set_xlabel("")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "05_zscores.png", dpi=150)
    plt.close(fig)

    pd.set_option("display.width", 200)
    show = metrics.round(4).T
    print(f"Settings: enter |z| > {ENTRY_Z}, exit |z| < {EXIT_Z}, trade {DELAY} day after the signal\n")
    for title in SETTINGS:
        print(f"=== {title}  (window {SETTINGS[title]['window']} days, "
              f"base cost {SETTINGS[title]['cost_bps']} bp per trade) ===")
        print(show[title].to_string())
        print()
    print(f"Tables and figures written to {out}")


if __name__ == "__main__":
    main()
