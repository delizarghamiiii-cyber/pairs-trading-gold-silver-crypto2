"""Stage 2: look at the data before building any strategy.

    python src/explore.py

Writes figures and tables to results/ and prints a short summary.
Questions answered here:
  1. Do the two assets of a pair move together?            (figure 1, correlation)
  2. Is the spread between them stable, or does it drift?  (figure 2, tests)
  3. If it comes back to its mean, how fast?               (half-life)
  4. Is that true in every year, or only in some?          (figure 3, by-year table)
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import stats_tests
from data import END, ROOT, START, gap_report, get_pair, load_prices, log_returns
from pairs import half_life, hedge_ratio, spread

PAIRS = {
    "Gold / Silver (GLD, SLV)": ("GOLD", "SILVER"),
    "BTC / ETH": ("BTC", "ETH"),
}
# Longest pause between two prices that is still normal: a long weekend for the
# ETFs, one day for crypto (which trades every day).
NORMAL_GAP = {"GOLD": 5, "SILVER": 5, "BTC": 1, "ETH": 1}


def analyse_pair(pair):
    """Full-sample numbers for one pair, plus the spread series."""
    a, b = pair.columns
    logp = np.log(pair)
    alpha, beta = hedge_ratio(logp[a], logp[b])
    s = spread(logp[a], logp[b], beta, alpha)
    ret = log_returns(pair)
    eg = stats_tests.coint_test(logp[a], logp[b])
    n_gaps, longest = gap_report(pair, NORMAL_GAP[a])
    row = {
        "first_date": pair.index[0].date(),
        "last_date": pair.index[-1].date(),
        "n_prices": len(pair),
        "gaps_longer_than_normal": n_gaps,
        "longest_gap_days": longest,
        "return_correlation": ret.corr().iloc[0, 1],
        "hedge_ratio_beta": beta,
        f"adf_p_log_{a}": stats_tests.adf_test(logp[a])["pvalue"],
        f"adf_p_log_{b}": stats_tests.adf_test(logp[b])["pvalue"],
        "adf_p_log_ratio": stats_tests.adf_test(logp[a] - logp[b])["pvalue"],
        "engle_granger_stat": eg["stat"],
        "engle_granger_crit_5pct": eg["crit_5pct"],
        "engle_granger_p": eg["pvalue"],
        "half_life_days": half_life(s),
        "spread_now_in_std": (s.iloc[-1] - s.mean()) / s.std(),
    }
    return row, s


def by_year(pair):
    """The same relationship, re-estimated separately inside every calendar year."""
    a, b = pair.columns
    rows = []
    for year, chunk in pair.groupby(pair.index.year):
        if len(chunk) < 100:
            continue
        logp = np.log(chunk)
        alpha, beta = hedge_ratio(logp[a], logp[b])
        s = spread(logp[a], logp[b], beta, alpha)
        eg = stats_tests.coint_test(logp[a], logp[b])
        rows.append({
            "year": year,
            "n_prices": len(chunk),
            "return_correlation": log_returns(chunk).corr().iloc[0, 1],
            "hedge_ratio_beta": beta,
            "engle_granger_stat": eg["stat"],
            "engle_granger_crit_5pct": eg["crit_5pct"],
            "engle_granger_p": eg["pvalue"],
            "half_life_days": half_life(s),
        })
    return pd.DataFrame(rows).set_index("year")


def figure_prices(pairs, path):
    fig, axes = plt.subplots(1, len(pairs), figsize=(12, 4.2))
    for ax, (title, pair) in zip(axes, pairs.items()):
        (pair / pair.iloc[0]).plot(ax=ax, logy=True)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("price, first day = 1 (log scale)")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_spreads(spreads, path):
    fig, axes = plt.subplots(len(spreads), 1, figsize=(11, 3.6 * len(spreads)))
    for ax, (title, s) in zip(np.atleast_1d(axes), spreads.items()):
        ax.plot(s.index, s.values, lw=0.8)
        m, sd = s.mean(), s.std()
        ax.axhline(m, color="k", lw=1)
        for k in (-2, 2):
            ax.axhline(m + k * sd, color="r", lw=0.8, ls="--")
        ax.set_title(f"{title}: spread  log(A) - beta*log(B) - alpha   (lines: mean and +-2 std)")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_by_year(tables, path):
    fig, axes = plt.subplots(1, len(tables), figsize=(12, 4.2), sharey=False)
    for ax, (title, t) in zip(np.atleast_1d(axes), tables.items()):
        ax.bar(t.index.astype(str), t["engle_granger_stat"], color="tab:blue")
        ax.axhline(t["engle_granger_crit_5pct"].iloc[0], color="r", ls="--",
                   label="5% critical value")
        ax.set_title(f"{title}\nEngle-Granger statistic by year (below the line = cointegrated)")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    prices = load_prices()
    print(f"Window: {START} to {END}\n")

    pairs = {title: get_pair(prices, a, b) for title, (a, b) in PAIRS.items()}
    summary, spreads, yearly = {}, {}, {}
    for title, pair in pairs.items():
        summary[title], spreads[title] = analyse_pair(pair)
        yearly[title] = by_year(pair)

    summary = pd.DataFrame(summary)
    summary.to_csv(out / "stage2_summary.csv", encoding="utf-8-sig")
    pd.concat(yearly, names=["pair"]).to_csv(out / "stage2_by_year.csv", encoding="utf-8-sig")

    figure_prices(pairs, out / "01_prices.png")
    figure_spreads(spreads, out / "02_spreads.png")
    figure_by_year(yearly, out / "03_cointegration_by_year.png")

    pd.set_option("display.width", 200)
    print("=== Whole period ===")
    print(summary.map(lambda v: round(v, 4) if isinstance(v, float) else v).to_string())
    for title, t in yearly.items():
        print(f"\n=== {title}: year by year ===")
        print(t.round(4).to_string())
    print(f"\nFigures and tables written to {out}")


if __name__ == "__main__":
    main()
