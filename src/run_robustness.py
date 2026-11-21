"""Stage 4: how much can we trust the stage 3 result?

    python src/run_robustness.py          (takes about a minute)

Everything below was decided in advance and EVERY result is reported, not only the
good ones.
  1. Sensitivity: the strategy is re-run for 3 window lengths x 3 entry levels
     (the window grid is half / the same / double the stage 3 window).
  2. Bootstrap: a 95 % confidence interval for the stage 3 Sharpe ratio.
  3. Permutation test: the same strategy is run on 500 fake pairs that have no
     structure over time (same daily returns, random order).  If the real pair is
     not clearly better than these, the result may be luck.
  4. Selection bias: if we had picked the BEST of the 9 settings, how good would
     that look on fake pairs?
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import ROOT, get_pair, load_prices
from robustness import (backtest_sharpe, block_bootstrap_sharpe, permutation_null,
                        sensitivity_grid)
from run_backtest import DELAY, ENTRY_Z, EXIT_Z, SETTINGS

# ---------------------------------------------------------------- fixed settings
GRID_ENTRIES = [1.5, 2.0, 2.5]
GRID_WINDOW_FACTORS = [0.5, 1.0, 2.0]     # times the stage 3 window
BLOCK_DAYS, N_BOOT, N_PERM, SEED = 20, 2000, 500, 42
# ---------------------------------------------------------------------------------


def main():
    out = ROOT / "results"
    prices = load_prices()
    grids, summary, nulls = [], [], {}

    for title, cfg in SETTINGS.items():
        pair = get_pair(prices, *cfg["assets"])
        ppy, cost = cfg["periods_per_year"], cfg["cost_bps"] * 1e-4
        windows = [int(round(cfg["window"] * f)) for f in GRID_WINDOW_FACTORS]

        # --- 1. sensitivity
        grid = sensitivity_grid(pair, windows, GRID_ENTRIES, EXIT_Z, cost, DELAY, ppy)
        grids.append(grid.assign(pair=title))

        # --- 2. bootstrap interval for the stage 3 setting
        obs, d, _ = backtest_sharpe(pair, cfg["window"], ENTRY_Z, EXIT_Z, cost, DELAY, ppy)
        boot = block_bootstrap_sharpe(d["pnl_net"].to_numpy(), ppy, BLOCK_DAYS, N_BOOT, SEED)
        lo, hi = np.nanpercentile(boot, [2.5, 97.5])

        # --- 3 and 4. permutation test on fake pairs
        configs = [(w, e, EXIT_Z) for w in windows for e in GRID_ENTRIES]
        null = permutation_null(pair, configs, cost, DELAY, ppy, N_PERM, SEED)
        base_col = configs.index((cfg["window"], ENTRY_Z, EXIT_Z))
        null_base = null[:, base_col]
        best_obs = grid["sharpe"].max()
        null_best = null.max(axis=1)
        nulls[title] = (null_base, obs)

        summary.append({
            "pair": title,
            "sharpe_stage3": obs,
            "bootstrap_ci_low": lo, "bootstrap_ci_high": hi,
            "bootstrap_share_above_0": float(np.nanmean(boot > 0)),
            "null_sharpe_median": np.median(null_base),
            "null_sharpe_95th_percentile": np.percentile(null_base, 95),
            "p_value_stage3_setting": float(np.mean(null_base >= obs)),
            "best_sharpe_of_9_settings": best_obs,
            "null_best_of_9_median": np.median(null_best),
            "p_value_best_of_9": float(np.mean(null_best >= best_obs)),
        })

    grid_all = pd.concat(grids, ignore_index=True)
    summary = pd.DataFrame(summary).set_index("pair")
    grid_all.to_csv(out / "stage4_sensitivity.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "stage4_summary.csv", encoding="utf-8-sig")

    # --- figure 6: Sharpe for every setting
    fig, axes = plt.subplots(1, len(SETTINGS), figsize=(11, 4))
    for ax, (title, g) in zip(axes, grid_all.groupby("pair", sort=False)):
        sh = g.pivot(index="window", columns="entry_z", values="sharpe")
        nt = g.pivot(index="window", columns="entry_z", values="n_trades")
        im = ax.imshow(sh.values, cmap="RdBu", vmin=-0.6, vmax=0.6, aspect="auto")
        ax.set_xticks(range(len(sh.columns)), [f"{c}" for c in sh.columns])
        ax.set_yticks(range(len(sh.index)), [f"{i}" for i in sh.index])
        ax.set_xlabel("entry level |z|")
        ax.set_ylabel("window (days)")
        for i in range(sh.shape[0]):
            for j in range(sh.shape[1]):
                ax.text(j, i, f"{sh.values[i, j]:.2f}\n({int(nt.values[i, j])} trades)",
                        ha="center", va="center", fontsize=9)
        ax.set_title(f"{title}\nSharpe ratio for every setting")
    fig.colorbar(im, ax=axes, shrink=0.8, label="Sharpe ratio")
    fig.savefig(out / "06_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- figure 7: real result against fake pairs
    fig, axes = plt.subplots(1, len(SETTINGS), figsize=(11, 4))
    for ax, (title, (null_base, obs)) in zip(axes, nulls.items()):
        ax.hist(null_base[~np.isnan(null_base)], bins=30, color="lightgray", edgecolor="gray")
        ax.axvline(obs, color="r", lw=2, label=f"real pair: {obs:.2f}")
        ax.set_title(f"{title}\nSharpe on {N_PERM} fake pairs (no structure over time)")
        ax.set_xlabel("Sharpe ratio")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "07_null_distribution.png", dpi=150)
    plt.close(fig)

    # --- print everything
    pd.set_option("display.width", 200)
    for title, g in grid_all.groupby("pair", sort=False):
        print(f"=== {title}: all 9 settings (exit |z| < {EXIT_Z}) ===")
        print(g.drop(columns="pair").round(3).to_string(index=False))
        print()
    print("=== Summary ===")
    print(summary.round(3).T.to_string())
    print(f"\nTables and figures written to {out}")


if __name__ == "__main__":
    main()
