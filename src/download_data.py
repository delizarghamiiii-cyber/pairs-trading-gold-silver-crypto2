"""Download daily prices from Yahoo Finance and save them to data/prices.csv.

Run this ONCE on your own computer (it needs internet, maybe a VPN):

    python src/download_data.py

Assets
  BTC, ETH   spot prices in USD (trade 7 days a week)
  GOLD       GLD, the SPDR Gold Shares ETF   (trades on weekdays only)
  SILVER     SLV, the iShares Silver Trust ETF (trades on weekdays only)

Gold and silver ETFs are used instead of futures because futures contracts
expire and their price history has jumps when the contract is rolled.
Because crypto trades on weekends and the ETFs do not, the file has empty cells
for GOLD/SILVER on weekends; the analysis code deals with that per pair.
"""
from pathlib import Path

import pandas as pd
import yfinance as yf

TICKERS = {"BTC": "BTC-USD", "ETH": "ETH-USD", "GOLD": "GLD", "SILVER": "SLV"}
START = "2018-01-01"

ROOT = Path(__file__).resolve().parent.parent


def main():
    raw = yf.download(list(TICKERS.values()), start=START, auto_adjust=True,
                      progress=False)
    if raw.empty:
        raise SystemExit("No data came back. Check your internet connection "
                         "(try turning the VPN on) and run the script again.")

    prices = raw["Close"].rename(columns={v: k for k, v in TICKERS.items()})
    prices = prices[list(TICKERS)]
    prices.index.name = "date"

    # Today's row is incomplete (the day is not over, prices still move): drop it.
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    prices = prices[prices.index < today]

    out = ROOT / "data"
    out.mkdir(exist_ok=True)
    prices.to_csv(out / "prices.csv")

    print(f"Saved {len(prices)} rows to {out / 'prices.csv'}")
    print("\nFirst and last date of every asset:")
    for name in prices:
        s = prices[name].dropna()
        print(f"  {name:7s} {s.index.min().date()}  ->  {s.index.max().date()}  ({len(s)} prices)")
    print("\nLast 5 rows:")
    print(prices.tail().round(2).to_string())


if __name__ == "__main__":
    main()
