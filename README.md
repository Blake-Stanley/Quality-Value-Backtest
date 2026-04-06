# Quality Value Backtest — Market Neutral Long-Short Equity

FIN 377 Group Assignment | UT Austin, Spring 2026

## Strategy

A market-neutral long-short equity strategy backtested against a Russell 1000 proxy universe.
Takes 130% long exposure in high-ranked stocks; short exposure is solved each rebalance month
to set the portfolio's net market beta to zero.

**Signal**: Equal-weight composite of three factors (Shareholder Yield, Gross Profitability, ROIC),
computed cross-sectionally each quarter and lagged 4 months to avoid look-ahead bias.

**Universe**: Top 1000 US common stocks by market cap (Russell 1000 proxy), price > $3, rebalanced monthly.

## How to Run

```bash
python run_all.py                  # full pipeline
python Code/export_holdings.py     # holdings snapshot only (uses cache if available)
```

## Structure

```
run_all.py  Entry point — runs full pipeline from repo root
Code/       Python scripts (backtest engine, holdings export, table formatter)
Data/       Source data (Compustat, CRSP, Fama-French) — not tracked in git
Cache/      Intermediate parquet files — auto-generated, safe to delete
Output/     Generated charts, CSVs, Excel files
```

## Charts

The cumulative return chart (`Output/backtest.png`) shows three lines per panel:
- **Long Book** — cumulative return of the long portfolio
- **Short Book** — inverted: goes up when the short book rises (i.e., when you are losing money on the short position)
- **Market Neutral Strategy** — combined long-short portfolio return

## Requirements

Python 3.x with: `pandas`, `numpy`, `scipy`, `statsmodels`, `openpyxl`, `pyarrow`
