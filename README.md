# Quality Value Backtest — 130/30 Long-Short Equity

FIN 377 Group Assignment | UT Austin, Spring 2026

## Strategy

A 130/30 long-short equity strategy backtested against an S&P 500 proxy universe.
Takes 130% long exposure in high-ranked stocks and 30% short exposure in low-ranked stocks,
resulting in 100% net equity exposure and 160% gross exposure.

**Signal**: Equal-weight composite of three factors (Shareholder Yield, Gross Profitability, ROIC),
computed cross-sectionally each quarter and lagged 4 months to avoid look-ahead bias.

**Universe**: Top 500 US common stocks by market cap, price > $3, rebalanced quarterly.

## How to Run

```bash
python run_all.py            # full pipeline
python run_all.py --refresh  # force reload from source data
```

## Structure

```
run_all.py  Entry point — runs full pipeline from repo root
Code/       Python scripts (backtest engine, holdings export, table formatter)
Data/       Source data (Compustat, CRSP, Fama-French) — not tracked in git
Cache/      Intermediate parquet files — auto-generated, safe to delete
Output/     Generated charts, CSVs, Excel files
```

## Requirements

Python 3.x with: `pandas`, `numpy`, `scipy`, `statsmodels`, `openpyxl`, `pyarrow`
