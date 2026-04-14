# Quality-Value Long-Short Equity Backtest

FIN 377 Group Assignment | UT Austin, Spring 2026

## Strategy

A long-short equity strategy backtested against a Russell 1000 proxy universe.
Takes 175% long exposure in high-quality/value stocks; short exposure targets
stocks exhibiting financial fragility (low FCF, high accruals, overvaluation,
dilution, deteriorating fundamentals). The short weight is solved each month to
achieve a target net portfolio beta of 0.30.

### Long Signal
Equal-weight composite z-score of three quality/value factors:
- **Shareholder Yield** — TTM (dividends + net buybacks) / market cap
- **Gross Profitability** — TTM (revenue - COGS) / total assets
- **ROIC** — TTM NOPAT / average invested capital

### Short Signal
Weighted composite z-score of seven fundamental failure factors:
- **FCF Yield** (negated, 1.5x weight) — TTM (operating CF - capex) / enterprise value
- **Accruals** — TTM (net income - operating CF) / avg assets
- **P/E Ratio** — price / TTM NOPAT per share (overvaluation)
- **Net External Financing** — TTM (equity issuance - buybacks + debt change) / assets
- **Piotroski F-Score** (negated) — 9-criteria fundamental quality score
- **Leverage** (0.5x weight) — (LT debt + ST debt) / total assets
- **Gross Profitability** (negated, 0.5x weight) — TTM (revenue - COGS) / total assets

All factors winsorized at 1/99 pct, z-scored cross-sectionally each quarter,
lagged 4 months from fiscal quarter end to avoid look-ahead bias.

### Portfolio Construction
- **Universe**: Top 1000 US common stocks by market cap, SHRCD 10/11, price > $3
- **Long book**: Top 100 by long composite (175% gross, equal-weighted)
- **Short book**: Top 100 by short composite (weight solved for 0.30 target beta)
- **Beta estimation**: 12-month trailing Vasicek-adjusted betas
- **Sector neutrality**: +/-5pp vs Russell 1000 proxy sector weights
- **Rebalancing**: Monthly

## How to Run

```bash
python run_all.py                  # full pipeline: backtest → table → charts → holdings snapshot
python Code/make_plots.py          # regenerate all charts (uses cache if available)
python Code/export_holdings.py     # holdings snapshot only (uses cache if available)
python Code/make_table.py          # regenerate styled Excel table
```

## Structure

```
run_all.py          Entry point -- runs full pipeline from repo root
Code/
  backtest.py       Main engine; all strategy parameters as constants at the top
  make_plots.py     Generates all charts to Output/Charts/
  make_table.py     Formats backtest_metrics.csv into styled Excel table
  export_holdings.py  Holdings snapshot with factor scores
Data/               Source data (Compustat, CRSP, Fama-French) -- not tracked in git
Cache/              Intermediate parquet files -- auto-generated, safe to delete
Output/
  Charts/           All PNG charts (cumulative returns, long_vs_short, drawdown, etc.)
  backtest_returns.csv
  backtest_metrics.csv
  Backtest_Table.xlsx
  holdings_snapshot.xlsx
```

## Requirements

Python 3.x with: `pandas`, `numpy`, `scipy`, `statsmodels`, `openpyxl`, `pyarrow`, `matplotlib`
