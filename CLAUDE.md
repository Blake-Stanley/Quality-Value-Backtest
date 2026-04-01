# 130/30 Long-Short Equity Backtest

## GitHub
Repo: https://github.com/Blake-Stanley/Quality-Value-Backtest

After any significant code changes, remind Blake to commit and push:
```bash
git add .
git commit -m "describe what changed"
git push
```

## Strategy Overview
A 130/30 long-short equity strategy backtested against an S&P 500 proxy universe.
The fund takes 130% long exposure in high-ranked stocks and 30% short exposure in
low-ranked stocks, for 100% net equity exposure and 160% gross exposure.

### Signal
Equal-weight composite z-score of three factors, computed cross-sectionally each quarter:
1. **Shareholder Yield** — TTM (dividends + net buybacks) / market cap
2. **Gross Profitability** — TTM (revenue - COGS) / total assets
3. **ROIC** — TTM NOPAT / average invested capital

Each factor is winsorized at 1/99 pct before z-scoring. Signal is lagged 4 months
from the fiscal quarter end date to avoid look-ahead bias.

### Universe
- Top 500 stocks by lagged market cap each month (S&P 500 proxy)
- CRSP share codes 10/11 only (common domestic equity)
- Lagged price > $3 (liquidity filter)

### Portfolio Construction
- **Long book**: top 100 stocks by composite z-score (130% gross weight, equal-weighted)
- **Short book**: bottom 100 stocks by composite z-score (30% gross weight, equal-weighted)
- **Sector neutrality**: long and short books each stay within ±5pp of S&P 500 sector weights
- **Rebalancing**: quarterly
- **Return**: R = 1.30 × R_long − 0.30 × R_short

### Data Sources
- **Compustat** (`compustat_with_permno.parquet`) — quarterly fundamentals, PERMNO-matched
- **CRSP** (`crsp_m.dta`) — monthly returns, prices, shares outstanding
- **Fama-French** (`ff5_plus_mom.dta`) — market factor and risk-free rate for CAPM

---

## Folder Structure
- `Code/` — all Python scripts; run everything from here
- `Data/` — source data files (Compustat parquet, CRSP stata, FF factors)
- `Output/` — all generated files (CSVs, PNGs, Excel)
- `Code/Cache/` — intermediate parquet cache, auto-generated, safe to delete

## How to Run
```bash
python run_all.py            # full pipeline: backtest → table → holdings snapshot
python run_all.py --refresh  # force reload from source data (bypasses cache)
python export_holdings.py    # holdings snapshot only (uses cache if available)
python make_table.py         # regenerate styled Excel table from existing metrics CSV
```

## Key Files
- `backtest_130_30.py` — main engine; all strategy parameters defined as constants at the top
- `export_holdings.py` — pulls latest long/short holdings with factor detail into Excel
- `make_table.py` — formats `130_30_backtest_metrics.csv` into a styled Excel table
- `run_all.py` — runs all three scripts in order

## Cache Behavior
`backtest_130_30.py` writes two files to `Code/Cache/` after each run:
- `merged.parquet` — portfolio assignments (long/short/mid) for every stock-month
- `signal_components.parquet` — signal values + factor detail (yield, GP, ROIC) per stock

`export_holdings.py` loads from cache instead of rerunning the full pipeline (~1-2s vs ~60s).
Pass `--refresh` to force a full rebuild from source data if Compustat/CRSP has been updated.

## Output Files
| File | Description |
|------|-------------|
| `130_30_backtest_returns.csv` | Monthly returns for long, short, 130/30 (EW + VW) |
| `130_30_backtest_metrics.csv` | Performance metrics table (input to make_table.py) |
| `130_30_backtest_metrics.txt` | Same metrics as plain text |
| `130_30_backtest.png` | Cumulative return chart (log scale, EW + VW side by side) |
| `130_30_Backtest_Table.xlsx` | Styled Excel metrics table |
| `130_30_holdings_snapshot.xlsx` | Current long/short holdings with factor scores |

## Config Constants (backtest_130_30.py)
| Constant | Value | Meaning |
|----------|-------|---------|
| `N_UNIVERSE` | 500 | Universe size (S&P 500 proxy) |
| `N_LONG` | 100 | Stocks in long book |
| `N_SHORT` | 100 | Stocks in short book |
| `SECTOR_TOL` | 0.05 | Sector neutrality tolerance (±5pp) |
| `REBALANCE_MONTHS` | 3 | Rebalance every N months |
| `LAG_MONTHS` | 4 | Signal availability lag from quarter end |
| `MIN_PRICE` | 3 | Minimum lagged price filter |
| `STALENESS_DAYS` | 365 | Max age of signal before dropping stock |
