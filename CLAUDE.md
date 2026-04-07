# Market Neutral Long-Short Equity Backtest

## GitHub
Repo: https://github.com/Blake-Stanley/Quality-Value-Backtest

After any significant code changes, remind Blake to commit and push:
```bash
git add .
git commit -m "describe what changed"
git push
```

Never include a "Co-Authored-By: Claude" line in commit messages. Commit as Blake only.

## Strategy Overview
A market-neutral long-short equity strategy backtested against a Russell 1000 proxy universe.
The fund takes 175% long exposure in high-ranked stocks; the short exposure is set each month
to make the portfolio's net market beta equal to zero.

### Signal
Equal-weight composite z-score of three factors, computed cross-sectionally each quarter:
1. **Shareholder Yield** — TTM (dividends + net buybacks) / market cap
2. **Gross Profitability** — TTM (revenue - COGS) / total assets
3. **ROIC** — TTM NOPAT / average invested capital

Each factor is winsorized at 1/99 pct before z-scoring. Signal is lagged 4 months
from the fiscal quarter end date to avoid look-ahead bias.

### Universe
- Top 1000 stocks by lagged market cap each month (Russell 1000 proxy)
- CRSP share codes 10/11 only (common domestic equity)
- Lagged price > $3 (liquidity filter)

### Portfolio Construction
- **Long book**: top 100 stocks by composite z-score (175% gross weight, equal-weighted)
- **Short book**: top 100 stocks by short composite z-score, equal-weighted; gross weight `w_short` is solved each rebalance month for beta neutrality: `w_short = 1.75 × β_long_book / β_short_book`
- **Beta estimation**: trailing 12-month OLS beta vs FF market factor, Vasicek-adjusted (2/3 raw + 1/3 × 1.0), minimum 8 months of data required; implemented in `compute_trailing_betas()` and `compute_market_neutral_weights()`
- **Sector neutrality**: long and short books each stay within ±5pp of Russell 1000 proxy sector weights — **implemented** via `_sector_neutral_select()` in `backtest.py`; sectors mapped from SIC codes using `_SECTOR_BREAKS`
- **Rebalancing**: monthly
- **Return**: R = w_L × R_long − w_S × R_short, where w_L = 1.75 and w_S varies to achieve zero net beta

### Data Sources
- **Compustat** (`compustat_with_permno.parquet`) — quarterly fundamentals, PERMNO-matched
- **CRSP** (`crsp_m.dta`) — monthly returns, prices, shares outstanding
- **Fama-French** (`ff5_plus_mom.dta`) — market factor and risk-free rate for CAPM

---

## Folder Structure
- `run_all.py` — entry point; run from repo root
- `Code/` — all Python scripts (backtest engine, holdings export, table formatter)
- `Data/` — source data files (Compustat parquet, CRSP stata, FF factors)
- `Output/` — all generated files (CSVs, PNGs, Excel)
- `Cache/` — intermediate parquet cache, auto-generated, safe to delete

## How to Run
```bash
python run_all.py                  # full pipeline: backtest → table → holdings snapshot
python Code/export_holdings.py     # holdings snapshot only (uses cache if available)
python Code/make_table.py          # regenerate styled Excel table from existing metrics CSV
```

## Key Files
- `Code/backtest.py` — main engine; all strategy parameters defined as constants at the top
- `Code/export_holdings.py` — pulls latest long/short holdings with factor detail into Excel
- `Code/make_table.py` — formats `backtest_metrics.csv` into a styled Excel table
- `run_all.py` — runs all three scripts in order (run from repo root)

## Cache Behavior
`backtest.py` writes two files to `Cache/` after each run:
- `merged.parquet` — portfolio assignments (long/short/mid) for every stock-month
- `signal_components.parquet` — signal values + factor detail (yield, GP, ROIC) per stock

`export_holdings.py` loads from cache if available (~1-2s vs ~60s). Delete the Cache/ folder to force a rebuild.

## Output Files
| File | Description |
|------|-------------|
| `backtest_returns.csv` | Monthly returns for long, short, mkt neutral strategy (EW + VW), plus w_long/w_short |
| `backtest_metrics.csv` | Performance metrics table (input to make_table.py) |
| `backtest_metrics.txt` | Same metrics as plain text |
| `backtest.png` | Cumulative return chart (log scale, EW + VW side by side) |
| `Backtest_Table.xlsx` | Styled Excel metrics table |
| `holdings_snapshot.xlsx` | Current long/short holdings with factor scores |

## Config Constants (backtest.py)
| Constant | Value | Meaning |
|----------|-------|---------|
| `N_UNIVERSE` | 1000 | Universe size (Russell 1000 proxy) |
| `N_LONG` | 100 | Stocks in long book |
| `N_SHORT` | 100 | Stocks in short book |
| `SECTOR_TOL` | 0.05 | Sector neutrality tolerance (±5pp) |
| `REBALANCE_MONTHS` | 1 | Rebalance every N months |
| `LAG_MONTHS` | 4 | Signal availability lag from quarter end |
| `MIN_PRICE` | 3 | Minimum lagged price filter |
| `STALENESS_DAYS` | 365 | Max age of signal before dropping stock |
| `BETA_WINDOW` | 12 | Trailing months for beta estimation |
| `BETA_MIN_OBS` | 8 | Minimum observations required for beta estimate |
| `LONG_WEIGHT` | 1.75 | Fixed long gross weight; short weight solved for beta neutrality |
