# Quality-Value Long-Short Equity Backtest

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
A long-short equity strategy backtested against a Russell 1000 proxy universe.
The fund takes 175% long exposure in high-quality/value stocks; short exposure targets
financially fragile companies. The short weight is solved each month to achieve a target
net portfolio beta of 0.30 (slight net long bias).

### Long Signal
Equal-weight composite z-score of three quality/value factors, computed cross-sectionally each quarter:
1. **Shareholder Yield** — TTM (dividends + net buybacks) / market cap
2. **Gross Profitability** — TTM (revenue - COGS) / total assets
3. **ROIC** — TTM NOPAT / average invested capital

### Short Signal
Weighted composite z-score of seven fundamental failure factors (inspired by Empirical Research
Partners Failure Model), computed cross-sectionally each quarter:
1. **FCF Yield** (negated, 1.5x weight) — TTM (operating CF - capex) / enterprise value
2. **Accruals** — TTM (net income - operating CF) / avg assets
3. **P/E Ratio** — price / TTM NOPAT per share (overvaluation)
4. **Net External Financing** — TTM (equity issuance - buybacks + debt change) / assets
5. **Piotroski F-Score** (negated) — 9-criteria fundamental quality score (0-9)
6. **Leverage** (0.5x weight) — (LT debt + ST debt) / total assets
7. **Gross Profitability** (negated, 0.5x weight) — TTM (revenue - COGS) / total assets

All factors are winsorized at 1/99 pct before z-scoring. Signals are lagged 4 months
from the fiscal quarter end date to avoid look-ahead bias.

### Universe
- Top 1000 stocks by lagged market cap each month (Russell 1000 proxy)
- CRSP share codes 10/11 only (common domestic equity)
- Lagged price > $3 (liquidity filter)

### Portfolio Construction
- **Long book**: top 100 stocks by long composite z-score (175% gross weight, equal-weighted)
- **Short book**: top 100 stocks by short composite z-score, equal-weighted; gross weight `w_short` is solved each rebalance month for target beta: `w_short = (1.75 × β_long - 0.30) / β_short`
- **Beta estimation**: trailing 12-month OLS beta vs FF market factor, Vasicek-adjusted (2/3 raw + 1/3 × 1.0), minimum 8 months of data required; implemented in `compute_trailing_betas()` and `compute_market_neutral_weights()`
- **Sector neutrality**: long and short books each stay within ±5pp of Russell 1000 proxy sector weights — **implemented** via `_sector_neutral_select()` in `backtest.py`; sectors mapped from SIC codes using `_SECTOR_BREAKS`
- **Rebalancing**: monthly
- **Return**: R = w_L × R_long − w_S × R_short, where w_L = 1.75 and w_S varies to achieve 0.30 net beta

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
| `LONG_WEIGHT` | 1.75 | Fixed long gross weight; short weight solved for target beta |
| `TARGET_BETA` | 0.30 | Target net portfolio beta (0.0 = market neutral) |

## Git Worktree Workflow

**What it solves:** Without worktrees, `git checkout other-branch` swaps every file in the folder.
Uncommitted changes bleed between branches, stash/pop gets messy, and you can't run two
branches side by side. Worktrees give each branch its own folder.

**How it works:** The main repo folder stays on `main`. Each experimental branch gets a sibling
folder (a "worktree") that shares the same `.git` history but has its own working copy.

### Commands
```bash
# Create a worktree for a new branch (from repo root)
git worktree add ../Backtest-<branch-name> -b <branch-name>
# Example:
git worktree add ../Backtest-momentum-decay -b momentum-decay

# Create a worktree for an existing branch
git worktree add ../Backtest-<branch-name> <branch-name>

# List active worktrees
git worktree list

# Remove a worktree when done (after merging)
git worktree remove ../Backtest-<branch-name>
```

### Workflow for model tweaks
1. **Create worktree:** `git worktree add ../Backtest-my-experiment -b my-experiment`
2. **Work in that folder:** `cd ../Backtest-my-experiment && python run_all.py`
3. **Commit there:** `git add . && git commit -m "describe change" && git push -u origin my-experiment`
4. **Compare results** side by side — main's Output/ and the worktree's Output/ exist simultaneously
5. **Merge if happy:** `cd ../Backtest && git merge my-experiment` (from the main worktree)
6. **Clean up:** `git worktree remove ../Backtest-my-experiment`

### Rules for Claude
- **Always use worktrees** (not `git checkout`) when working on a separate branch
- Never switch branches in the main repo folder — create a worktree instead
- Worktree folders go in the parent directory as `Backtest-<branch-name>`
- Data/ folder is large — worktrees share git objects so this doesn't duplicate disk space,
  but `Cache/` and `Output/` will be regenerated per worktree
- Commit and push the worktree branch before merging into main
