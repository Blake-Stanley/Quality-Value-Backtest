"""
make_plots.py — Generate all charts for the Market Neutral Long-Short Equity backtest.

Reads from:
  Output/backtest_returns.csv   — monthly returns, weights, S&P 500
  Cache/merged.parquet          — stock-level portfolio assignments + signals (optional)

Saves all charts to Output/Charts/.

Usage:
    python Code/make_plots.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from scipy.stats import spearmanr

# ── paths ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
OUT_DIR    = ROOT / "Output"
CHARTS_DIR = OUT_DIR / "Charts"
CACHE_DIR  = ROOT / "Cache"

STRATEGY   = "Market Neutral Long-Short Equity ETF"
N_LONG     = 100
N_SHORT    = 100
LONG_WEIGHT = 1.30

# ── helpers ──────────────────────────────────────────────────────────────────
def _pct_fmt(ax, axis="y"):
    fmt = mticker.FuncFormatter(lambda x, _: f"{x:.0%}")
    if axis == "y":
        ax.yaxis.set_major_formatter(fmt)
    else:
        ax.xaxis.set_major_formatter(fmt)

def _save(fig, name):
    path = CHARTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")

def _drawdown_series(ret):
    cum = (1 + ret).cumprod()
    return (cum - cum.cummax()) / cum.cummax()


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS THAT ONLY NEED backtest_returns.csv
# ══════════════════════════════════════════════════════════════════════════════

def plot_cumulative_returns(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, pfx, title in [
        (axes[0], "ew", "Equal-Weighted"),
        (axes[1], "vw", "Value-Weighted"),
    ]:
        for col, color, lbl, invert in [
            (f"{pfx}_long",        "steelblue",  f"Long Book (top {N_LONG})",     False),
            (f"{pfx}_short",       "firebrick",  f"Short Book (top {N_SHORT})",   True),
            (f"{pfx}_mkt_neutral", "black",      "Market Neutral Strategy",       False),
        ]:
            ret = results[col].dropna()
            if invert:
                ret = -ret
            ax.plot((1 + ret).cumprod().index, (1 + ret).cumprod().values,
                    color=color, linewidth=1.2, label=lbl)
        sp = (1 + results["sp500"].dropna()).cumprod()
        ax.plot(sp.index, sp.values, color="darkorange", linewidth=1.2,
                linestyle="--", label="S&P 500", alpha=0.85)
        ax.set_yscale("log")
        ax.set_title(f"{STRATEGY} ({title})", fontsize=10)
        ax.set_ylabel("Cumulative Return (log scale, $1 invested)")
        ax.axhline(1, color="gray", linestyle="--", linewidth=0.5)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    _save(fig, "cumulative_returns.png")


def plot_rolling_volatility(results, window=12):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    series = [
        ("long",        "steelblue", f"Long Book (top {N_LONG})"),
        ("short",       "firebrick", f"Short Book (top {N_SHORT})"),
        ("mkt_neutral", "black",     "Market Neutral Strategy"),
    ]
    for ax, pfx, title in [(axes[0], "ew", "Equal-Weighted"), (axes[1], "vw", "Value-Weighted")]:
        for suffix, color, lbl in series:
            ret = results[f"{pfx}_{suffix}"].dropna()
            vol = ret.rolling(window, min_periods=window // 2).std() * np.sqrt(12)
            ax.plot(vol.index, vol.values, color=color, linewidth=1.2, label=lbl)
        sp_vol = results["sp500"].dropna().rolling(window, min_periods=window // 2).std() * np.sqrt(12)
        ax.plot(sp_vol.index, sp_vol.values, color="darkorange", linewidth=1.2,
                linestyle="--", label="S&P 500", alpha=0.85)
        _pct_fmt(ax)
        ax.set_title(f"{STRATEGY} — Rolling {window}m Volatility ({title})", fontsize=10)
        ax.set_ylabel(f"Annualised Volatility (rolling {window}m)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "rolling_volatility.png")


def plot_return_distributions(results):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    dist_series = [
        ("ew_mkt_neutral", "Market Neutral (EW)",           "black"),
        ("ew_long",        f"Long Book EW (top {N_LONG})",  "steelblue"),
        ("ew_short",       f"Short Book EW (top {N_SHORT})", "firebrick"),
        ("vw_mkt_neutral", "Market Neutral (VW)",           "dimgray"),
        ("vw_long",        f"Long Book VW (top {N_LONG})",  "cornflowerblue"),
        ("vw_short",       f"Short Book VW (top {N_SHORT})", "salmon"),
    ]
    for ax, (col, lbl, color) in zip(axes.flat, dist_series):
        ret = results[col].dropna()
        ax.hist(ret, bins=40, color=color, alpha=0.75, edgecolor="white", linewidth=0.4)
        ax.axvline(ret.mean(), color="black", linewidth=1.2, linestyle="--",
                   label=f"Mean {ret.mean():.2%}")
        ax.axvline(0, color="gray", linewidth=0.8, linestyle=":")
        ax.set_title(lbl, fontsize=10)
        ax.set_xlabel("Monthly Return")
        ax.set_ylabel("Frequency")
        _pct_fmt(ax, axis="x")
        ax.legend(fontsize=8)
        ax.text(0.97, 0.95, f"Skew: {ret.skew():.2f}\nKurt: {ret.kurt():.2f}",
                transform=ax.transAxes, fontsize=8, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"{STRATEGY} — Monthly Return Distributions", fontsize=13)
    fig.tight_layout()
    _save(fig, "return_distributions.png")


def plot_drawdown(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    series = [
        ("mkt_neutral", "black",      "Market Neutral Strategy"),
        ("long",        "steelblue",  f"Long Book (top {N_LONG})"),
        ("short",       "firebrick",  f"Short Book (top {N_SHORT})"),
    ]
    for ax, pfx, title in [(axes[0], "ew", "Equal-Weighted"), (axes[1], "vw", "Value-Weighted")]:
        for suffix, color, lbl in series:
            dd = _drawdown_series(results[f"{pfx}_{suffix}"].dropna())
            ax.fill_between(dd.index, dd.values, 0, color=color, alpha=0.25, label=lbl)
            ax.plot(dd.index, dd.values, color=color, linewidth=0.8)
        sp_dd = _drawdown_series(results["sp500"].dropna())
        ax.plot(sp_dd.index, sp_dd.values, color="darkorange", linewidth=1.0,
                linestyle="--", label="S&P 500", alpha=0.85)
        _pct_fmt(ax)
        ax.set_title(f"{STRATEGY} — Drawdown ({title})", fontsize=10)
        ax.set_ylabel("Drawdown from Peak")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "drawdown.png")


def plot_rolling_sharpe(results, window=24):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    series = [
        ("mkt_neutral", "black",      "Market Neutral Strategy"),
        ("long",        "steelblue",  f"Long Book (top {N_LONG})"),
        ("short",       "firebrick",  f"Short Book (top {N_SHORT})"),
    ]
    for ax, pfx, title in [(axes[0], "ew", "Equal-Weighted"), (axes[1], "vw", "Value-Weighted")]:
        for suffix, color, lbl in series:
            ret = results[f"{pfx}_{suffix}"].dropna()
            roll_sharpe = (
                ret.rolling(window, min_periods=window // 2).mean() /
                ret.rolling(window, min_periods=window // 2).std()
            ) * np.sqrt(12)
            ax.plot(roll_sharpe.index, roll_sharpe.values, color=color, linewidth=1.2, label=lbl)
        sp_ret = results["sp500"].dropna()
        sp_sharpe = (
            sp_ret.rolling(window, min_periods=window // 2).mean() /
            sp_ret.rolling(window, min_periods=window // 2).std()
        ) * np.sqrt(12)
        ax.plot(sp_sharpe.index, sp_sharpe.values, color="darkorange", linewidth=1.2,
                linestyle="--", label="S&P 500", alpha=0.85)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.7)
        ax.set_title(f"{STRATEGY} — Rolling {window}m Sharpe ({title})", fontsize=10)
        ax.set_ylabel(f"Annualised Sharpe (rolling {window}m)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "rolling_sharpe.png")


def plot_annual_returns(results):
    annual = results[["ew_mkt_neutral", "vw_mkt_neutral", "sp500"]].copy()
    annual.index = pd.to_datetime(annual.index)
    annual = annual.groupby(annual.index.year).apply(lambda x: (1 + x).prod() - 1)

    x = np.arange(len(annual))
    width = 0.28

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, annual["ew_mkt_neutral"], width, label="Market Neutral (EW)",
           color="black", alpha=0.85)
    ax.bar(x,         annual["vw_mkt_neutral"], width, label="Market Neutral (VW)",
           color="dimgray", alpha=0.75)
    ax.bar(x + width, annual["sp500"],          width, label="S&P 500",
           color="darkorange", alpha=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels(annual.index.astype(str), rotation=45, ha="right")
    ax.axhline(0, color="black", linewidth=0.8)
    _pct_fmt(ax)
    ax.set_title(f"{STRATEGY} — Annual Returns", fontsize=12)
    ax.set_ylabel("Annual Return")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    _save(fig, "annual_returns.png")


def plot_weight_over_time(results):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1 = axes[0]
    ax1.plot(results.index, results["w_long"],  color="steelblue", linewidth=1.2, label="w_long (fixed)")
    ax1.plot(results.index, results["w_short"], color="firebrick", linewidth=1.2, label="w_short (solved)")
    ax1.axhline(LONG_WEIGHT, color="steelblue", linestyle="--", linewidth=0.7, alpha=0.6)
    ax1.set_ylabel("Gross Portfolio Weight")
    ax1.set_title(f"{STRATEGY} — Portfolio Weights Over Time", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    beta_ratio = results["w_short"] / results["w_long"]
    ax2.plot(results.index, beta_ratio, color="purple", linewidth=1.2, label="β_long / β_short")
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=0.7)
    ax2.set_ylabel("Implied Beta Ratio (w_short / w_long)")
    ax2.set_title("Implied β_long / β_short Ratio (= 1.0 means equal betas)", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, "portfolio_weights.png")


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS THAT NEED Cache/merged.parquet
# ══════════════════════════════════════════════════════════════════════════════

def plot_factor_ic(merged):
    """Monthly IC (Spearman rank corr) of composite_z vs next-month return."""
    df = merged[merged["port"].isin(["long", "short", "mid"])].copy()
    df = df.sort_values(["PERMNO", "month"])

    # next-month return per stock
    df["next_ret"] = df.groupby("PERMNO")["RET"].shift(-1)
    df = df.dropna(subset=["composite_z", "next_ret"])

    ic_long  = df.groupby("month").apply(
        lambda g: spearmanr(g["composite_z"], g["next_ret"])[0]
    ).rename("IC (long signal)")
    ic_short = df.groupby("month").apply(
        lambda g: spearmanr(g["short_composite_z"], g["next_ret"])[0]
        if "short_composite_z" in g.columns and g["short_composite_z"].notna().sum() > 5 else np.nan
    ).rename("IC (short signal, inverted)")

    ROLL = 12
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    for ax, ic, color, title in [
        (axes[0], ic_long,  "steelblue", "Long Signal — Composite Z-Score (Yield + GP + ROIC)"),
        (axes[1], ic_short, "firebrick", "Short Signal — Composite Z-Score (NEF + Leverage + F-Score + GP)"),
    ]:
        ic_clean = ic.dropna()
        ax.bar(ic_clean.index, ic_clean.values, color=color, alpha=0.4, width=25, label="Monthly IC")
        roll = ic_clean.rolling(ROLL, min_periods=ROLL // 2).mean()
        ax.plot(roll.index, roll.values, color=color, linewidth=1.5, label=f"{ROLL}m Rolling Mean IC")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.7)
        mean_ic = ic_clean.mean()
        ax.axhline(mean_ic, color="black", linestyle=":", linewidth=1.0,
                   label=f"Full-period Mean IC: {mean_ic:.3f}")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Spearman Rank Correlation")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{STRATEGY} — Factor Information Coefficient (IC) Over Time", fontsize=12)
    fig.tight_layout()
    _save(fig, "factor_ic.png")


def plot_factor_decay(merged):
    """IC at forward lags 1–6 months to show signal half-life."""
    df = merged[merged["port"].isin(["long", "short", "mid"])].copy()
    df = df.sort_values(["PERMNO", "month"])

    lags = range(1, 7)
    long_ics, short_ics = [], []

    for lag in lags:
        df[f"fwd_{lag}"] = df.groupby("PERMNO")["RET"].shift(-lag)
        tmp = df.dropna(subset=["composite_z", f"fwd_{lag}"])
        long_ics.append(
            tmp.groupby("month").apply(
                lambda g: spearmanr(g["composite_z"], g[f"fwd_{lag}"])[0]
            ).mean()
        )
        if "short_composite_z" in df.columns:
            tmp2 = df.dropna(subset=["short_composite_z", f"fwd_{lag}"])
            short_ics.append(
                tmp2.groupby("month").apply(
                    lambda g: spearmanr(g["short_composite_z"], g[f"fwd_{lag}"])[0]
                    if g["short_composite_z"].notna().sum() > 5 else np.nan
                ).mean()
            )
        else:
            short_ics.append(np.nan)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(list(lags), long_ics,  "o-", color="steelblue", linewidth=1.5,
            markersize=7, label="Long signal (Yield + GP + ROIC)")
    ax.plot(list(lags), short_ics, "s-", color="firebrick", linewidth=1.5,
            markersize=7, label="Short signal (NEF + Lev + F-Score + GP)")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.7)
    ax.set_xlabel("Forward Lag (months)")
    ax.set_ylabel("Mean Spearman IC")
    ax.set_xticks(list(lags))
    ax.set_title(f"{STRATEGY} — Factor IC Decay (Signal Half-Life)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "factor_ic_decay.png")


def plot_sector_heatmap(merged):
    """Long-book vs short-book sector weights over time, as a line chart of over/underweights."""
    df = merged[merged["port"].isin(["long", "short"])].copy()
    if "sector" not in df.columns:
        print("  Skipping sector heatmap: 'sector' column not in merged.parquet")
        return

    # Fraction of each book in each sector, per month
    long_wt  = (df[df["port"] == "long"]
                .groupby(["month", "sector"])["PERMNO"].count()
                .unstack("sector").fillna(0))
    short_wt = (df[df["port"] == "short"]
                .groupby(["month", "sector"])["PERMNO"].count()
                .unstack("sector").fillna(0))

    long_wt  = long_wt.div(long_wt.sum(axis=1), axis=0)
    short_wt = short_wt.div(short_wt.sum(axis=1), axis=0)

    # Active weight = long weight - short weight
    all_sectors = sorted(set(long_wt.columns) | set(short_wt.columns))
    long_wt  = long_wt.reindex(columns=all_sectors, fill_value=0)
    short_wt = short_wt.reindex(columns=all_sectors, fill_value=0)
    active   = long_wt - short_wt

    n = len(all_sectors)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3), sharex=True)
    axes_flat = axes.flat if rows > 1 else [axes] if cols == 1 else list(axes)

    for ax, sec in zip(axes_flat, all_sectors):
        s = active[sec].dropna()
        ax.fill_between(s.index, s.values, 0,
                        where=s.values >= 0, color="steelblue", alpha=0.5, label="Long tilt")
        ax.fill_between(s.index, s.values, 0,
                        where=s.values < 0,  color="firebrick",  alpha=0.5, label="Short tilt")
        ax.plot(s.index, s.values, color="black", linewidth=0.6)
        ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
        _pct_fmt(ax)
        ax.set_title(sec, fontsize=9)
        ax.grid(True, alpha=0.3)

    # hide unused axes
    for ax in list(axes_flat)[n:]:
        ax.set_visible(False)

    # shared legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="steelblue", alpha=0.5),
        plt.Rectangle((0, 0), 1, 1, color="firebrick",  alpha=0.5),
    ]
    fig.legend(handles, ["Long overweight", "Short overweight"],
               loc="lower right", fontsize=9, ncol=2)

    fig.suptitle(f"{STRATEGY} — Sector Active Weights (Long − Short) Over Time", fontsize=12)
    fig.tight_layout()
    _save(fig, "sector_active_weights.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading backtest_returns.csv ...")
    results = pd.read_csv(OUT_DIR / "backtest_returns.csv", index_col=0, parse_dates=True)

    print("Generating return-series plots ...")
    plot_cumulative_returns(results)
    plot_rolling_volatility(results)
    plot_return_distributions(results)
    plot_drawdown(results)
    plot_rolling_sharpe(results)
    plot_annual_returns(results)
    plot_weight_over_time(results)

    merged_path = CACHE_DIR / "merged.parquet"
    if merged_path.exists():
        print("Loading Cache/merged.parquet ...")
        merged = pd.read_parquet(merged_path)
        merged["month"] = pd.to_datetime(merged["month"])

        print("Generating signal/sector plots ...")
        plot_factor_ic(merged)
        plot_factor_decay(merged)
        plot_sector_heatmap(merged)
    else:
        print("  Cache/merged.parquet not found — skipping factor IC, decay, and sector plots.")
        print("  Run the full backtest first to generate the cache.")

    print(f"\nAll charts saved to: {CHARTS_DIR}")


if __name__ == "__main__":
    main()
