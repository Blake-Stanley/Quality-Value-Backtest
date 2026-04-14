"""
make_plots.py — Generate all charts for the Market Neutral Long-Short Equity backtest.

Reads from:
  Output/backtest_returns.csv   — monthly returns, weights, S&P 500
  Cache/merged.parquet          — stock-level portfolio assignments + signals (optional)

Saves charts to:
  Output/Charts/                — EW + VW combined charts
  Output/Charts/Equal Weighted/ — EW-only charts styled for presentation slides

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
EW_DIR     = CHARTS_DIR / "Equal Weighted"
CACHE_DIR  = ROOT / "Cache"

from backtest import LONG_WEIGHT, N_LONG, N_SHORT, STRATEGY_NAME as STRATEGY, DATA_DIR, FF_FILE, TARGET_BETA

# ── slide theme ───────────────────────────────────────────────────────────────
SLIDE_BG  = "#bedeff"   # light blue — matches slide background
AXES_BG   = "#EEF5FB"   # slightly lighter blue-white for axes interior
NAVY      = "#062044"   # dark navy — all text, ticks, labels
C_LONG    = "#1A5999"   # deep blue — long book
C_SHORT   = "#B22222"   # dark red — short book
C_STRAT   = "#042550"   # navy — strategy line
C_SP500   = "#D4812A"   # warm orange — S&P 500
C_PURPLE  = "#7030A0"   # purple — misc (beta ratio)
C_LEVLONG = "#4472C4"   # cornflower blue — levered long
C_VW_STRAT = "#4A4A4A"  # dark gray — VW strategy in annual bar chart


# ── helpers ──────────────────────────────────────────────────────────────────
def _apply_theme(fig, axes_list):
    """Apply presentation-ready slide theme to a figure and its axes."""
    fig.patch.set_facecolor(SLIDE_BG)
    for ax in axes_list:
        if ax is None:
            continue
        ax.set_facecolor(AXES_BG)
        ax.title.set_color(NAVY)
        ax.title.set_fontweight("bold")
        ax.xaxis.label.set_color(NAVY)
        ax.yaxis.label.set_color(NAVY)
        ax.tick_params(colors=NAVY, which="both")
        for spine in ax.spines.values():
            spine.set_edgecolor(NAVY)
            spine.set_alpha(0.35)
        # restyle existing grid lines to match theme
        for line in ax.get_xgridlines() + ax.get_ygridlines():
            line.set_color(NAVY)
            line.set_alpha(0.15)
        leg = ax.get_legend()
        if leg:
            leg.get_frame().set_facecolor(AXES_BG)
            leg.get_frame().set_edgecolor(NAVY)
            leg.get_frame().set_alpha(0.9)
            for text in leg.get_texts():
                text.set_color(NAVY)


def _pct_fmt(ax, axis="y"):
    fmt = mticker.FuncFormatter(lambda x, _: f"{x:.0%}")
    if axis == "y":
        ax.yaxis.set_major_formatter(fmt)
    else:
        ax.xaxis.set_major_formatter(fmt)


def _save(fig, name, out_dir=None):
    d = out_dir if out_dir is not None else CHARTS_DIR
    d.mkdir(parents=True, exist_ok=True)
    path = d / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path.name}")


def _drawdown_series(ret):
    cum = (1 + ret).cumprod()
    return (cum - cum.cummax()) / cum.cummax()


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS THAT ONLY NEED backtest_returns.csv
# ══════════════════════════════════════════════════════════════════════════════

def plot_cumulative_returns(results, ew_only=False):
    if ew_only:
        fig, ax = plt.subplots(figsize=(10, 6))
        panels = [(ax, "ew", "Equal-Weighted")]
    else:
        fig, arr = plt.subplots(1, 2, figsize=(14, 6))
        panels = [(arr[0], "ew", "Equal-Weighted"), (arr[1], "vw", "Value-Weighted")]

    all_axes = [p[0] for p in panels]
    for ax, pfx, title in panels:
        for col, color, lbl, invert in [
            (f"{pfx}_long",        C_LONG,  "Long Book",   False),
            (f"{pfx}_short",       C_SHORT, "Short Book",  True),
            (f"{pfx}_mkt_neutral", C_STRAT, "Market Neutral Strategy",     False),
        ]:
            ret = results[col].dropna()
            if invert:
                ret = -ret
            cum = (1 + ret).cumprod()
            ax.plot(cum.index, cum.values, linewidth=1.8, label=lbl, color=color)
        sp = (1 + results["sp500"].dropna()).cumprod()
        ax.plot(sp.index, sp.values, color=C_SP500, linewidth=1.6,
                linestyle="--", label="S&P 500", alpha=0.9)
        ax.set_yscale("log")
        ax.set_title(f"{STRATEGY}\n({title})", fontsize=11)
        ax.set_ylabel("Cumulative Return (log scale, $1 invested)")
        ax.axhline(1, color=NAVY, linestyle=":", linewidth=0.6, alpha=0.4)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which="both")

    _apply_theme(fig, all_axes)
    fig.tight_layout(pad=2.0)
    _save(fig, "cumulative_returns.png", EW_DIR if ew_only else None)


def plot_long_vs_short(results, ew_only=False):
    if ew_only:
        fig, axes = plt.subplots(2, 1, figsize=(10, 12))
        panels = [(0, "ew", "Equal-Weighted")]
        all_axes = list(axes)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        panels = [(0, "ew", "Equal-Weighted"), (1, "vw", "Value-Weighted")]
        all_axes = list(axes.flat)

    for col_idx, pfx, title in panels:
        ax_top = axes[0] if ew_only else axes[0][col_idx]
        ax_bot = axes[1] if ew_only else axes[1][col_idx]

        r_long    = results[f"{pfx}_long"].dropna()
        r_neutral = results[f"{pfx}_mkt_neutral"].dropna()
        idx       = r_long.index.intersection(r_neutral.index)

        w_l = results["w_long"].reindex(idx).fillna(LONG_WEIGHT)
        w_s = results["w_short"].reindex(idx).fillna(LONG_WEIGHT)

        r_short_pl         = results[f"{pfx}_short"].reindex(idx)
        short_contribution = w_s * r_short_pl
        levered_long       = w_l * r_long.reindex(idx)

        for ret, color, lbl, ls in [
            (r_long.reindex(idx),    C_LONG,    f"Unlevered Long Only (1×)",              "-"),
            (levered_long,           C_LEVLONG, f"Levered Long Only ({LONG_WEIGHT:.2g}×)", "--"),
            (r_neutral.reindex(idx), C_STRAT,   "Market Neutral Strategy",                "-"),
        ]:
            cum = (1 + ret).cumprod()
            ax_top.plot(cum.index, cum.values, color=color, linewidth=1.6,
                        linestyle=ls, label=lbl)

        sp = (1 + results["sp500"].dropna()).cumprod().reindex(idx)
        ax_top.plot(sp.index, sp.values, color=C_SP500, linewidth=1.4,
                    linestyle="--", label="S&P 500", alpha=0.9)
        ax_top.set_yscale("log")
        ax_top.set_title(f"Levered Long vs Strategy ({title})", fontsize=11)
        ax_top.set_ylabel("Cumulative Return (log scale, $1 of capital)")
        ax_top.axhline(1, color=NAVY, linestyle=":", linewidth=0.5, alpha=0.4)
        ax_top.legend(fontsize=9)
        ax_top.grid(True, alpha=0.3, which="both")

        cum_short = (1 + short_contribution).cumprod()
        ax_bot.plot(cum_short.index, cum_short.values,
                    color=C_SHORT, linewidth=1.6, label="Short Leg P&L")
        ax_bot.axhline(1, color=NAVY, linestyle="--", linewidth=0.8, alpha=0.5)
        ax_bot.fill_between(cum_short.index, cum_short.values, 1,
                            where=cum_short.values >= 1,
                            color=C_SHORT, alpha=0.15, label="Adding value")
        ax_bot.fill_between(cum_short.index, cum_short.values, 1,
                            where=cum_short.values < 1,
                            color=C_SHORT, alpha=0.35, label="Drag on strategy")
        ax_bot.set_title(f"Short Leg Cumulative P&L per $1 of Capital ({title})", fontsize=10)
        ax_bot.set_ylabel("Cumulative Short P&L ($1 of capital)")
        ax_bot.legend(fontsize=9)
        ax_bot.grid(True, alpha=0.3)

    fig.suptitle(
        f"{STRATEGY} — Decomposing the Short Leg's Contribution\n"
        "Gap between Levered Long and Strategy = short leg drag/benefit",
        fontsize=12, color=NAVY, fontweight="bold",
    )
    _apply_theme(fig, all_axes)
    fig.tight_layout(pad=2.0)
    _save(fig, "long_vs_short.png", EW_DIR if ew_only else None)


def plot_rolling_volatility(results, window=12, ew_only=False):
    if ew_only:
        fig, ax = plt.subplots(figsize=(10, 5))
        panels = [(ax, "ew", "Equal-Weighted")]
    else:
        fig, arr = plt.subplots(1, 2, figsize=(14, 5))
        panels = [(arr[0], "ew", "Equal-Weighted"), (arr[1], "vw", "Value-Weighted")]

    all_axes = [p[0] for p in panels]
    series = [
        ("long",        C_LONG,  "Long Book"),
        ("short",       C_SHORT, "Short Book"),
        ("mkt_neutral", C_STRAT, "Market Neutral Strategy"),
    ]
    for ax, pfx, title in panels:
        for suffix, color, lbl in series:
            ret = results[f"{pfx}_{suffix}"].dropna()
            vol = ret.rolling(window, min_periods=window // 2).std() * np.sqrt(12)
            ax.plot(vol.index, vol.values, color=color, linewidth=1.6, label=lbl)
        sp_vol = results["sp500"].dropna().rolling(window, min_periods=window // 2).std() * np.sqrt(12)
        ax.plot(sp_vol.index, sp_vol.values, color=C_SP500, linewidth=1.4,
                linestyle="--", label="S&P 500", alpha=0.9)
        _pct_fmt(ax)
        ax.set_title(f"{STRATEGY}\nRolling {window}m Volatility ({title})", fontsize=11)
        ax.set_ylabel(f"Annualised Volatility (rolling {window}m)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    _apply_theme(fig, all_axes)
    fig.tight_layout(pad=2.0)
    _save(fig, "rolling_volatility.png", EW_DIR if ew_only else None)


def plot_return_distributions(results, ew_only=False):
    if ew_only:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        dist_series = [
            ("ew_mkt_neutral", "Market Neutral (EW)", C_STRAT),
            ("ew_long",        "Long Book (EW)",      C_LONG),
            ("ew_short",       "Short Book (EW)",     C_SHORT),
        ]
    else:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        dist_series = [
            ("ew_mkt_neutral", "Market Neutral (EW)", C_STRAT),
            ("ew_long",        "Long Book (EW)",       C_LONG),
            ("ew_short",       "Short Book (EW)",      C_SHORT),
            ("vw_mkt_neutral", "Market Neutral (VW)",  C_VW_STRAT),
            ("vw_long",        "Long Book (VW)",        "#5B9BD5"),
            ("vw_short",       "Short Book (VW)",       "#E06060"),
        ]

    all_axes = list(axes.flat)
    for ax, (col, lbl, color) in zip(all_axes, dist_series):
        ret = results[col].dropna()
        ax.hist(ret, bins=40, color=color, alpha=0.75, edgecolor="white", linewidth=0.4)
        ax.axvline(ret.mean(), color=NAVY, linewidth=1.4, linestyle="--",
                   label=f"Mean {ret.mean():.2%}")
        ax.axvline(0, color=C_SP500, linewidth=0.9, linestyle=":")
        ax.set_title(lbl, fontsize=10)
        ax.set_xlabel("Monthly Return")
        ax.set_ylabel("Frequency")
        _pct_fmt(ax, axis="x")
        ax.legend(fontsize=8)
        ax.text(0.97, 0.95, f"Skew: {ret.skew():.2f}\nKurt: {ret.kurt():.2f}",
                transform=ax.transAxes, fontsize=8, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", fc=AXES_BG, ec=NAVY, alpha=0.85))
        ax.grid(True, alpha=0.3)

    for ax in all_axes[len(dist_series):]:
        ax.set_visible(False)

    fig.suptitle(f"{STRATEGY} — Monthly Return Distributions",
                 fontsize=13, color=NAVY, fontweight="bold")
    _apply_theme(fig, [ax for ax in all_axes if ax.get_visible()])
    fig.tight_layout(pad=2.0)
    _save(fig, "return_distributions.png", EW_DIR if ew_only else None)


def plot_drawdown(results, ew_only=False):
    if ew_only:
        fig, ax = plt.subplots(figsize=(10, 5))
        panels = [(ax, "ew", "Equal-Weighted")]
    else:
        fig, arr = plt.subplots(1, 2, figsize=(14, 5))
        panels = [(arr[0], "ew", "Equal-Weighted"), (arr[1], "vw", "Value-Weighted")]

    all_axes = [p[0] for p in panels]
    series = [
        ("mkt_neutral", C_STRAT, "Market Neutral Strategy"),
        ("long",        C_LONG,  "Long Book"),
        ("short",       C_SHORT, "Short Book"),
    ]
    for ax, pfx, title in panels:
        for suffix, color, lbl in series:
            dd = _drawdown_series(results[f"{pfx}_{suffix}"].dropna())
            ax.fill_between(dd.index, dd.values, 0, color=color, alpha=0.2, label=lbl)
            ax.plot(dd.index, dd.values, color=color, linewidth=1.2)
        sp_dd = _drawdown_series(results["sp500"].dropna())
        ax.plot(sp_dd.index, sp_dd.values, color=C_SP500, linewidth=1.2,
                linestyle="--", label="S&P 500", alpha=0.9)
        _pct_fmt(ax)
        ax.set_title(f"{STRATEGY}\nDrawdown ({title})", fontsize=11)
        ax.set_ylabel("Drawdown from Peak")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    _apply_theme(fig, all_axes)
    fig.tight_layout(pad=2.0)
    _save(fig, "drawdown.png", EW_DIR if ew_only else None)


def plot_rolling_sharpe(results, window=24, ew_only=False):
    if ew_only:
        fig, ax = plt.subplots(figsize=(10, 5))
        panels = [(ax, "ew", "Equal-Weighted")]
    else:
        fig, arr = plt.subplots(1, 2, figsize=(14, 5))
        panels = [(arr[0], "ew", "Equal-Weighted"), (arr[1], "vw", "Value-Weighted")]

    all_axes = [p[0] for p in panels]
    series = [
        ("mkt_neutral", C_STRAT, "Market Neutral Strategy"),
        ("long",        C_LONG,  "Long Book"),
        ("short",       C_SHORT, "Short Book"),
    ]
    for ax, pfx, title in panels:
        for suffix, color, lbl in series:
            ret = results[f"{pfx}_{suffix}"].dropna()
            roll_sharpe = (
                ret.rolling(window, min_periods=window // 2).mean() /
                ret.rolling(window, min_periods=window // 2).std()
            ) * np.sqrt(12)
            ax.plot(roll_sharpe.index, roll_sharpe.values, color=color, linewidth=1.6, label=lbl)
        sp_ret = results["sp500"].dropna()
        sp_sharpe = (
            sp_ret.rolling(window, min_periods=window // 2).mean() /
            sp_ret.rolling(window, min_periods=window // 2).std()
        ) * np.sqrt(12)
        ax.plot(sp_sharpe.index, sp_sharpe.values, color=C_SP500, linewidth=1.4,
                linestyle="--", label="S&P 500", alpha=0.9)
        ax.axhline(0, color=NAVY, linestyle="--", linewidth=0.7, alpha=0.4)
        ax.set_title(f"{STRATEGY}\nRolling {window}m Sharpe ({title})", fontsize=11)
        ax.set_ylabel(f"Annualised Sharpe (rolling {window}m)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    _apply_theme(fig, all_axes)
    fig.tight_layout(pad=2.0)
    _save(fig, "rolling_sharpe.png", EW_DIR if ew_only else None)


def plot_annual_returns(results, ew_only=False):
    annual = results[["ew_mkt_neutral", "vw_mkt_neutral", "sp500"]].copy()
    annual.index = pd.to_datetime(annual.index)
    annual = annual.groupby(annual.index.year).apply(lambda x: (1 + x).prod() - 1)

    x = np.arange(len(annual))
    fig, ax = plt.subplots(figsize=(14, 6))

    if ew_only:
        width = 0.35
        ax.bar(x - width / 2, annual["ew_mkt_neutral"], width,
               label="Market Neutral (EW)", color=C_STRAT, alpha=0.85)
        ax.bar(x + width / 2, annual["sp500"], width,
               label="S&P 500", color=C_SP500, alpha=0.85)
    else:
        width = 0.28
        ax.bar(x - width, annual["ew_mkt_neutral"], width,
               label="Market Neutral (EW)", color=C_STRAT, alpha=0.85)
        ax.bar(x,         annual["vw_mkt_neutral"], width,
               label="Market Neutral (VW)", color=C_VW_STRAT, alpha=0.75)
        ax.bar(x + width, annual["sp500"], width,
               label="S&P 500", color=C_SP500, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(annual.index.astype(str), rotation=45, ha="right")
    ax.axhline(0, color=NAVY, linewidth=0.8)
    _pct_fmt(ax)
    ax.set_title(f"{STRATEGY} — Annual Returns", fontsize=12)
    ax.set_ylabel("Annual Return")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    _apply_theme(fig, [ax])
    fig.tight_layout(pad=2.0)
    _save(fig, "annual_returns.png", EW_DIR if ew_only else None)


def plot_weight_over_time(results, ew_only=False):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax1, ax2 = axes

    ax1.plot(results.index, results["w_long"],  color=C_LONG,  linewidth=1.6, label="w_long (fixed)")
    ax1.plot(results.index, results["w_short"], color=C_SHORT, linewidth=1.6, label="w_short (solved)")
    ax1.axhline(LONG_WEIGHT, color=C_LONG, linestyle="--", linewidth=0.8, alpha=0.5)
    ax1.set_ylabel("Gross Portfolio Weight")
    ax1.set_title(f"{STRATEGY} — Portfolio Weights Over Time", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(results.index, results["w_short"] / results["w_long"],
             color=C_PURPLE, linewidth=1.6, label="β_long / β_short")
    ax2.axhline(1.0, color=NAVY, linestyle="--", linewidth=0.7, alpha=0.4)
    ax2.set_ylabel("Implied Beta Ratio (w_short / w_long)")
    ax2.set_title("Implied β_long / β_short Ratio (= 1.0 means equal betas)", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    _apply_theme(fig, [ax1, ax2])
    fig.tight_layout(pad=2.0)
    _save(fig, "portfolio_weights.png", EW_DIR if ew_only else None)


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS THAT NEED Cache/merged.parquet
# ══════════════════════════════════════════════════════════════════════════════

def plot_factor_ic(merged, ew_only=False):
    """Monthly IC (Spearman rank corr) of composite_z vs next-month return."""
    df = merged[merged["port"].isin(["long", "short", "mid"])].copy()
    df = df.sort_values(["PERMNO", "month"])
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
        (axes[0], ic_long,  C_LONG,  "Long Signal — Composite Z-Score (Yield + GP + ROIC)"),
        (axes[1], ic_short, C_SHORT, "Short Signal — Composite Z-Score (FCF + Accruals + EV/EBIT + NEF + F-Score + Leverage + GP)"),
    ]:
        ic_clean = ic.dropna()
        ax.bar(ic_clean.index, ic_clean.values, color=color, alpha=0.4, width=25, label="Monthly IC")
        roll = ic_clean.rolling(ROLL, min_periods=ROLL // 2).mean()
        ax.plot(roll.index, roll.values, color=color, linewidth=1.8, label=f"{ROLL}m Rolling Mean IC")
        ax.axhline(0, color=NAVY, linestyle="--", linewidth=0.7, alpha=0.4)
        mean_ic = ic_clean.mean()
        ax.axhline(mean_ic, color=NAVY, linestyle=":", linewidth=1.0,
                   label=f"Full-period Mean IC: {mean_ic:.3f}")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Spearman Rank Correlation")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{STRATEGY} — Factor Information Coefficient (IC) Over Time",
                 fontsize=12, color=NAVY, fontweight="bold")
    _apply_theme(fig, list(axes))
    fig.tight_layout(pad=2.0)
    _save(fig, "factor_ic.png", EW_DIR if ew_only else None)


def plot_factor_decay(merged, ew_only=False):
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
    ax.plot(list(lags), long_ics,  "o-", color=C_LONG,  linewidth=1.8,
            markersize=7, label="Long signal (Yield + GP + ROIC)")
    ax.plot(list(lags), short_ics, "s-", color=C_SHORT, linewidth=1.8,
            markersize=7, label="Short signal (FCF + Accruals + EV/EBIT + NEF + Lev + F-Score + GP)")
    ax.axhline(0, color=NAVY, linestyle="--", linewidth=0.7, alpha=0.4)
    ax.set_xlabel("Forward Lag (months)")
    ax.set_ylabel("Mean Spearman IC")
    ax.set_xticks(list(lags))
    ax.set_title(f"{STRATEGY} — Factor IC Decay (Signal Half-Life)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    _apply_theme(fig, [ax])
    fig.tight_layout(pad=2.0)
    _save(fig, "factor_ic_decay.png", EW_DIR if ew_only else None)


def plot_sector_heatmap(merged, ew_only=False):
    """Long-book vs short-book sector active weights over time."""
    df = merged[merged["port"].isin(["long", "short"])].copy()
    if "sector" not in df.columns:
        print("  Skipping sector heatmap: 'sector' column not in merged.parquet")
        return

    long_wt  = (df[df["port"] == "long"]
                .groupby(["month", "sector"])["PERMNO"].count()
                .unstack("sector").fillna(0))
    short_wt = (df[df["port"] == "short"]
                .groupby(["month", "sector"])["PERMNO"].count()
                .unstack("sector").fillna(0))

    long_wt  = long_wt.div(long_wt.sum(axis=1), axis=0)
    short_wt = short_wt.div(short_wt.sum(axis=1), axis=0)

    all_sectors = sorted(set(long_wt.columns) | set(short_wt.columns))
    long_wt  = long_wt.reindex(columns=all_sectors, fill_value=0)
    short_wt = short_wt.reindex(columns=all_sectors, fill_value=0)
    active   = long_wt - short_wt

    n    = len(all_sectors)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3), sharex=True)
    axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for ax, sec in zip(axes_flat, all_sectors):
        s = active[sec].dropna()
        ax.fill_between(s.index, s.values, 0,
                        where=s.values >= 0, color=C_LONG,  alpha=0.5, label="Long tilt")
        ax.fill_between(s.index, s.values, 0,
                        where=s.values < 0,  color=C_SHORT, alpha=0.5, label="Short tilt")
        ax.plot(s.index, s.values, color=NAVY, linewidth=0.7)
        ax.axhline(0, color=NAVY, linewidth=0.7, linestyle="--", alpha=0.4)
        _pct_fmt(ax)
        ax.set_title(sec, fontsize=9)
        ax.grid(True, alpha=0.3)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=C_LONG,  alpha=0.5),
        plt.Rectangle((0, 0), 1, 1, color=C_SHORT,  alpha=0.5),
    ]
    fig.legend(handles, ["Long overweight", "Short overweight"],
               loc="lower right", fontsize=9, ncol=2)

    fig.suptitle(f"{STRATEGY} — Sector Active Weights (Long − Short) Over Time",
                 fontsize=12, color=NAVY, fontweight="bold")
    _apply_theme(fig, [ax for ax in axes_flat if ax.get_visible()])
    fig.tight_layout(pad=2.0)
    _save(fig, "sector_active_weights.png", EW_DIR if ew_only else None)


def plot_rolling_beta(results, ff, window=12, ew_only=False):
    """Rolling OLS beta of strategy and books vs FF market factor."""
    # Align dates: results uses month-start, FF uses month-end — match by period
    r = results.copy()
    r.index = pd.PeriodIndex(r.index, freq="M")
    f = ff.copy()
    f.index = pd.PeriodIndex(f.index, freq="M")

    df = r.join(f[["mktrf"]], how="left")
    mkt = df["mktrf"]

    def roll_beta(ret):
        """Rolling Cov(ret, mkt) / Var(mkt)."""
        cov = ret.rolling(window, min_periods=window // 2).cov(mkt)
        var = mkt.rolling(window, min_periods=window // 2).var()
        return cov / var

    if ew_only:
        fig, ax = plt.subplots(figsize=(10, 5))
        panels = [(ax, "ew", "Equal-Weighted")]
    else:
        fig, arr = plt.subplots(1, 2, figsize=(14, 5))
        panels = [(arr[0], "ew", "Equal-Weighted"), (arr[1], "vw", "Value-Weighted")]

    all_axes = [p[0] for p in panels]
    for ax, pfx, title in panels:
        b_strat = roll_beta(df[f"{pfx}_mkt_neutral"])
        b_long  = roll_beta(df[f"{pfx}_long"])
        b_short = roll_beta(df[f"{pfx}_short"])
        idx = b_strat.index.to_timestamp()

        ax.plot(idx, b_strat.values, color=C_STRAT,  linewidth=1.8, label="Market Neutral Strategy")
        ax.plot(idx, b_long.values,  color=C_LONG,   linewidth=1.6, linestyle="--", label="Long Book")
        ax.plot(idx, b_short.values, color=C_SHORT,  linewidth=1.6, linestyle="--", label="Short Book")
        ax.axhline(TARGET_BETA, color=C_SP500, linewidth=1.2, linestyle=":",
                   label=f"Target Beta ({TARGET_BETA:.2g})")
        ax.axhline(0, color=NAVY, linewidth=0.7, linestyle="--", alpha=0.4)

        ax.set_title(f"{STRATEGY}\nRolling {window}m Beta vs Market ({title})", fontsize=11)
        ax.set_ylabel(f"Rolling {window}m Beta (OLS vs FF Mkt-RF)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    _apply_theme(fig, all_axes)
    fig.tight_layout(pad=2.0)
    _save(fig, "rolling_beta.png", EW_DIR if ew_only else None)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    EW_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading backtest_returns.csv ...")
    results = pd.read_csv(OUT_DIR / "backtest_returns.csv", index_col=0, parse_dates=True)

    print("Loading FF factors ...")
    ff = pd.read_stata(DATA_DIR / FF_FILE, columns=["dateff", "mktrf"])
    ff["dateff"] = pd.to_datetime(ff["dateff"])
    ff = ff.set_index("dateff").sort_index()

    print("Generating return-series plots (EW + VW) ...")
    plot_cumulative_returns(results)
    plot_long_vs_short(results)
    plot_rolling_volatility(results)
    plot_return_distributions(results)
    plot_drawdown(results)
    plot_rolling_sharpe(results)
    plot_annual_returns(results)
    plot_weight_over_time(results)
    plot_rolling_beta(results, ff)

    print("Generating EW-only plots for presentation slides ...")
    plot_cumulative_returns(results, ew_only=True)
    plot_long_vs_short(results, ew_only=True)
    plot_rolling_volatility(results, ew_only=True)
    plot_return_distributions(results, ew_only=True)
    plot_drawdown(results, ew_only=True)
    plot_rolling_sharpe(results, ew_only=True)
    plot_annual_returns(results, ew_only=True)
    plot_weight_over_time(results, ew_only=True)
    plot_rolling_beta(results, ff, ew_only=True)

    merged_path = CACHE_DIR / "merged.parquet"
    if merged_path.exists():
        print("Loading Cache/merged.parquet ...")
        merged = pd.read_parquet(merged_path)
        merged["month"] = pd.to_datetime(merged["month"])

        print("Generating signal/sector plots ...")
        plot_factor_ic(merged)
        plot_factor_decay(merged)
        plot_sector_heatmap(merged)

        print("Generating EW signal/sector plots for presentation slides ...")
        plot_factor_ic(merged, ew_only=True)
        plot_factor_decay(merged, ew_only=True)
        plot_sector_heatmap(merged, ew_only=True)
    else:
        print("  Cache/merged.parquet not found — skipping factor IC, decay, and sector plots.")
        print("  Run the full backtest first to generate the cache.")

    print(f"\nAll charts saved to: {CHARTS_DIR}")


if __name__ == "__main__":
    main()
