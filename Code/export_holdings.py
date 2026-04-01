"""
Generate the latest long/short holdings snapshot for the 130/30 strategy.

The script reuses the existing backtest pipeline to rebuild the signal,
identify the most recent rebalance, and save both books to an Excel file
along with the valuation inputs that drive the ranking.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

import backtest_130_30 as bt

OUTPUT_FILE = bt.OUT_DIR / "130_30_holdings_snapshot.xlsx"


def _latest_component_snapshot(signal_df: pd.DataFrame) -> pd.DataFrame:
    """Return latest component rows keyed by PERMNO and signal month."""
    df = signal_df.copy()
    df["signal_month"] = df["signal_avail"].dt.to_period("M").dt.to_timestamp()
    sort_cols = ["PERMNO", "signal_month", "datadate"]
    df.sort_values(sort_cols, inplace=True)
    keep_cols = [
        "PERMNO",
        "signal_month",
        "datadate",
        "tic",
        "conm",
        "sh_yield",
        "gross_prof",
        "roic",
        "sh_yield_z",
        "gross_prof_z",
        "roic_z",
        "pe_ratio",
    ]
    return df[keep_cols].drop_duplicates(subset=["PERMNO", "signal_month"], keep="last")


def _format_holdings_table(df: pd.DataFrame) -> pd.DataFrame:
    ordered_cols = [
        "Book",
        "Rank",
        "Book Weight",
        "Ticker",
        "Company",
        "Sector",
        "PERMNO",
        "Rebalance Month",
        "Signal Month",
        "Financials Through",
        "Composite Z",
        "Shareholder Yield (TTM)",
        "Shareholder Yield Z",
        "Gross Profitability",
        "Gross Profitability Z",
        "ROIC",
        "ROIC Z",
        "P/E (TTM)",
    ]
    numeric_cols = [
        "Composite Z",
        "Shareholder Yield (TTM)",
        "Shareholder Yield Z",
        "Gross Profitability",
        "Gross Profitability Z",
        "ROIC",
        "ROIC Z",
        "P/E (TTM)",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)
    return df[ordered_cols]


def main():
    _cache_merged  = bt.CACHE_DIR / "merged.parquet"
    _cache_signal  = bt.CACHE_DIR / "signal_components.parquet"
    _refresh       = "--refresh" in sys.argv

    if not _refresh and _cache_merged.exists() and _cache_signal.exists():
        print("Loading from cache (use --refresh to rebuild from source data) ...")
        merged     = pd.read_parquet(_cache_merged,  engine="pyarrow")
        signal_raw = pd.read_parquet(_cache_signal,  engine="pyarrow")
        for col in ["month", "signal_source_month"]:
            if col in merged.columns:
                merged[col] = pd.to_datetime(merged[col])
        for col in ["signal_avail", "datadate"]:
            if col in signal_raw.columns:
                signal_raw[col] = pd.to_datetime(signal_raw[col])
    else:
        if _refresh:
            print("--refresh flag set: rebuilding from source data ...")
        comp, crsp, _ = bt.load_data()
        signal_raw = bt.build_signal(comp, include_components=True)
        signal_monthly = bt.resample_signal(signal_raw)
        crsp_clean = bt.clean_crsp(crsp)
        merged = bt.merge_and_form_portfolios(crsp_clean, signal_monthly)
        bt.CACHE_DIR.mkdir(exist_ok=True)
        merged.to_parquet(_cache_merged,  engine="pyarrow", index=False)
        signal_raw.to_parquet(_cache_signal, engine="pyarrow", index=False)

    if merged.empty:
        raise RuntimeError("No merged rows were produced; cannot build holdings.")

    latest_month = merged["month"].max()
    snapshot = merged[
        (merged["month"] == latest_month) & (merged["port"].isin(["long", "short"]))
    ].copy()
    if snapshot.empty:
        raise RuntimeError("No holdings found for the latest month.")

    snapshot["Book"] = snapshot["port"].str.title()
    snapshot["Book Weight"] = np.where(
        snapshot["Book"] == "Long",
        1.30 / bt.N_LONG,
        0.30 / bt.N_SHORT,
    )

    snapshot["Rank"] = np.nan
    long_mask = snapshot["Book"] == "Long"
    short_mask = snapshot["Book"] == "Short"
    snapshot.loc[long_mask, "Rank"] = snapshot.loc[long_mask, bt.SIGNAL_COL].rank(
        ascending=False, method="first"
    )
    snapshot.loc[short_mask, "Rank"] = snapshot.loc[short_mask, bt.SIGNAL_COL].rank(
        ascending=True, method="first"
    )
    snapshot["Rank"] = snapshot["Rank"].astype(int)

    components = _latest_component_snapshot(signal_raw)
    snapshot = snapshot.merge(
        components,
        left_on=["PERMNO", "signal_source_month"],
        right_on=["PERMNO", "signal_month"],
        how="left",
    )
    snapshot["Sector"] = snapshot["sich"].map(bt._SIC_MAP).fillna("Industrials")
    snapshot["Rebalance Month"] = snapshot["month"].dt.to_period("M").dt.to_timestamp()
    snapshot["Signal Month"] = snapshot["signal_source_month"].dt.to_period("M").dt.to_timestamp()
    snapshot["Financials Through"] = snapshot["datadate"].dt.to_period("M").dt.to_timestamp()
    snapshot.rename(
        columns={
            "tic": "Ticker",
            "conm": "Company",
            bt.SIGNAL_COL: "Composite Z",
            "sh_yield": "Shareholder Yield (TTM)",
            "gross_prof": "Gross Profitability",
            "roic": "ROIC",
            "sh_yield_z": "Shareholder Yield Z",
            "gross_prof_z": "Gross Profitability Z",
            "roic_z": "ROIC Z",
            "pe_ratio": "P/E (TTM)",
        },
        inplace=True,
    )

    cols_to_keep = [
        "Book",
        "Rank",
        "Book Weight",
        "Ticker",
        "Company",
        "Sector",
        "PERMNO",
        "Rebalance Month",
        "Signal Month",
        "Financials Through",
        "Composite Z",
        "Shareholder Yield (TTM)",
        "Shareholder Yield Z",
        "Gross Profitability",
        "Gross Profitability Z",
        "ROIC",
        "ROIC Z",
        "P/E (TTM)",
    ]
    snapshot = snapshot[cols_to_keep].sort_values(["Book", "Rank"]).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        _format_holdings_table(snapshot).to_excel(writer, sheet_name="All Holdings", index=False)
        snapshot[snapshot["Book"] == "Long"].to_excel(writer, sheet_name="Long Book", index=False)
        snapshot[snapshot["Book"] == "Short"].to_excel(writer, sheet_name="Short Book", index=False)

    print(f"Saved holdings snapshot to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
