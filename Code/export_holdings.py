"""
Generate the latest long/short holdings snapshot for the 130/30 strategy.

Exports a styled Excel workbook (three sheets: All Holdings, Long Book, Short Book)
using the same visual style as make_table.py.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

import backtest_130_30 as bt

OUTPUT_FILE = bt.OUT_DIR / "130_30_holdings_snapshot.xlsx"

# ── Color palette (matches make_table.py) ─────────────────────────────────────
DARK_BLUE   = "1F3864"
MED_BLUE    = "2E5FA3"
LIGHT_BLUE  = "D6E4F7"
ALT_ROW     = "EEF4FB"
WHITE       = "FFFFFF"
DARK_GREEN  = "375623"
MED_GREEN   = "548235"
LIGHT_GREEN = "E2EFDA"
ALT_GREEN   = "F0F7EE"
DARK_RED    = "C00000"
LIGHT_RED   = "FCE4D6"
ALT_RED     = "FEF2EE"

# ── Column groups: (label, header_color, [column_names]) ──────────────────────
_ALL_GROUPS = [
    ("POSITION",        DARK_BLUE,  ["Book", "Rank", "Book Weight"]),
    ("IDENTITY",        MED_BLUE,   ["Ticker", "Company", "Sector", "PERMNO"]),
    ("DATES",           "4472C4",   ["Rebalance Month", "Signal Month", "Financials Through"]),
    ("COMPOSITE SCORE", DARK_GREEN, ["Composite Z"]),
    ("FACTOR Z-SCORES", MED_GREEN,  ["Shareholder Yield Z", "Gross Profitability Z", "ROIC Z"]),
    ("RAW FACTORS",     "70AD47",   ["Shareholder Yield (TTM)", "Gross Profitability", "ROIC", "P/E (TTM)"]),
]

_SINGLE_GROUPS = [
    ("POSITION",        DARK_BLUE,  ["Rank", "Book Weight"]),
    ("IDENTITY",        MED_BLUE,   ["Ticker", "Company", "Sector", "PERMNO"]),
    ("DATES",           "4472C4",   ["Rebalance Month", "Signal Month", "Financials Through"]),
    ("COMPOSITE SCORE", DARK_GREEN, ["Composite Z"]),
    ("FACTOR Z-SCORES", MED_GREEN,  ["Shareholder Yield Z", "Gross Profitability Z", "ROIC Z"]),
    ("RAW FACTORS",     "70AD47",   ["Shareholder Yield (TTM)", "Gross Profitability", "ROIC", "P/E (TTM)"]),
]

_COL_WIDTHS = {
    "Book": 8, "Rank": 6, "Book Weight": 11,
    "Ticker": 8, "Company": 28, "Sector": 16, "PERMNO": 8,
    "Rebalance Month": 14, "Signal Month": 13, "Financials Through": 15,
    "Composite Z": 11,
    "Shareholder Yield Z": 15, "Gross Profitability Z": 17, "ROIC Z": 9,
    "Shareholder Yield (TTM)": 17, "Gross Profitability": 17, "ROIC": 9,
    "P/E (TTM)": 10,
}

def _thick(): return Side(style="medium")
def _hair():  return Side(style="hair")
def _thin():  return Side(style="thin")


def _fmt_val(col: str, val):
    """Format a cell value for display based on its column."""
    try:
        is_nan = pd.isna(val)
    except (TypeError, ValueError):
        is_nan = False
    if is_nan:
        return "—"
    if col == "Book Weight":
        return f"{float(val):.2%}"
    if col == "Rank":
        return int(val)
    if col in ("Composite Z", "Shareholder Yield Z", "Gross Profitability Z", "ROIC Z"):
        return round(float(val), 2)
    if col in ("Shareholder Yield (TTM)", "Gross Profitability", "ROIC"):
        return f"{float(val):.2%}"
    if col == "P/E (TTM)":
        fv = float(val)
        if np.isinf(fv) or np.isnan(fv):
            return "—"
        return round(fv, 1)
    if hasattr(val, "strftime"):
        return val.strftime("%Y-%m")
    return val


def _write_styled_sheet(ws, df: pd.DataFrame, title: str, subtitle: str,
                        column_groups: list, book_type: str = None):
    """Write a fully styled holdings table into an openpyxl worksheet."""
    # Build column order from groups, keeping only columns present in df
    all_cols = [c for _, _, cols in column_groups for c in cols if c in df.columns]
    df = df[all_cols].reset_index(drop=True)
    n_cols = len(all_cols)

    # ── Row 1: Title ──────────────────────────────────────────────────────────
    ws.merge_cells(f"A1:{get_column_letter(n_cols)}1")
    c = ws["A1"]
    c.value = title
    c.font = Font(name="Calibri", bold=True, size=14, color="FFFFFF")
    c.fill = PatternFill("solid", fgColor=DARK_BLUE)
    c.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[1].height = 28

    # ── Row 2: Subtitle ───────────────────────────────────────────────────────
    ws.merge_cells(f"A2:{get_column_letter(n_cols)}2")
    c = ws["A2"]
    c.value = subtitle
    c.font = Font(name="Calibri", italic=True, size=9, color="FFFFFF")
    c.fill = PatternFill("solid", fgColor=MED_BLUE)
    c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.row_dimensions[2].height = 22

    # ── Row 3: Spacer ─────────────────────────────────────────────────────────
    ws.row_dimensions[3].height = 5

    # ── Row 4: Column group headers ───────────────────────────────────────────
    ws.row_dimensions[4].height = 16
    col_idx = 1
    for grp_label, grp_color, grp_cols in column_groups:
        present = [c for c in grp_cols if c in df.columns]
        if not present:
            continue
        n = len(present)
        start_ltr = get_column_letter(col_idx)
        end_ltr   = get_column_letter(col_idx + n - 1)
        if n > 1:
            ws.merge_cells(f"{start_ltr}4:{end_ltr}4")
        cell = ws.cell(4, col_idx, grp_label)
        cell.font      = Font(name="Calibri", bold=True, size=9, color="FFFFFF")
        cell.fill      = PatternFill("solid", fgColor=grp_color)
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border    = Border(left=_thick(), right=_thick(), top=_thick(), bottom=_thin())
        col_idx += n

    # ── Row 5: Column headers ─────────────────────────────────────────────────
    ws.row_dimensions[5].height = 32
    col_idx = 1
    for _, grp_color, grp_cols in column_groups:
        present = [c for c in grp_cols if c in df.columns]
        for col_name in present:
            cell = ws.cell(5, col_idx, col_name)
            cell.font      = Font(name="Calibri", bold=True, size=9, color="FFFFFF")
            cell.fill      = PatternFill("solid", fgColor=grp_color)
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border    = Border(left=_thin(), right=_thin(), top=_thin(), bottom=_thick())
            col_idx += 1

    # ── Data rows ─────────────────────────────────────────────────────────────
    n_data = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        excel_row = i + 6
        book = row.get("Book", book_type or "")
        if book == "Long":
            bg = LIGHT_GREEN if i % 2 == 0 else ALT_GREEN
        elif book == "Short":
            bg = LIGHT_RED if i % 2 == 0 else ALT_RED
        else:
            bg = WHITE if i % 2 == 0 else ALT_ROW

        ws.row_dimensions[excel_row].height = 14
        is_last_row = (i == n_data - 1)
        bot = _thick() if is_last_row else _hair()

        for j, col_name in enumerate(all_cols, start=1):
            val = _fmt_val(col_name, row[col_name])
            cell = ws.cell(excel_row, j, val)
            cell.font  = Font(name="Calibri", size=9)
            cell.fill  = PatternFill("solid", fgColor=bg)
            cell.alignment = Alignment(
                horizontal="left" if col_name in ("Company", "Sector") else "center",
                vertical="center",
            )
            cell.border = Border(
                left=_thick() if j == 1 else _hair(),
                right=_thick() if j == n_cols else _hair(),
                top=_hair(),
                bottom=bot,
            )

    # ── Column widths & freeze ────────────────────────────────────────────────
    col_idx = 1
    for _, _, grp_cols in column_groups:
        for col_name in grp_cols:
            if col_name in df.columns:
                ws.column_dimensions[get_column_letter(col_idx)].width = _COL_WIDTHS.get(col_name, 12)
                col_idx += 1

    ws.freeze_panes = "A6"


def _latest_component_snapshot(signal_df: pd.DataFrame) -> pd.DataFrame:
    df = signal_df.copy()
    df["signal_month"] = df["signal_avail"].dt.to_period("M").dt.to_timestamp()
    df.sort_values(["PERMNO", "signal_month", "datadate"], inplace=True)
    keep_cols = [
        "PERMNO", "signal_month", "datadate", "tic", "conm",
        "sh_yield", "gross_prof", "roic",
        "sh_yield_z", "gross_prof_z", "roic_z", "pe_ratio",
    ]
    return df[keep_cols].drop_duplicates(subset=["PERMNO", "signal_month"], keep="last")


def main():
    _cache_merged = bt.CACHE_DIR / "merged.parquet"
    _cache_signal = bt.CACHE_DIR / "signal_components.parquet"
    _refresh      = "--refresh" in sys.argv

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
        signal_raw    = bt.build_signal(comp, include_components=True)
        signal_monthly = bt.resample_signal(signal_raw)
        crsp_clean    = bt.clean_crsp(crsp)
        merged        = bt.merge_and_form_portfolios(crsp_clean, signal_monthly)
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
    long_mask  = snapshot["Book"] == "Long"
    short_mask = snapshot["Book"] == "Short"
    snapshot.loc[long_mask,  "Rank"] = snapshot.loc[long_mask,  bt.SIGNAL_COL].rank(ascending=False, method="first")
    snapshot.loc[short_mask, "Rank"] = snapshot.loc[short_mask, bt.SHORT_SIGNAL_COL].rank(ascending=False, method="first")
    snapshot["Rank"] = snapshot["Rank"].astype(int)

    components = _latest_component_snapshot(signal_raw)
    snapshot = snapshot.merge(
        components,
        left_on=["PERMNO", "signal_source_month"],
        right_on=["PERMNO", "signal_month"],
        how="left",
    )
    snapshot["Sector"]            = snapshot["sich"].map(bt._SIC_MAP).fillna("Industrials")
    snapshot["Rebalance Month"]   = snapshot["month"].dt.to_period("M").dt.to_timestamp()
    snapshot["Signal Month"]      = snapshot["signal_source_month"].dt.to_period("M").dt.to_timestamp()
    snapshot["Financials Through"] = snapshot["datadate"].dt.to_period("M").dt.to_timestamp()
    snapshot.rename(columns={
        "tic":           "Ticker",
        "conm":          "Company",
        bt.SIGNAL_COL:   "Composite Z",
        "sh_yield":      "Shareholder Yield (TTM)",
        "gross_prof":    "Gross Profitability",
        "roic":          "ROIC",
        "sh_yield_z":    "Shareholder Yield Z",
        "gross_prof_z":  "Gross Profitability Z",
        "roic_z":        "ROIC Z",
        "pe_ratio":      "P/E (TTM)",
    }, inplace=True)

    snapshot = snapshot.sort_values(["Book", "Rank"]).reset_index(drop=True)

    rebal_str = latest_month.strftime("%B %Y")
    title_all   = f"130/30 Long-Short Equity — Holdings Snapshot  ({rebal_str})"
    title_long  = f"130/30 Long-Short Equity — Long Book  ({rebal_str})"
    title_short = f"130/30 Long-Short Equity — Short Book  ({rebal_str})"
    subtitle = (
        "Long signal: Shareholder Yield + Gross Profitability + ROIC  |  "
        "Short signal: Net Ext. Financing + Leverage + F-Score + Gross Profitability  |  "
        "130% long top-100 / 30% short top-100 (S&P 500 proxy)  |  "
        "Quarterly rebalancing, ±5 pp sector neutrality"
    )

    long_df  = snapshot[snapshot["Book"] == "Long"].copy()
    short_df = snapshot[snapshot["Book"] == "Short"].copy()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    wb = openpyxl.Workbook()

    ws_all = wb.active
    ws_all.title = "All Holdings"
    _write_styled_sheet(ws_all, snapshot, title_all, subtitle, _ALL_GROUPS)

    ws_long = wb.create_sheet("Long Book")
    _write_styled_sheet(ws_long, long_df, title_long, subtitle, _SINGLE_GROUPS, book_type="Long")

    ws_short = wb.create_sheet("Short Book")
    _write_styled_sheet(ws_short, short_df, title_short, subtitle, _SINGLE_GROUPS, book_type="Short")

    wb.save(OUTPUT_FILE)
    print(f"Saved holdings snapshot to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
