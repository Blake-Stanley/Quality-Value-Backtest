"""
Backtest: 130/30 Long-Short Equity ETF
========================================
Long signal  : Equal-weight composite z-score of:
                 1. Shareholder Yield  = (TTM dividends + TTM net buybacks) / market cap
                 2. Gross Profitability = TTM (revenue - COGS) / total assets
                 3. ROIC               = TTM NOPAT / avg invested capital
Short signal : Separate equal-weight composite z-score (high = bad company = short candidate):
                 1. Net External Financing = TTM (stock issuance - buybacks + debt issuance - repayments) / assets
                 2. Piotroski F-Score (negated) = 9-criteria fundamental quality score (0-9)
                 3. Leverage = (LT debt + ST debt) / total assets
                 4. Gross Profitability (negated) = TTM (revenue - COGS) / total assets
Universe     : S&P 500 proxy (top 500 by mktcap each month), SHRCD 10/11, lagged price > $3
Long         : top 100 by long composite score  (130% gross weight, equal-weighted)
Short        : top 100 by short composite score (30% gross weight, equal-weighted)
Sector       : +/-5pp sector neutrality vs S&P 500 proxy weights
Rebal        : Quarterly
Return       : R = 1.30 * R_long - 0.30 * R_short
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats as scipy_stats
from collections import Counter
import warnings, time

warnings.filterwarnings("ignore")

# =====================================================================
# CONFIG
# =====================================================================
STRATEGY_NAME   = "130/30 Long-Short Equity ETF"
SIGNAL_COL       = "composite_z"
SHORT_SIGNAL_COL = "short_composite_z"
DATA_DIR        = Path(__file__).resolve().parent.parent / "Data"
OUT_DIR         = Path(__file__).resolve().parent.parent / "Output"
CACHE_DIR       = Path(__file__).resolve().parent.parent / "Cache"
COMP_FILE       = "compustat_with_permno.parquet"
CRSP_FILE       = "crsp_m.dta"
FF_FILE         = "ff5_plus_mom.dta"
LAG_MONTHS      = 4
MIN_PRICE       = 3
N_QUANTILES     = 10
STALENESS_DAYS  = 365
N_UNIVERSE      = 500    # S&P 500 proxy size
N_LONG          = 100    # stocks in long book
N_SHORT         = 100    # stocks in short book
SECTOR_TOL      = 0.05   # +/-5pp sector neutrality
REBALANCE_MONTHS = 3     # rebalance frequency in months


# =====================================================================
# SIC -> GICS-like SECTOR MAP
# =====================================================================
_SECTOR_BREAKS = [
    (100,   999,   "Consumer Staples"),
    (1000,  1499,  "Materials"),
    (1500,  1799,  "Industrials"),
    (2000,  2199,  "Consumer Staples"),
    (2200,  2399,  "Consumer Discretionary"),
    (2400,  2499,  "Industrials"),
    (2500,  2599,  "Consumer Discretionary"),
    (2600,  2699,  "Materials"),
    (2700,  2799,  "Communication Services"),
    (2800,  2829,  "Materials"),
    (2830,  2836,  "Health Care"),
    (2837,  2899,  "Materials"),
    (2900,  2999,  "Energy"),
    (3000,  3199,  "Materials"),
    (3200,  3299,  "Materials"),
    (3300,  3399,  "Materials"),
    (3400,  3499,  "Industrials"),
    (3500,  3569,  "Industrials"),
    (3570,  3579,  "Information Technology"),
    (3580,  3599,  "Industrials"),
    (3600,  3669,  "Consumer Discretionary"),
    (3670,  3679,  "Information Technology"),
    (3680,  3699,  "Consumer Discretionary"),
    (3700,  3799,  "Consumer Discretionary"),
    (3800,  3829,  "Information Technology"),
    (3830,  3851,  "Health Care"),
    (3852,  3999,  "Information Technology"),
    (4000,  4599,  "Industrials"),
    (4600,  4699,  "Energy"),
    (4700,  4799,  "Industrials"),
    (4800,  4899,  "Communication Services"),
    (4900,  4999,  "Utilities"),
    (5000,  5099,  "Industrials"),
    (5100,  5199,  "Consumer Staples"),
    (5200,  5399,  "Consumer Discretionary"),
    (5400,  5499,  "Consumer Staples"),
    (5500,  5799,  "Consumer Discretionary"),
    (5800,  5899,  "Consumer Discretionary"),
    (5900,  5999,  "Consumer Staples"),
    (6000,  6199,  "Financials"),
    (6200,  6299,  "Financials"),
    (6300,  6499,  "Financials"),
    (6500,  6552,  "Real Estate"),
    (6553,  6799,  "Financials"),
    (7000,  7299,  "Consumer Discretionary"),
    (7300,  7369,  "Industrials"),
    (7370,  7379,  "Information Technology"),
    (7380,  7499,  "Industrials"),
    (7500,  7799,  "Consumer Discretionary"),
    (7800,  7999,  "Communication Services"),
    (8000,  8099,  "Health Care"),
    (8100,  8999,  "Industrials"),
]

def _build_sic_map():
    mapping = {}
    for lo, hi, sector in _SECTOR_BREAKS:
        for s in range(lo, hi + 1):
            mapping[s] = sector
    return mapping

_SIC_MAP = _build_sic_map()

def sic_to_sector(sic):
    try:
        return _SIC_MAP.get(int(sic), "Industrials")
    except (ValueError, TypeError):
        return "Industrials"


# =====================================================================
# LOAD DATA
# =====================================================================
def load_data():
    # Read all columns to avoid PyArrow encoding issues with column-level filtering,
    # then keep only what we need.
    _needed = [
        "gvkey", "datadate", "fyearq", "fqtr",
        "indfmt", "consol", "popsrc", "datafmt",
        "fic", "permno", "sich", "tic", "conm",
        "revtq", "saleq", "cogsq", "atq",
        "oiadpq", "nopiq", "txtq", "piq",
        "ceqq", "dlttq", "dlcq", "cheq",
        "dvpsxq", "cshoq",
        "prstkcy", "sstky",
        "ibq", "oancfy", "actq", "lctq",
        "mkvaltq", "prccq",
    ]
    comp = pd.read_parquet(DATA_DIR / COMP_FILE, engine="fastparquet",
                           columns=[c for c in _needed])
    comp = comp[[c for c in _needed if c in comp.columns]]
    crsp = pd.read_stata(
        DATA_DIR / CRSP_FILE,
        columns=["PERMNO", "date", "RET", "PRC", "SHROUT", "SHRCD"],
    )
    ff = pd.read_stata(DATA_DIR / FF_FILE, columns=["dateff", "mktrf", "rf"])
    return comp, crsp, ff


# =====================================================================
# BUILD SIGNAL
# =====================================================================
def build_signal(comp, include_components=False):
    # Standard Compustat filters
    comp = comp[
        (comp["indfmt"] == "INDL") & (comp["consol"] == "C") &
        (comp["popsrc"] == "D")    & (comp["datafmt"] == "STD") &
        (comp["fic"] == "USA")
    ].copy()

    comp["datadate"] = pd.to_datetime(comp["datadate"])
    comp["permno"]   = comp["permno"].astype("Int64")

    for c in ["dlttq", "dlcq", "cheq", "txtq", "piq",
              "dvpsxq", "prstkcy", "sstky", "mkvaltq"]:
        comp[c] = comp[c].fillna(0)

    comp["rev"] = comp["revtq"].fillna(comp["saleq"])
    comp.dropna(subset=["permno", "rev", "cogsq", "atq",
                         "ceqq", "oiadpq", "prccq", "cshoq"], inplace=True)
    comp = comp[(comp["atq"] > 0) & (comp["cshoq"] > 0)].copy()

    comp.sort_values(["gvkey", "datadate"], inplace=True)
    comp.drop_duplicates(subset=["gvkey", "datadate"], keep="last", inplace=True)

    # Market cap
    comp["mktcap"] = np.where(comp["mkvaltq"] > 0,
                               comp["mkvaltq"],
                               comp["prccq"].abs() * comp["cshoq"])
    comp = comp[comp["mktcap"] > 0].copy()

    # ---- Shareholder yield inputs ----
    # Quarterly dividends = DPS * shares
    comp["divs_q"] = comp["dvpsxq"].abs() * comp["cshoq"]

    # Net buybacks (quarterly): prstkcy & sstky are YTD annual CF items.
    # Convert YTD -> quarterly by differencing within each fiscal year.
    for col in ["prstkcy", "sstky"]:
        comp[f"{col}_q"] = comp.groupby(["gvkey", "fyearq"])[col].diff()
        comp[f"{col}_q"] = comp[f"{col}_q"].fillna(comp[col])  # first obs of each group = YTD value
        # Q1: the YTD value is already the quarterly value
        mask_q1 = comp["fqtr"] == 1
        comp.loc[mask_q1, f"{col}_q"] = comp.loc[mask_q1, col]

    comp["buybacks_q"] = comp["prstkcy_q"].clip(lower=0) - comp["sstky_q"].clip(lower=0)

    # ---- TTM sums ----
    for col in ["rev", "cogsq", "oiadpq", "nopiq", "divs_q", "buybacks_q"]:
        comp[col] = comp[col].fillna(0)
        comp[f"{col}_ttm"] = (
            comp.groupby("gvkey")[col]
                .rolling(4, min_periods=4).sum()
                .reset_index(level=0, drop=True)
        )

    comp.dropna(subset=["rev_ttm", "cogsq_ttm", "oiadpq_ttm"], inplace=True)

    # ---- Factor 1: Shareholder Yield ----
    comp["sh_yield"] = (comp["divs_q_ttm"] + comp["buybacks_q_ttm"]) / comp["mktcap"]

    # ---- Factor 2: Gross Profitability ----
    comp["gross_prof"] = (comp["rev_ttm"] - comp["cogsq_ttm"]) / comp["atq"]

    # ---- Factor 3: ROIC ----
    comp["ic"]      = comp["ceqq"] + comp["dlttq"] + comp["dlcq"] - comp["cheq"]
    comp["ic_lag4"] = comp.groupby("gvkey")["ic"].shift(4)
    comp["ic_avg"]  = (comp["ic"] + comp["ic_lag4"]) / 2

    tax_rate = (comp["txtq"] / comp["piq"].replace(0, np.nan)).clip(0, 0.50).fillna(0.21)
    comp["nopat_ttm"] = np.where(
        comp["nopiq_ttm"].notna() & (comp["nopiq_ttm"] != 0),
        comp["nopiq_ttm"],
        comp["oiadpq_ttm"] * (1 - tax_rate),
    )
    comp.dropna(subset=["ic_avg", "nopat_ttm"], inplace=True)
    comp = comp[comp["ic_avg"] > 0].copy()
    comp["roic"] = comp["nopat_ttm"] / comp["ic_avg"]

    # ==============================================================
    # SHORT BOOK SIGNALS (separate composite, high = bad = short)
    # ==============================================================

    # ---- Short Factor 1: Net External Financing ----
    # Equity issuance from CF statement (already quarterly-ized above)
    # Debt change from balance sheet: quarterly change in (dlttq + dlcq)
    comp["total_debt"] = comp["dlttq"] + comp["dlcq"]
    comp["debt_chg_q"] = comp["total_debt"] - comp.groupby("gvkey")["total_debt"].shift(1)
    comp["debt_chg_q"] = comp["debt_chg_q"].fillna(0)

    comp["nef_q"] = comp["sstky_q"] - comp["prstkcy_q"] + comp["debt_chg_q"]
    comp["nef_q"] = comp["nef_q"].fillna(0)
    comp["nef_ttm"] = (
        comp.groupby("gvkey")["nef_q"]
            .rolling(4, min_periods=4).sum()
            .reset_index(level=0, drop=True)
    )
    comp["nef"] = comp["nef_ttm"] / comp["atq"].replace(0, np.nan)

    # ---- Short Factor 2: Leverage ----
    comp["leverage"] = (comp["dlttq"] + comp["dlcq"]) / comp["atq"].replace(0, np.nan)

    # ---- Short Factor 3: Piotroski F-Score ----
    comp["ibq"]    = comp["ibq"].fillna(0)
    comp["ib_ttm"] = (
        comp.groupby("gvkey")["ibq"]
            .rolling(4, min_periods=4).sum()
            .reset_index(level=0, drop=True)
    )
    comp["atq_lag4"] = comp.groupby("gvkey")["atq"].shift(4)
    comp["at_avg4"]  = (comp["atq"] + comp["atq_lag4"]) / 2
    comp["roa_ttm"]  = comp["ib_ttm"] / comp["at_avg4"].replace(0, np.nan)
    comp["roa_lag4"] = comp.groupby("gvkey")["roa_ttm"].shift(4)

    comp["oancfy"]    = comp["oancfy"].fillna(0)
    comp["oancf_q"]   = comp.groupby(["gvkey", "fyearq"])["oancfy"].diff()
    comp["oancf_q"]   = comp["oancf_q"].fillna(comp["oancfy"])
    comp.loc[mask_q1, "oancf_q"] = comp.loc[mask_q1, "oancfy"]
    comp["oancf_ttm"] = (
        comp.groupby("gvkey")["oancf_q"]
            .rolling(4, min_periods=4).sum()
            .reset_index(level=0, drop=True)
    )
    comp["cfo_scaled"]   = comp["oancf_ttm"] / comp["at_avg4"].replace(0, np.nan)
    comp["lev_lag4"]     = comp.groupby("gvkey")["leverage"].shift(4)
    comp["actq"]         = comp["actq"].fillna(0)
    comp["lctq"]         = comp["lctq"].fillna(0)
    comp["curr_ratio"]   = comp["actq"] / comp["lctq"].replace(0, np.nan)
    comp["cr_lag4"]      = comp.groupby("gvkey")["curr_ratio"].shift(4)
    comp["csho_lag4"]    = comp.groupby("gvkey")["cshoq"].shift(4)
    comp["gross_margin"] = ((comp["rev_ttm"] - comp["cogsq_ttm"])
                            / comp["rev_ttm"].replace(0, np.nan))
    comp["gm_lag4"]      = comp.groupby("gvkey")["gross_margin"].shift(4)
    comp["asset_turn"]   = comp["rev_ttm"] / comp["at_avg4"].replace(0, np.nan)
    comp["at_turn_lag4"] = comp.groupby("gvkey")["asset_turn"].shift(4)

    comp["f1"] = (comp["roa_ttm"]      > 0).astype(float)
    comp["f2"] = (comp["oancf_ttm"]    > 0).astype(float)
    comp["f3"] = (comp["roa_ttm"]      > comp["roa_lag4"]).astype(float)
    comp["f4"] = (comp["cfo_scaled"]   > comp["roa_ttm"]).astype(float)
    comp["f5"] = (comp["leverage"]     < comp["lev_lag4"]).astype(float)
    comp["f6"] = (comp["curr_ratio"]   > comp["cr_lag4"]).astype(float)
    comp["f7"] = (comp["cshoq"]        <= comp["csho_lag4"]).astype(float)
    comp["f8"] = (comp["gross_margin"] > comp["gm_lag4"]).astype(float)
    comp["f9"] = (comp["asset_turn"]   > comp["at_turn_lag4"]).astype(float)
    comp["f_score"] = comp[["f1","f2","f3","f4","f5","f6","f7","f8","f9"]].sum(axis=1)

    # ---- Winsorise at 1/99 pct ----
    for col in ["sh_yield", "gross_prof", "roic", "nef", "leverage", "f_score"]:
        lo, hi = comp[col].quantile(0.01), comp[col].quantile(0.99)
        comp[col] = comp[col].clip(lo, hi)

    comp.dropna(subset=["sh_yield", "gross_prof", "roic"], inplace=True)
    # Short factors: allow NaN (stocks without short signal are simply ineligible for short book)
    for col in ["nef", "leverage", "f_score"]:
        comp[col] = comp[col].where(comp[col].notna(), other=np.nan)

    # ---- Cross-sectional z-score each quarter (long factors) ----
    for col in ["sh_yield", "gross_prof", "roic"]:
        grp = comp.groupby("datadate")[col]
        comp[f"{col}_z"] = (comp[col] - grp.transform("mean")) / grp.transform("std")

    comp[SIGNAL_COL] = (comp["sh_yield_z"] + comp["gross_prof_z"] + comp["roic_z"]) / 3.0

    # ---- Cross-sectional z-score each quarter (short factors) ----
    for col in ["nef", "leverage", "f_score"]:
        grp = comp.groupby("datadate")[col]
        comp[f"{col}_z"] = (comp[col] - grp.transform("mean")) / grp.transform("std")

    # Short composite: high score = bad company = short candidate
    # nef_z: high = more dilution/debt issuance = bad
    # f_score_z negated: low F-score = deteriorating fundamentals = bad
    # leverage_z: high = more leveraged = financially fragile = bad
    # gross_prof_z negated: low GP = structurally poor business = bad
    # Use row-wise mean (skipna=True) so one missing factor doesn't nullify the composite
    _short_z = pd.DataFrame({
        "nef_z":          comp["nef_z"],
        "f_score_z_neg":  -comp["f_score_z"],
        "leverage_z":     comp["leverage_z"],
        "gross_prof_z_neg": -comp["gross_prof_z"],
    })
    comp[SHORT_SIGNAL_COL] = _short_z.mean(axis=1)

    comp["eps_ttm"] = comp["nopat_ttm"] / comp["cshoq"].replace(0, np.nan)
    comp["pe_ratio"] = np.where(
        comp["eps_ttm"] > 0,
        comp["prccq"].abs() / comp["eps_ttm"],
        np.nan,
    )

    comp["signal_avail"] = (
        (comp["datadate"] + pd.DateOffset(months=LAG_MONTHS))
        .dt.to_period("M").dt.to_timestamp()
    )

    base_cols = ["permno", "signal_avail", SIGNAL_COL, SHORT_SIGNAL_COL, "datadate", "sich"]
    extra_cols = [
        "tic", "conm",
        "sh_yield", "gross_prof", "roic",
        "sh_yield_z", "gross_prof_z", "roic_z",
        "pe_ratio",
        "nef", "leverage", "f_score",
        "nef_z", "leverage_z", "f_score_z",
    ]
    out_cols = base_cols + extra_cols if include_components else base_cols
    out = comp[out_cols].copy()
    out.rename(columns={"permno": "PERMNO"}, inplace=True)
    out["PERMNO"] = out["PERMNO"].astype(int)
    return out


# =====================================================================
# RESAMPLE SIGNAL TO MONTHLY GRID  (same as HW6)
# =====================================================================
def resample_signal(sig):
    sig_cols = [c for c in [SIGNAL_COL, SHORT_SIGNAL_COL] if c in sig.columns]

    sich_lookup = (sig.sort_values("signal_avail")
                      .drop_duplicates("PERMNO", keep="last")
                      [["PERMNO", "sich"]].set_index("PERMNO")["sich"])

    sig_core = (sig[["PERMNO", "signal_avail"] + sig_cols]
                .sort_values(["PERMNO", "signal_avail"])
                .drop_duplicates(subset=["PERMNO", "signal_avail"], keep="last"))

    # Pivot each signal column to wide, forward-fill on monthly grid, stack back
    monthly_idx = None
    parts = []
    for col in sig_cols:
        wide = sig_core.pivot(index="signal_avail", columns="PERMNO", values=col)
        if monthly_idx is None:
            monthly_idx = pd.date_range(wide.index.min(), wide.index.max(), freq="MS")
        wide = wide.reindex(wide.index.union(monthly_idx)).sort_index().ffill()
        wide = wide.reindex(monthly_idx)
        part = wide.stack().rename(col).reset_index()
        part.columns = ["month", "PERMNO", col]
        parts.append(part.set_index(["month", "PERMNO"]))

    signal = pd.concat(parts, axis=1).reset_index()

    # Staleness filter (based on SIGNAL_COL availability — same fiscal quarter for all signals)
    sig_dates = sig_core[["PERMNO", "signal_avail"]].copy()
    sig_dates = sig_dates.rename(columns={"signal_avail": "month"})
    sig_dates["sig_origin"] = sig_dates["month"]
    signal = signal.merge(sig_dates, on=["PERMNO", "month"], how="left")
    signal["sig_origin"] = signal.groupby("PERMNO")["sig_origin"].ffill()
    signal = signal[
        (signal["month"] - signal["sig_origin"]) <= pd.Timedelta(days=STALENESS_DAYS)
    ].copy()
    signal.rename(columns={"sig_origin": "signal_source_month"}, inplace=True)
    signal.dropna(subset=[SIGNAL_COL], inplace=True)

    signal["sich"] = signal["PERMNO"].map(sich_lookup).fillna(0).astype(int)
    return signal


# =====================================================================
# CLEAN CRSP  (same as HW6)
# =====================================================================
def clean_crsp(crsp):
    crsp.columns = crsp.columns.str.upper()
    crsp.rename(columns={"DATE": "date"}, inplace=True)
    crsp["date"]    = pd.to_datetime(crsp["date"])
    crsp            = crsp[crsp["SHRCD"].isin([10, 11])].copy()
    crsp.dropna(subset=["RET"], inplace=True)
    crsp["abs_prc"] = crsp["PRC"].abs()
    crsp["mktcap"]  = crsp["abs_prc"] * crsp["SHROUT"]
    crsp["PERMNO"]  = crsp["PERMNO"].astype(int)
    crsp["month"]   = crsp["date"].dt.to_period("M").dt.to_timestamp()

    crsp.sort_values(["PERMNO", "month"], inplace=True)
    _lag = crsp.groupby("PERMNO")[["abs_prc", "mktcap"]].shift(1)
    crsp["lag_prc"]    = _lag["abs_prc"]
    crsp["lag_mktcap"] = _lag["mktcap"]
    crsp = crsp[crsp["lag_prc"] > MIN_PRICE].copy()
    return crsp


# =====================================================================
# MERGE & FORM PORTFOLIOS  (modified from HW6 for 130/30)
# =====================================================================
def _sector_neutral_select(df, n_names, ascending, signal_col=SIGNAL_COL):
    """Select n_names PERMNOs meeting +/- SECTOR_TOL sector weights."""
    if n_names <= 0 or df.empty:
        return set()

    sector_mktcap = df.groupby("sector")["lag_mktcap"].sum()
    total = sector_mktcap.sum()
    if total <= 0:
        weights = pd.Series(1.0, index=sector_mktcap.index)
        weights /= weights.sum()
    else:
        weights = sector_mktcap / total

    min_share = np.clip(weights - SECTOR_TOL, 0, None)
    max_share = np.clip(weights + SECTOR_TOL, None, 1.0)
    min_counts = np.floor(min_share * n_names).astype(int)
    max_counts = np.ceil(max_share * n_names).astype(int)
    max_counts = np.maximum(max_counts, min_counts)

    # Make sure we do not require more than available slots
    while min_counts.sum() > n_names:
        sec = min_counts.idxmax()
        if min_counts[sec] == 0:
            break
        min_counts[sec] -= 1

    selected_idx = []
    counts = Counter()
    used = set()
    df_sorted = df.sort_values(signal_col, ascending=ascending)
    ordered_sector_views = {
        sec: df_sorted[df_sorted["sector"] == sec]
        for sec in weights.index
    }

    # Satisfy minimum requirements first
    for sec, req in min_counts.items():
        if req <= 0:
            continue
        avail = ordered_sector_views.get(sec)
        if avail is None or avail.empty:
            continue
        picks = avail.head(req)
        selected_idx.extend(picks.index.tolist())
        used.update(picks.index.tolist())
        counts[sec] += len(picks)

    ordered_idx = df_sorted.index.tolist()
    for idx in ordered_idx:
        if idx in used or len(selected_idx) >= n_names:
            continue
        sec = df.at[idx, "sector"]
        cap = max_counts.get(sec, n_names)
        if counts[sec] >= cap:
            continue
        selected_idx.append(idx)
        used.add(idx)
        counts[sec] += 1

    if len(selected_idx) < n_names:
        for idx in ordered_idx:
            if idx in used:
                continue
            selected_idx.append(idx)
            used.add(idx)
            if len(selected_idx) >= n_names:
                break

    return set(df.loc[selected_idx, "PERMNO"].tolist())


def merge_and_form_portfolios(crsp, signal):
    merged = crsp.merge(signal, on=["PERMNO", "month"], how="inner")
    merged.dropna(subset=["RET", SIGNAL_COL, "lag_mktcap"], inplace=True)

    merged["univ_rank"] = merged.groupby("month")["lag_mktcap"].rank(
        ascending=False, method="first"
    )
    sp500 = merged[merged["univ_rank"] <= N_UNIVERSE].copy()
    sp500["sector"] = sp500["sich"].map(_SIC_MAP).fillna("Industrials")
    sp500["sec_mktcap"] = sp500.groupby(["month", "sector"])["lag_mktcap"].transform("sum")
    sp500["tot_mktcap"] = sp500.groupby("month")["lag_mktcap"].transform("sum")
    sp500["sp500_sec_wt"] = sp500["sec_mktcap"] / sp500["tot_mktcap"]

    monthly_frames = []
    current_long = set()
    current_short = set()

    for idx, (month, month_df) in enumerate(sp500.groupby("month")):
        month_df = month_df.copy()
        rebalance = (idx % REBALANCE_MONTHS == 0) or not current_long or not current_short
        if rebalance:
            long_ids = _sector_neutral_select(month_df, N_LONG, ascending=False)
            short_universe = month_df[~month_df["PERMNO"].isin(long_ids)].copy()
            short_universe = short_universe.dropna(subset=[SHORT_SIGNAL_COL])
            short_ids = _sector_neutral_select(short_universe, N_SHORT, ascending=False,
                                               signal_col=SHORT_SIGNAL_COL)
            current_long = long_ids
            current_short = short_ids

        month_df["port"] = "mid"
        if current_long:
            month_df.loc[month_df["PERMNO"].isin(current_long), "port"] = "long"
        if current_short:
            month_df.loc[month_df["PERMNO"].isin(current_short), "port"] = "short"
        monthly_frames.append(month_df[month_df["port"].isin(["long", "short"])])

    if not monthly_frames:
        return pd.DataFrame(columns=sp500.columns)

    result = pd.concat(monthly_frames, axis=0)
    print(f"  {len(result):,} obs | "
          f"{result['PERMNO'].nunique():,} stocks | "
          f"{result['month'].nunique()} months")
    return result


# =====================================================================
# PORTFOLIO RETURNS  (modified: 1.3*long - 0.3*short)
# =====================================================================
def compute_portfolio_returns(merged):
    ew = merged.groupby(["month", "port"])["RET"].mean().unstack("port")
    merged["wt_ret"] = merged["lag_mktcap"] * merged["RET"]
    vw_agg = merged.groupby(["month", "port"])[["wt_ret", "lag_mktcap"]].sum()
    vw_agg["vw"] = vw_agg["wt_ret"] / vw_agg["lag_mktcap"]
    vw = vw_agg["vw"].unstack("port")

    results = pd.DataFrame(index=ew.index)
    results.index.name = "date"
    for pfx, src in [("ew", ew), ("vw", vw)]:
        results[f"{pfx}_long"]   = src["long"]
        # Short-leg returns are recorded as P&L (i.e., negative of the underlying asset return)
        results[f"{pfx}_short"]  = -src["short"]
        # 130/30 combined strategy return
        results[f"{pfx}_130_30"] = 1.30 * src["long"] - 0.30 * src["short"]

    results.dropna(how="all", inplace=True)
    return results


# =====================================================================
# PERFORMANCE METRICS  (same as HW6)
# =====================================================================
def compute_metrics(r, name, ff_df, is_long_short=False):
    r = r.dropna()
    n = len(r)
    mu      = r.mean()
    sigma   = r.std(ddof=1)
    ann_ret = mu * 12
    ann_vol = sigma * np.sqrt(12)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan
    se_mu   = sigma / np.sqrt(n)
    t_mu    = mu / se_mu if se_mu > 0 else np.nan
    p_mu    = 2 * (1 - scipy_stats.t.cdf(abs(t_mu), df=n-1))
    se_ann  = se_mu * 12
    ci_lo   = ann_ret - 1.96 * se_ann
    ci_hi   = ann_ret + 1.96 * se_ann
    geo     = (1 + r).prod() ** (12 / n) - 1
    cum     = (1 + r).cumprod()
    max_dd  = ((cum - cum.cummax()) / cum.cummax()).min()
    skew    = float(scipy_stats.skew(r))
    kurt    = float(scipy_stats.kurtosis(r))
    calmar  = ann_ret / abs(max_dd) if max_dd != 0 else np.nan
    down_r  = np.minimum(r.values, 0)
    sortino = ann_ret / (np.sqrt((down_r**2).mean()) * np.sqrt(12)) if (down_r**2).mean() > 0 else np.nan
    pct_pos = float((r > 0).mean() * 100)
    worst   = float(r.min())

    df = pd.DataFrame({"ret": r}).reset_index()
    df.columns = ["month", "ret"]
    df = df.merge(ff_df, on="month", how="inner")
    df["y"] = df["ret"] if is_long_short else df["ret"] - df["rf"]
    X = sm.add_constant(df["mktrf"])
    capm = sm.OLS(df["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    alpha_m    = capm.params["const"]
    beta       = capm.params["mktrf"]
    alpha_a    = alpha_m * 12
    alpha_t    = capm.tvalues["const"]
    alpha_p    = capm.pvalues["const"]
    alpha_se_a = capm.bse["const"] * 12
    alpha_ci_lo = alpha_a - 1.96 * alpha_se_a
    alpha_ci_hi = alpha_a + 1.96 * alpha_se_a

    return {
        "name": name, "start": str(r.index.min().date()), "end": str(r.index.max().date()),
        "months": n, "ann_ret": ann_ret, "geo_ret": geo, "ann_vol": ann_vol,
        "sharpe": sharpe, "t_stat_ret": t_mu, "p_val_ret": p_mu,
        "ret_ci_lo": ci_lo, "ret_ci_hi": ci_hi, "max_dd": max_dd,
        "capm_alpha": alpha_a, "alpha_se": alpha_se_a, "alpha_t": alpha_t,
        "alpha_p": alpha_p, "alpha_ci_lo": alpha_ci_lo, "alpha_ci_hi": alpha_ci_hi,
        "capm_beta": beta, "beta_t": capm.tvalues["mktrf"],
        "skewness": skew, "kurtosis": kurt, "calmar": calmar, "sortino": sortino,
        "pct_positive": pct_pos, "worst_month": worst,
    }


# =====================================================================
# OUTPUT  (same as HW6, updated labels)
# =====================================================================
def output_results(results, metrics):
    def fmt_table(m):
        out = pd.DataFrame(index=m.index)
        out["Period"]         = m["start"] + " to " + m["end"]
        out["Months"]         = m["months"].astype(int)
        out["Arith Mean"]     = m["ann_ret"].map(lambda x: f"{x:.2%}")
        out["Geo Mean"]       = m["geo_ret"].map(lambda x: f"{x:.2%}")
        out["Volatility"]     = m["ann_vol"].map(lambda x: f"{x:.2%}")
        out["Sharpe"]         = m["sharpe"].map(lambda x: f"{x:.2f}")
        out["t(mean)"]        = m["t_stat_ret"].map(lambda x: f"{x:.2f}")
        out["p(mean)"]        = m["p_val_ret"].map(lambda x: f"{x:.4f}")
        out["95% CI (ret)"]   = [f"[{lo:.2%}, {hi:.2%}]"
                                 for lo, hi in zip(m["ret_ci_lo"], m["ret_ci_hi"])]
        out["Max DD"]         = m["max_dd"].map(lambda x: f"{x:.2%}")
        out["CAPM alpha"]     = m["capm_alpha"].map(lambda x: f"{x:.2%}")
        out["alpha SE"]       = m["alpha_se"].map(lambda x: f"{x:.2%}")
        out["t(alpha)"]       = m["alpha_t"].map(lambda x: f"{x:.2f}")
        out["p(alpha)"]       = m["alpha_p"].map(lambda x: f"{x:.4f}")
        out["95% CI (alpha)"] = [f"[{lo:.2%}, {hi:.2%}]"
                                 for lo, hi in zip(m["alpha_ci_lo"], m["alpha_ci_hi"])]
        out["CAPM beta"]      = m["capm_beta"].map(lambda x: f"{x:.3f}")
        out["t(beta)"]        = m["beta_t"].map(lambda x: f"{x:.2f}")
        out["Sortino"]        = m["sortino"].map(lambda x: f"{x:.2f}")
        out["Calmar"]         = m["calmar"].map(lambda x: f"{x:.2f}")
        out["Worst Month"]    = m["worst_month"].map(lambda x: f"{x:.2%}")
        out["% Pos Months"]   = m["pct_positive"].map(lambda x: f"{x:.1f}%")
        out["Skewness"]       = m["skewness"].map(lambda x: f"{x:.2f}")
        out["Ex. Kurtosis"]   = m["kurtosis"].map(lambda x: f"{x:.2f}")
        return out

    display = fmt_table(metrics)
    sep = "=" * 100
    header = (f"\n{sep}\n{STRATEGY_NAME.upper()} BACKTEST\n"
              f"Signal: {SIGNAL_COL}   |   Lag: datadate + {LAG_MONTHS} months\n"
              f"Universe: top-{N_UNIVERSE} by mktcap (S&P 500 proxy), SHRCD 10/11, "
              f"lagged |PRC| > ${MIN_PRICE}   |   Rebalancing: quarterly\n"
              f"130% long top-{N_LONG} (Yield+GP+ROIC) / 30% short top-{N_SHORT} (NEF+Leverage+F-Score+GP), "
              f"equal-weight, +/-{SECTOR_TOL:.0%} sector neutrality\n{sep}")

    print(header)
    for label in ["130/30 (EW)", "130/30 (VW)",
                  "EW Long", "EW Short", "VW Long", "VW Short", "S&P 500"]:
        if label in display.index:
            print(f"\n--- {label} ---")
            print(display.loc[label].to_string())
    print(f"\n{sep}")

    csv_path     = OUT_DIR / "130_30_backtest_returns.csv"
    metrics_path = OUT_DIR / "130_30_backtest_metrics.csv"
    txt_path     = OUT_DIR / "130_30_backtest_metrics.txt"

    results.to_csv(csv_path)
    metrics.to_csv(metrics_path)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(header + "\n\n")
        f.write(display.T.to_string())
        f.write(f"\n\n{sep}\n")

    print("\nPlotting cumulative returns ...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, pfx, title in [
        (axes[0], "ew", "Equal-Weighted"),
        (axes[1], "vw", "Value-Weighted"),
    ]:
        for col, color, lbl in [
            (f"{pfx}_long",    "steelblue", f"Long Book (top {N_LONG})"),
            (f"{pfx}_short",   "firebrick", f"Short Book (bottom {N_SHORT})"),
            (f"{pfx}_130_30",  "black",     "130/30 Strategy"),
        ]:
            cum = (1 + results[col].dropna()).cumprod()
            ax.plot(cum.index, cum.values, color=color, linewidth=1.2, label=lbl)

        sp500_cum = (1 + results["sp500"].dropna()).cumprod()
        ax.plot(sp500_cum.index, sp500_cum.values, color="darkorange",
                linewidth=1.2, linestyle="--", label="S&P 500", alpha=0.85)
        ax.set_yscale("log")
        ax.set_title(f"{STRATEGY_NAME} ({title})", fontsize=11)
        ax.set_ylabel("Cumulative Return (log scale, $1 invested)")
        ax.axhline(1, color="gray", linestyle="--", linewidth=0.5)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    plot_path = OUT_DIR / "130_30_backtest.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {plot_path}")
    print(f"\nReturns CSV : {csv_path}\nMetrics CSV : {metrics_path}\nMetrics TXT : {txt_path}")


# =====================================================================
# MAIN  (same as HW6)
# =====================================================================
def _tick(label, t_prev, t0):
    elapsed = time.time() - t_prev
    total   = time.time() - t0
    print(f"  [{elapsed:5.1f}s | {total:6.1f}s total] {label}")
    return time.time()

def main():
    t0 = t = time.time()
    print("Loading data ...")
    comp, crsp, ff = load_data()
    t = _tick("data loaded", t, t0)

    print("Building signal ...")
    sig = build_signal(comp, include_components=True)
    t = _tick(f"{len(sig):,} firm-quarter signals", t, t0)
    CACHE_DIR.mkdir(exist_ok=True)
    sig.to_parquet(CACHE_DIR / "signal_components.parquet", engine="pyarrow", index=False)

    print("Resampling signals to monthly grid ...")
    signal = resample_signal(sig)
    t = _tick(f"{len(signal):,} permno-month rows", t, t0)

    print("Cleaning CRSP ...")
    crsp = clean_crsp(crsp)
    t = _tick("CRSP cleaned", t, t0)

    print("Merging & forming portfolios ...")
    merged = merge_and_form_portfolios(crsp, signal)
    t = _tick("portfolios formed", t, t0)
    merged.to_parquet(CACHE_DIR / "merged.parquet", engine="pyarrow", index=False)

    print("Computing portfolio returns ...")
    results = compute_portfolio_returns(merged)
    t = _tick("returns computed", t, t0)

    ff["month"] = ff["dateff"].dt.to_period("M").dt.to_timestamp()
    ff = ff[["month", "mktrf", "rf"]].drop_duplicates("month")

    print("Running CAPM regressions ...")
    series_list = [
        ("ew_130_30", "130/30 (EW)", False),
        ("ew_long",   "EW Long",     False),
        ("ew_short",  "EW Short",    True),
        ("vw_130_30", "130/30 (VW)", False),
        ("vw_long",   "VW Long",     False),
        ("vw_short",  "VW Short",    True),
    ]
    metrics = pd.DataFrame(
        [compute_metrics(results[c], n, ff, is_long_short=ls)
         for c, n, ls in series_list]
    ).set_index("name")

    # S&P 500 benchmark
    sp500 = (ff.set_index("month")["mktrf"] + ff.set_index("month")["rf"])
    sp500 = sp500.reindex(results.index).rename("sp500")
    results["sp500"] = sp500
    sp500_m = compute_metrics(sp500.dropna(), "S&P 500", ff, is_long_short=False)
    metrics = pd.concat([metrics, pd.DataFrame([sp500_m]).set_index("name")])
    t = _tick("CAPM regressions done", t, t0)

    output_results(results, metrics)
    _tick("outputs written", t, t0)
    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
