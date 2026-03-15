"""
PH RRG Engine – Daily + Weekly Version

Architecture
-------------
Daily RRG
    • 52 trading day window
    • 3-period smoothing on RS-Ratio

Weekly RRG
    • 10 week window
    • no smoothing

Bucket Index
    • equal-weight normalized index

Outputs
-------
rrg_buckets_ph_latest.csv
rrg_buckets_ph_history.parquet

Changes from v3
---------------
- bucket_granular is now always filled (no nulls)
  → For broad RRG rows: bucket_granular = bucket_broad
- is_broad column added (True/False) to distinguish the two RRG types
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm


# ============================================================
# FILE PATHS
# ============================================================

PRICE_FILE = "stock_data_ph/wide_prices_ph.csv"
STOCK_DATA_FILE = "stock_metadata_ph.csv"

OUTPUT_LATEST = "stock_data_ph/rrg_buckets_ph_latest.csv"
OUTPUT_HISTORY = "stock_data_ph/rrg_buckets_ph_history.parquet"

BENCHMARK = "PSEI.PS"


# ============================================================
# RRG CONFIG
# ============================================================

RRG_CONFIG = {
    "daily_52": {
        "label": "daily_52",
        "window": 52,
        "frequency": "daily",
        "smooth": True
    },
    "weekly_10": {
        "label": "weekly_10",
        "window": 10,
        "frequency": "weekly",
        "smooth": False
    }
}

MIN_BUCKET_SIZE = 1
HISTORY_MULTIPLIER = 2


# ============================================================
# QUADRANT CLASSIFICATION
# ============================================================

def classify_quadrant(rsr, rsm):

    if rsr >= 100 and rsm >= 100:
        return "Leading"

    elif rsr < 100 and rsm >= 100:
        return "Improving"

    elif rsr < 100 and rsm < 100:
        return "Lagging"

    elif rsr >= 100 and rsm < 100:
        return "Weakening"

    return np.nan


# ============================================================
# BUILD BUCKET INDEX
# ============================================================

def compute_bucket_indices(price_df, stock_df, bucket_col):

    bucket_indices = {}

    for bucket, group in stock_df.groupby(bucket_col):

        symbols = [s for s in group["symbol"] if s in price_df.columns]

        if len(symbols) < MIN_BUCKET_SIZE:
            continue

        bucket_prices = price_df[symbols].ffill()

        normalized = bucket_prices / bucket_prices.iloc[0]

        bucket_index = normalized.mean(axis=1)

        bucket_indices[bucket] = bucket_index

    return pd.DataFrame(bucket_indices)


# ============================================================
# COMPUTE RRG
# ============================================================

def compute_rrg(bucket_indices, benchmark_prices, window, smooth=True):

    benchmark_prices = benchmark_prices.ffill()

    benchmark_index = benchmark_prices / benchmark_prices.iloc[0]

    bucket_indices, benchmark_index = bucket_indices.align(
        benchmark_index, join="inner", axis=0
    )

    # ------------------------------------------------
    # RS LINE
    # ------------------------------------------------

    rs_line = bucket_indices.div(benchmark_index, axis=0)

    # ------------------------------------------------
    # RS RATIO
    # ------------------------------------------------

    rs_mean = rs_line.rolling(window).mean()

    rs_std = rs_line.rolling(window).std(ddof=0).replace(0, np.nan)

    rs_ratio = 100 + ((rs_line - rs_mean) / rs_std) * 10

    # ------------------------------------------------
    # RS MOMENTUM
    # ------------------------------------------------

    if smooth:

        rs_ratio_smooth = rs_ratio.rolling(3).mean()

        rs_ratio_diff = rs_ratio_smooth.diff()

    else:

        rs_ratio_diff = rs_ratio.diff()

    roc_mean = rs_ratio_diff.rolling(window).mean()

    roc_std = rs_ratio_diff.rolling(window).std(ddof=0).replace(0, np.nan)

    rs_momentum = 100 + ((rs_ratio_diff - roc_mean) / roc_std) * 10

    return rs_ratio, rs_momentum


# ============================================================
# BUILD TIMESERIES OUTPUT
# ============================================================

def build_rrg_timeseries(rsr, rsm, bucket_type, rrg_type, stock_df):

    rows = []

    for bucket in rsr.columns:

        if bucket_type == "bucket_broad":

            bucket_broad = bucket
            bucket_granular = bucket      # ← CHANGE 1: use broad value, no nulls
            is_broad = True               # ← CHANGE 2: flag for Power BI filtering

        else:

            bucket_granular = bucket
            is_broad = False

            bucket_broad = stock_df.loc[
                stock_df["bucket_granular"] == bucket,
                "bucket_broad"
            ].iloc[0]

        valid_dates = rsr[bucket].dropna().index.intersection(
            rsm[bucket].dropna().index
        )

        for dt in valid_dates:

            rows.append({
                "date": dt,
                "bucket_broad": bucket_broad,
                "bucket_granular": bucket_granular,
                "is_broad": is_broad,
                "rrg_type": rrg_type,
                "RS_Ratio": rsr.loc[dt, bucket],
                "RS_Momentum": rsm.loc[dt, bucket],
                "Quadrant": classify_quadrant(
                    rsr.loc[dt, bucket],
                    rsm.loc[dt, bucket]
                )
            })

    return pd.DataFrame(rows)


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("=== PH RRG Engine (Daily + Weekly) ===")

    prices = pd.read_csv(
        PRICE_FILE,
        index_col="date",
        parse_dates=True
    ).sort_index()

    stock_df = pd.read_csv(STOCK_DATA_FILE)

    prices_weekly = prices.resample("W-FRI").last()

    all_rrg = []

    with tqdm(total=len(RRG_CONFIG) * 2, desc="RRG Progress") as pbar:

        for bucket_type in ["bucket_broad", "bucket_granular"]:

            for cfg in RRG_CONFIG.values():

                if cfg["frequency"] == "daily":

                    price_source = prices

                else:

                    price_source = prices_weekly

                benchmark_prices = price_source[BENCHMARK]

                stock_prices = price_source.drop(columns=[BENCHMARK])

                bucket_indices = compute_bucket_indices(
                    stock_prices,
                    stock_df,
                    bucket_type
                )

                if bucket_indices.empty:
                    pbar.update(1)
                    continue

                min_required = HISTORY_MULTIPLIER * cfg["window"]

                valid_cols = [
                    col for col in bucket_indices.columns
                    if bucket_indices[col].notna().sum() >= min_required
                ]

                bucket_filtered = bucket_indices[valid_cols]

                if bucket_filtered.empty:
                    pbar.update(1)
                    continue

                rsr, rsm = compute_rrg(
                    bucket_filtered,
                    benchmark_prices,
                    cfg["window"],
                    smooth=cfg["smooth"]
                )

                rrg_ts = build_rrg_timeseries(
                    rsr,
                    rsm,
                    bucket_type=bucket_type,
                    rrg_type=cfg["label"],
                    stock_df=stock_df
                )

                if not rrg_ts.empty:
                    all_rrg.append(rrg_ts)

                pbar.update(1)

    if not all_rrg:
        raise ValueError("No valid RRG data generated.")

    rrg_full = pd.concat(all_rrg, ignore_index=True)

    latest_date = rrg_full["date"].max()

    rrg_latest = rrg_full[rrg_full["date"] == latest_date]

    rrg_latest.to_csv(OUTPUT_LATEST, index=False)

    if os.path.exists(OUTPUT_HISTORY):

        hist = pd.read_parquet(OUTPUT_HISTORY)

        rrg_full = pd.concat([hist, rrg_full], ignore_index=True)

    rrg_full = rrg_full.drop_duplicates(
        subset=[
            "date",
            "bucket_broad",
            "bucket_granular",
            "is_broad",
            "rrg_type"
        ],
        keep="last"
    )

    rrg_full.to_parquet(OUTPUT_HISTORY, index=False)

    print("✅ RRG Completed Successfully")
    print("Latest rows:", len(rrg_latest))
    print("Total history rows:", len(rrg_full))


if __name__ == "__main__":
    main()
