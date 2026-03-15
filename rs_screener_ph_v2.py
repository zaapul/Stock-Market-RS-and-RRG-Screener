import pandas as pd
import numpy as np
import os

# ----------------------------------------------------
# File paths
# ----------------------------------------------------
PRICE_FILE = "stock_data_ph/wide_prices_ph.csv"
OUTPUT_SCORES = "stock_data_ph/rs_scores_ph_latest.csv"
OUTPUT_HISTORY = "stock_data_ph/rs_scores_ph_history.parquet"

# ----------------------------------------------------
# RS Settings
# ----------------------------------------------------
periods = [20, 60, 120, 180, 250]
weights = {20: 0.10, 60: 0.30, 120: 0.25, 180: 0.20, 250: 0.15}
benchmark = "PSEI.PS"

# ----------------------------------------------------
# RS computation helper
# ----------------------------------------------------
def compute_rs_for_date(df, rs_date):
    """
    Compute RS snapshot using prices up to rs_date.
    rs_date MUST exist in df.index.
    """

    stock_px = df.drop(columns=[benchmark]).loc[:rs_date]
    index_px = df[benchmark].loc[:rs_date]

    rs_raw = pd.DataFrame(index=stock_px.columns)

    for d in periods:
        stock_ret = stock_px.pct_change(d).iloc[-1]
        index_ret = index_px.pct_change(d).iloc[-1]
        rs_raw[f"RS_{d}d"] = (1 + stock_ret) / (1 + index_ret)

    # Z-score standardization
    rs_std = (rs_raw - rs_raw.mean()) / rs_raw.std(ddof=0)
    rs_std = rs_std.add_suffix("_z")

    # Weighted RS score
    rs_score = pd.Series(0.0, index=rs_raw.index)
    for d in periods:
        rs_score += weights[d] * rs_std[f"RS_{d}d_z"]
    rs_score.name = "RS_Score"

    # Percentile rating
    rs_rating = rs_score.rank(pct=True) * 100
    rs_rating.name = "RS_Rating"

    final_df = pd.concat([rs_raw, rs_std, rs_score, rs_rating], axis=1)
    final_df.insert(0, "symbol", final_df.index)
    final_df.insert(1, "rs_date", rs_date.strftime("%Y-%m-%d"))

    return final_df.reset_index(drop=True)

# ----------------------------------------------------
# Missing-date detector (price-aligned pattern)
# ----------------------------------------------------
def generate_missing_dates(price_dates, last_saved_date):
    return price_dates[price_dates > last_saved_date]

# ----------------------------------------------------
# MAIN PIPELINE
# ----------------------------------------------------
def main():

    # Load price data
    df = pd.read_csv(PRICE_FILE, index_col="date", parse_dates=True).sort_index()
    price_dates = df.index

    # Load RS history
    if os.path.exists(OUTPUT_HISTORY):
        hist = pd.read_parquet(OUTPUT_HISTORY)
        last_saved_date = pd.to_datetime(hist["rs_date"].max())
    else:
        hist = pd.DataFrame()
        last_saved_date = price_dates.min() - pd.Timedelta(days=1)

    # Identify missing RS dates
    missing_dates = generate_missing_dates(price_dates, last_saved_date)
    print(f"Missing RS dates to compute: {len(missing_dates)}")

    # Guard: nothing to do
    if len(missing_dates) == 0:
        print("Already up to date. Nothing to compute.")
        return

    new_blocks = []

    for rs_date in missing_dates:
        rs_block = compute_rs_for_date(df, rs_date)
        new_blocks.append(rs_block)
        print(f"✓ RS computed for {rs_date.date()}")

    # Save latest snapshot (last available trading day)
    latest_rs = new_blocks[-1]
    latest_rs.to_csv(OUTPUT_SCORES, index=False)

    # Merge & persist history
    updated_hist = pd.concat([hist] + new_blocks, ignore_index=True)
    updated_hist.to_parquet(OUTPUT_HISTORY, index=False)

    print("\n✅ PH RS Screener (Daily, price-aligned) completed.")
    print(f"Latest RS → {OUTPUT_SCORES}")
    print(f"History   → {OUTPUT_HISTORY}")

if __name__ == "__main__":
    main()