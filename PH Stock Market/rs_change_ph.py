"""
RS Momentum & Acceleration Screener

Uses RS history snapshots to compute:
- RS rating change
- RS acceleration score
- Acceleration ranking

Outputs:
- rs_momentum_screener.csv
"""

import pandas as pd
import numpy as np
import os

# ---------------------------------------------------
# FILE PATHS
# ---------------------------------------------------

RS_HISTORY_FILE = "stock_data_ph/rs_scores_ph_history.parquet"
OUTPUT_FILE = "stock_data_ph/rs_change_screener.csv"

# ---------------------------------------------------
# SETTINGS
# ---------------------------------------------------

LOOKBACK_WINDOWS = [5, 10, 20, 60]

# Acceleration weights (institutional style)
ACCEL_WEIGHTS = {
    5:  0.10,
    10: 0.25,
    20: 0.40,
    60: 0.25
  }

# ---------------------------------------------------
# LOAD RS HISTORY
# ---------------------------------------------------

def load_rs_history():

    df = pd.read_parquet(RS_HISTORY_FILE)

    df["rs_date"] = pd.to_datetime(df["rs_date"])

    df = df.sort_values(["symbol", "rs_date"])

    return df


# ---------------------------------------------------
# GET SNAPSHOT BY DATE OFFSET
# ---------------------------------------------------

def get_snapshot(df, target_date):

    snapshot = df[df["rs_date"] <= target_date]

    snapshot = (
        snapshot
        .sort_values("rs_date")
        .groupby("symbol")
        .tail(1)
        [["symbol", "RS_Rating"]]
        .set_index("symbol")
    )

    return snapshot


# ---------------------------------------------------
# COMPUTE RS CHANGE
# ---------------------------------------------------

def compute_rs_change(df):

    latest_date = df["rs_date"].max()

    latest_snapshot = df[df["rs_date"] == latest_date][["symbol", "RS_Rating"]]
    latest_snapshot = latest_snapshot.set_index("symbol")

    changes = {}

    for lb in LOOKBACK_WINDOWS:

        past_date = latest_date - pd.Timedelta(days=lb)

        past_snapshot = get_snapshot(df, past_date)

        change = latest_snapshot["RS_Rating"] - past_snapshot["RS_Rating"]

        changes[f"RS_Change_{lb}D"] = change

    change_df = pd.DataFrame(changes)

    change_df["RS_Current"] = latest_snapshot["RS_Rating"]

    return change_df.reset_index()


# ---------------------------------------------------
# COMPUTE ACCELERATION SCORE
# ---------------------------------------------------

def compute_acceleration(df):

    accel = (
        ACCEL_WEIGHTS[5] * df["RS_Change_5D"] +
        ACCEL_WEIGHTS[10] * df["RS_Change_10D"] +
        ACCEL_WEIGHTS[20] * df["RS_Change_20D"] +
        ACCEL_WEIGHTS[60] * df["RS_Change_60D"]
    )

    df["RS_Accel"] = accel

    df["RS_Accel_Rank"] = df["RS_Accel"].rank(ascending=False)

    return df


# ---------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------

def main():

    print("Loading RS history...")

    rs_hist = load_rs_history()

    print("Computing RS change...")

    rs_change = compute_rs_change(rs_hist)

    print("Computing RS acceleration...")

    rs_final = compute_acceleration(rs_change)

    rs_final = rs_final.sort_values("RS_Accel", ascending=False)

    rs_final.to_csv(OUTPUT_FILE, index=False)

    print("\nRS Momentum Screener completed.")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()