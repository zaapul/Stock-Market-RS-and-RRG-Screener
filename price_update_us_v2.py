#!/usr/bin/env python3
"""
price_update_us_v2.2.py

- Per-ticker multithreaded downloading using yfinance
- Dual tqdm progress bars (batch-level + ticker-level)
- Incremental updater with missing pair detection
- Adds ^GSPC (S&P 500 index) automatically
- Robust retry handling
- Saves parquet + raw CSV + wide price/volume CSVs
"""

import os
import time
import math
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import yfinance as yf
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================
TICKERS_FILE = "stock_metadata_usv3.csv"

OUTPUT_DIR = "stock_data_usv3"
RAW_PARQUET = os.path.join(OUTPUT_DIR, "raw_prices_usv3.parquet")
RAW_CSV = os.path.join(OUTPUT_DIR, "raw_prices_usv3.csv")
WIDE_PRICES_CSV = os.path.join(OUTPUT_DIR, "wide_prices_usv3.csv")
WIDE_VOLUME_CSV = os.path.join(OUTPUT_DIR, "wide_volume_usv3.csv")

DATE_START = "2026-01-01"

THREAD_WORKERS = 12
BATCH_SIZE = 150
SLEEP_BETWEEN_BATCHES = 1.5
MAX_RETRIES = 2


# ============================================================
# UTILITIES
# ============================================================
def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_tickers(file_path=TICKERS_FILE):
    df = pd.read_csv(file_path)
    col = "Symbol" if "Symbol" in df.columns else df.columns[0]
    tickers = df[col].dropna().astype(str).str.strip().unique().tolist()
    print(f"Loaded {len(tickers)} US tickers.")
    return tickers


def read_existing_parquet(path=RAW_PARQUET):
    if not os.path.exists(path):
        print("No existing US parquet found. Full backfill will be executed.")
        return pd.DataFrame(columns=["symbol", "date", "price", "volume"])

    try:
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        print(f"Existing parquet found: {path} ({len(df)} rows)")
        return df
    except Exception as e:
        print("⚠️ WARNING: Parquet read error:", e)
        return pd.DataFrame(columns=["symbol", "date", "price", "volume"])


def generate_all_business_dates(start_date, end_date):
    return [
        d.strftime("%Y-%m-%d")
        for d in pd.date_range(start=start_date, end=end_date, freq="B")
    ]


def determine_missing_pairs(tickers, existing_df, start_date):
    today = datetime.today().strftime("%Y-%m-%d")
    all_dates = generate_all_business_dates(start_date, today)

    if existing_df.empty:
        print("No existing data → all symbol-date pairs are missing.")
        return [(sym, d) for sym in tickers for d in all_dates]

    existing_set = set(zip(existing_df["symbol"], existing_df["date"]))
    missing = []

    for sym in tickers:
        for d in all_dates:
            if (sym, d) not in existing_set:
                missing.append((sym, d))

    print(f"Missing symbol-date pairs: {len(missing):,}")
    return missing


# ============================================================
# SINGLE-TICKER DOWNLOAD (with array-fix)
# ============================================================
def download_single_ticker(sym, min_date, max_date):

    try:
        data = yf.download(
            sym,
            start=min_date,
            end=max_date,
            progress=False,
            auto_adjust=False,
            threads=False
        )
    except Exception:
        return pd.DataFrame()

    if data.empty:
        return pd.DataFrame()

    # Price column selection
    if "Adj Close" in data.columns:
        price_series = data["Adj Close"]
    elif "Close" in data.columns:
        price_series = data["Close"]
    else:
        return pd.DataFrame()

    # Volume
    vol_series = data["Volume"] if "Volume" in data.columns else pd.Series(
        [None] * len(price_series), index=price_series.index
    )

    rows = []
    for dt, p, v in zip(price_series.index, price_series.values, vol_series.values):

        # FIX ARRAY CASE FOR p
        if hasattr(p, "__len__") and not isinstance(p, (float, int)):
            try:
                if len(p) == 1:
                    p = p[0]
                else:
                    continue
            except:
                continue

        if pd.isna(p):
            continue

        # FIX ARRAY CASE FOR v
        if hasattr(v, "__len__") and not isinstance(v, (float, int)):
            try:
                if len(v) == 1:
                    v = v[0]
                else:
                    v = None
            except:
                v = None

        rows.append({
            "symbol": sym,
            "date": pd.to_datetime(dt).strftime("%Y-%m-%d"),
            "price": float(p),
            "volume": None if pd.isna(v) else int(v),
        })

    return pd.DataFrame(rows)


# ============================================================
# MULTITHREADED FETCH (outer batch tqdm + inner ticker tqdm)
# ============================================================
def fetch_missing_data(missing_pairs):

    if not missing_pairs:
        return pd.DataFrame()

    dates = sorted({d for _, d in missing_pairs})
    min_date = dates[0]
    max_date = dates[-1]

    unique_symbols = sorted({sym for sym, _ in missing_pairs})
    total_batches = math.ceil(len(unique_symbols) / BATCH_SIZE)

    print(f"Fetching {len(unique_symbols)} symbols in {total_batches} batches.")
    print(f"Date range: {min_date} → {max_date}\n")

    all_new_rows = []

    # OUTER PROGRESS BAR
    for batch_idx in tqdm(range(total_batches), desc="Batches", unit="batch"):
        start = batch_idx * BATCH_SIZE
        batch = unique_symbols[start:start + BATCH_SIZE]

        print(f"\n--- Batch {batch_idx+1}/{total_batches} | {len(batch)} tickers ---")

        for attempt in range(MAX_RETRIES + 1):
            collected_rows = []
            ticker_bar = tqdm(batch, desc=f"Batch {batch_idx+1} Tickers", unit="ticker", leave=False)

            with ThreadPoolExecutor(max_workers=THREAD_WORKERS) as executor:
                futures = {executor.submit(download_single_ticker, sym, min_date, max_date): sym for sym in batch}

                for future in as_completed(futures):
                    df_sym = future.result()
                    if df_sym is not None and not df_sym.empty:
                        collected_rows.append(df_sym)
                    ticker_bar.update(1)

            ticker_bar.close()

            if collected_rows:
                df_batch = pd.concat(collected_rows, ignore_index=True)
                print(f"✓ Batch {batch_idx+1} fetched {len(df_batch)} rows.")
                all_new_rows.append(df_batch)
                break
            else:
                print(f"⚠️ WARNING: Batch empty. Retry {attempt+1}/{MAX_RETRIES}")
                time.sleep(1)

        time.sleep(SLEEP_BETWEEN_BATCHES)

    if not all_new_rows:
        print("⚠️ WARNING: No new data downloaded.")
        return pd.DataFrame()

    df_all = pd.concat(all_new_rows, ignore_index=True)
    df_all.drop_duplicates(subset=["symbol", "date"], inplace=True)
    return df_all


# ============================================================
# FETCH SPX (^GSPC)
# ============================================================
def fetch_spx_data(start_date, end_date):
    """
    Fetch SPX (^GSPC) as long-format: symbol | date | price | volume
    """

    print("\n--- FETCHING ^GSPC (S&P 500) ---")

    try:
        raw = yf.download(
            "^GSPC",
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False,
            threads=False
        )

        if raw.empty:
            print("⚠️ WARNING: No ^GSPC data returned.")
            return pd.DataFrame([])

        # Flatten MultiIndex if needed
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        price_col = "Adj Close" if "Adj Close" in raw.columns else "Close"

        df = raw.reset_index()[["Date", price_col]].rename(columns={
            "Date": "date",
            price_col: "price"
        })

        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df["symbol"] = "^GSPC"
        df["volume"] = pd.NA
        df["volume"] = df["volume"].astype("Int64")

        df = df[["symbol", "date", "price", "volume"]]

        print(f"Fetched {len(df)} ^GSPC records.")
        return df

    except Exception as e:
        print(f"❌ ERROR fetching ^GSPC: {e}")
        return pd.DataFrame([])


# ============================================================
# MERGING + SAVING
# ============================================================
def merge_new_with_existing(existing, new):
    if new.empty:
        return existing

    if existing.empty:
        merged = new
    else:
        merged = pd.concat([existing, new], ignore_index=True)

    merged.drop_duplicates(subset=["symbol", "date"], keep="last", inplace=True)
    merged.sort_values(["symbol", "date"], inplace=True)
    merged["date"] = pd.to_datetime(merged["date"]).dt.strftime("%Y-%m-%d")
    print(f"Merged dataset now has {len(merged)} rows.")
    return merged


def save_outputs(df):
    df.to_parquet(RAW_PARQUET, index=False)
    print(f"Saved parquet: {RAW_PARQUET}")

    df.to_csv(RAW_CSV, index=False)
    print(f"Saved CSV: {RAW_CSV}")

    price_wide = df.pivot(index="date", columns="symbol", values="price").sort_index()
    price_wide.to_csv(WIDE_PRICES_CSV)
    print(f"Saved wide price CSV: {WIDE_PRICES_CSV}")

    volume_wide = df.pivot(index="date", columns="symbol", values="volume").sort_index()
    volume_wide.to_csv(WIDE_VOLUME_CSV)
    print(f"Saved wide volume CSV: {WIDE_VOLUME_CSV}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n=== US PRICE UPDATE v2.2 (MULTITHREADED + TQDM + SPX ADDED) ===\n")

    ensure_output_dir()
    tickers = load_tickers()

    existing = read_existing_parquet(RAW_PARQUET)
    missing_pairs = determine_missing_pairs(tickers, existing, DATE_START)

    # Fetch missing prices
    if missing_pairs:
        df_new = fetch_missing_data(missing_pairs)
    else:
        print("No missing US stock data.")
        df_new = pd.DataFrame([])

    # Merge
    df_merged = merge_new_with_existing(existing, df_new)

    # FETCH SPX
    if not df_merged.empty:
        min_date = df_merged["date"].min()
        max_date = df_merged["date"].max()
        spx_df = fetch_spx_data(min_date, max_date)

        if not spx_df.empty:
            df_merged = merge_new_with_existing(df_merged, spx_df)
            print("✓ ^GSPC merged successfully.\n")
        else:
            print("⚠️ SPX not merged because no data returned.\n")

    # SAVE OUTPUTS
    save_outputs(df_merged)

    print("\n✓ US price-update completed successfully!\n")


if __name__ == "__main__":
    main()
