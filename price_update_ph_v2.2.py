"""
PH Stock Prices Updated ver2.3 (Modular + Duplicate-Fix + Correct PSEI Long Format)

This code updates recorded historical price and volume for Philippine stocks.
The data for PH stocks comes from the PHISIX API.
The data for PSEI.PS comes from Yahoo Finance.

Outputs:
- raw_prices.parquet   (long format)
- raw_prices.csv
- wide_prices.csv      (date x symbols)
- wide_volume.csv
"""

# ---------------------------------------------------
# IMPORTS
# ---------------------------------------------------
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import requests
import pandas as pd
import yfinance as yf
import os
import time
from typing import Optional, Dict, List

# ---------------------------------------------------
# CONFIGURATIONS
# ---------------------------------------------------
DATA_DIR = "stock_data_ph"
os.makedirs(DATA_DIR, exist_ok=True)

PARQUET_FILE = os.path.join(DATA_DIR, "raw_prices_ph.parquet")
CSV_FILE = os.path.join(DATA_DIR, "raw_prices_ph.csv")
WIDE_PRICE_CSV = os.path.join(DATA_DIR, "wide_prices_ph.csv")
WIDE_VOLUME_CSV = os.path.join(DATA_DIR, "wide_volume_ph.csv")

TICKERS_FILE = "stock_metadata_ph.csv"
DATE_START = "2026-01-01"
DATE_END = pd.to_datetime("today").strftime("%Y-%m-%d")

MAX_WORKERS = 12
REQUEST_TIMEOUT = 20
BATCH_SIZE = 2000



# ---------------------------------------------------
# LOAD TICKERS
# ---------------------------------------------------
def load_tickers(path: str, max_rows: Optional[int] = None) -> List[str]:
    df = pd.read_csv(path)
    symbols = df["symbol"].astype(str).str.strip().tolist()
    return symbols if max_rows is None else symbols[:max_rows]


# ---------------------------------------------------
# FETCH PHISIX DATA
# ---------------------------------------------------
def fetch_phisix_price_volume(symbol: str, date: str) -> Optional[Dict]:
    url = f"https://phisix-api3.appspot.com/stocks/{symbol}.{date}.json"

    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
    except Exception as e:
        #print(f"[ERROR] {symbol} {date} -> request exception: {e}")
        return None

    if resp.status_code != 200:
        return None

    try:
        j = resp.json()
    except:
        return None

    stocks = j.get("stocks") or []
    if not stocks:
        return None

    item = stocks[0]
    price = item.get("price", {}).get("amount")
    volume = item.get("volume")

    if price is None:
        return None

    return {
        "symbol": item.get("symbol"),
        "date": date,
        "price": float(price),
        "volume": int(volume) if volume is not None else None,
    }


# ---------------------------------------------------
# PHISIX MULTITHREADED RUNNER
# ---------------------------------------------------
def run_phisix_fetch(pairs: List[tuple]) -> List[Dict]:
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        future_map = {ex.submit(fetch_phisix_price_volume, s, d): (s, d) for s, d in pairs}

        for fut in tqdm(as_completed(future_map), total=len(future_map), desc="Fetching PHISIX"):
            res = fut.result()
            if res:
                results.append(res)

    return results


# ---------------------------------------------------
# CHECK MISSING SYMBOL-DATE PAIRS
# ---------------------------------------------------
def build_missing_pairs(tickers: List[str], start_date: str, end_date: str):
    dates = (
        pd.date_range(start=start_date, end=end_date, freq="B")
        .strftime("%Y-%m-%d")
        .tolist()
    )

    if os.path.exists(PARQUET_FILE):
        hist = pd.read_parquet(PARQUET_FILE)
        hist["date"] = pd.to_datetime(hist["date"]).dt.strftime("%Y-%m-%d")
    else:
        hist = pd.DataFrame(columns=["symbol", "date", "price", "volume"])

    existing = set((row["symbol"], row["date"]) for row in hist.to_dict("records"))

    missing = [(s, d) for s in tickers for d in dates if (s, d) not in existing]
    return missing, hist


# ---------------------------------------------------
# FETCH PSEI DATA 
# ---------------------------------------------------
def fetch_psei_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch PSEI.PS from Yahoo Finance in correct long format:
    symbol | date | price | volume
    """

    print("\n--- FETCHING PSEI.PS (Yahoo Finance) ---")

    try:
        psei_raw = yf.download(
            "PSEI.PS",
            start=start_date,
            end=end_date,
            auto_adjust=False,
            threads=False
        )

        if psei_raw.empty:
            print("Warning: No PSEI.PS data returned.")
            return pd.DataFrame([])

        # 🔥 FIX: Flatten MultiIndex columns if present
        if isinstance(psei_raw.columns, pd.MultiIndex):
            psei_raw.columns = psei_raw.columns.get_level_values(0)

        # Prefer Adj Close if exists
        price_col = "Adj Close" if "Adj Close" in psei_raw.columns else "Close"

        # Create long-format DataFrame
        psei_df = psei_raw.reset_index()[[
            "Date", price_col
        ]].rename(columns={
            "Date": "date",
            price_col: "price"
        })

        psei_df["date"] = pd.to_datetime(psei_df["date"]).dt.strftime("%Y-%m-%d")
        psei_df["symbol"] = "PSEI.PS"
        psei_df["volume"] = pd.NA
        psei_df["volume"] = psei_df["volume"].astype("Int64")

        # Ensure correct column order
        psei_df = psei_df[["symbol", "date", "price", "volume"]]

        print(f"Fetched {len(psei_df)} PSEI.PS records")
        return psei_df

    except Exception as e:
        print(f"Error fetching PSEI.PS: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame([])


# ---------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------
def main():
    print("\n--- LOADING TICKERS ---")
    tickers = load_tickers(TICKERS_FILE)
    print(f"Loaded {len(tickers)} tickers.")

    print("\n--- BUILDING REQUEST LIST ---")
    missing_pairs, hist = build_missing_pairs(tickers, DATE_START, DATE_END)
    print(f"Missing symbol-date pairs: {len(missing_pairs)}")

    # ------------ FETCH PHISIX DATA ------------
    new_records = []

    if missing_pairs:
        print("\n--- FETCHING PHISIX DATA IN BATCHES ---")

        for i in range(0, len(missing_pairs), BATCH_SIZE):
            chunk = missing_pairs[i:i + BATCH_SIZE]
            print(f"[Batch {i//BATCH_SIZE + 1}] Count: {len(chunk)}")

            batch_rows = run_phisix_fetch(chunk)
            print(f" → Batch returned {len(batch_rows)} rows")

            new_records.extend(batch_rows)
            time.sleep(0.5)

    else:
        print("No new PHISIX data to fetch.")

    # ------------ CLEAN NEW DATA ------------
    if new_records:
        new_df = pd.DataFrame(new_records)
        new_df["date"] = pd.to_datetime(new_df["date"]).dt.strftime("%Y-%m-%d")
        new_df["volume"] = new_df["volume"].astype("Int64")
    else:
        new_df = pd.DataFrame([], columns=["symbol", "date", "price", "volume"])

    # Merge with historical data
    combined = pd.concat([hist, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["symbol", "date"], keep="last")
    combined = combined.sort_values(["symbol", "date"]).reset_index(drop=True)

    # ------------ FETCH & MERGE PSEI DATA ------------
    min_date = combined["date"].min()
    max_date = combined["date"].max()

    psei_df = fetch_psei_data(min_date, max_date)

    if not psei_df.empty:
        combined = pd.concat([combined, psei_df], ignore_index=True)

        # 🔥 Critical fix: remove duplicates AFTER adding PSEI
        combined = combined.drop_duplicates(subset=["symbol", "date"], keep="last")

        print("PSEI.PS merged successfully.")
    else:
        print("PSEI.PS not merged due to empty data.")

    # ------------ SAVE OUTPUTS ------------
    print("\n--- SAVING OUTPUT FILES ---")
    combined.to_parquet(PARQUET_FILE, index=False)
    combined.to_csv(CSV_FILE, index=False)

    price_wide = combined.pivot(index="date", columns="symbol", values="price").sort_index()
    volume_wide = combined.pivot(index="date", columns="symbol", values="volume").sort_index()

    price_wide.to_csv(WIDE_PRICE_CSV, index=True)
    volume_wide.to_csv(WIDE_VOLUME_CSV, index=True)

    print("Saved raw_prices.parquet, raw_prices.csv, wide_prices.csv, wide_volume.csv")
    print("\n✅ DONE. Execution successful.\n")


# ---------------------------------------------------
# RUN SCRIPT
# ---------------------------------------------------
if __name__ == "__main__":
    main()