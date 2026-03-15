import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import yfinance as yf

# Load all tickers
sample_ticker = pd.read_csv('tickers.csv')['symbol'].tolist()

# Date range
end_date = pd.to_datetime('2025-06-20') ## change the date if necessary
start_date = pd.to_datetime('2024-01-01') ##change the date if necessary
sample_dates = pd.date_range(start=start_date, end=end_date, freq='B').strftime('%Y-%m-%d').tolist()

def fetch_price(symbol, date):
    ##print(f"Requesting {symbol} on {date} ...")
    url = f'https://phisix-api3.appspot.com/stocks/{symbol}.{date}.json'
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return None
        json_data = resp.json()
        if 'stock' not in json_data:
            print(f"No 'stock' in response for {symbol} on {date}")
            return None
        for item in json_data['stock']:
            return {
                'symbol': item['symbol'],
                'date': date,
                'price': item['price']['amount'],
            }
    except requests.exceptions.Timeout:
        print(f"Timeout for {symbol} on {date}, skipping this date.")
    except requests.exceptions.RequestException as e:
        print(f"Request failed for {symbol} on {date}: {e}")
    return None

rows = []
failed_queries = []
max_workers = 16  # Adjust as needed

def run_queries(pairs):
    results = []
    failed = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_price, symbol, date): (symbol, date) for symbol, date in pairs}
        for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Fetching prices")):
            result = future.result()
            symbol, date = futures[future]
            if result:
                results.append(result)
            else:
                failed.append((symbol, date))
            if i % 100 == 0:
                print(f"Processed {i} requests out of {len(futures)}...")
    return results, failed

# Prepare all (symbol, date) pairs
all_pairs = [(symbol, date) for symbol in sample_ticker for date in sample_dates]

# First run
rows, failed_queries = run_queries(all_pairs)

# Retry failed queries once
if failed_queries:
    print(f"Retrying {len(failed_queries)} failed queries...")
    retry_rows, retry_failed = run_queries(failed_queries)
    rows.extend(retry_rows)
    if retry_failed:
        print(f"Still failed after retry: {len(retry_failed)} queries.")
    else:
        print("All failed queries succeeded on retry.")

df = pd.DataFrame(rows)
if not df.empty:
    df_pivot = df.pivot(index='date', columns='symbol', values='price')
    df = df_pivot.reset_index()

    print('Downloading PSEI closing prices')

    psei = yf.download('PSEI.PS', start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    if not psei.empty and 'Close' in psei.columns:
        psei_close = psei['Close'].reset_index()
        psei_close = psei_close.rename(columns={'Date': 'date', 'Close': 'PSEI.PS'})
        df['date'] = pd.to_datetime(df['date'])
        df = pd.merge(df, psei_close, on='date', how='left')
    else:
        print("Warning: No PSEI.PS close data found from yfinance.")

    df.to_csv('historical_prices_2.csv', index=False)
    print("Code is finished running")
    print(df)

# Summary of results
    total_requests = len(all_pairs)
    total_success = len(rows)
    total_failed = len(failed_queries) + (len(retry_failed) if 'retry_failed' in locals() else 0)
    print("\n--- Summary ---")
    print(f"Total requests attempted: {total_requests}")
    print(f"Total successful fetches: {total_success}")
    print(f"Total failed fetches after retry: {total_failed}")
    print(f"Final DataFrame shape: {df.shape}")
else:
    print("No data was fetched.")