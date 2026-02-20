"""
Deriv Historical Data Fetcher — with pagination for long history
"""

import asyncio
import json
import websockets
import pandas as pd
from datetime import datetime
import os
import time

APP_ID = "36544"
WS_URL = f"wss://ws.binaryws.com/websockets/v3?app_id={APP_ID}"

MAJOR_PAIRS = {
    "EURUSD": "frxEURUSD", "GBPUSD": "frxGBPUSD", "USDJPY": "frxUSDJPY",
    "AUDUSD": "frxAUDUSD", "USDCAD": "frxUSDCAD", "USDCHF": "frxUSDCHF",
    "NZDUSD": "frxNZDUSD", "EURGBP": "frxEURGBP", "EURJPY": "frxEURJPY",
    "GBPJPY": "frxGBPJPY",
}

TIMEFRAMES = {
    "M1": 60, "M5": 300, "M15": 900,
    "H1": 3600, "H4": 14400, "D1": 86400,
}

MAX_PER_REQUEST = 5000   # Deriv hard limit per request


async def _fetch_batch(ws, symbol: str, granularity: int,
                       count: int, end_epoch: str = "latest") -> list:
    """Fetch a single batch of candles."""
    req = {
        "ticks_history": symbol,
        "count":         count,
        "end":           end_epoch,
        "style":         "candles",
        "granularity":   granularity,
    }
    await ws.send(json.dumps(req))
    resp = json.loads(await ws.recv())
    if "error" in resp:
        raise RuntimeError(resp["error"]["message"])
    return resp.get("candles", [])


async def fetch_candles(symbol: str, granularity: int, total_count: int = 20000,
                        api_token: str = None, app_id: str = APP_ID) -> pd.DataFrame:
    """
    Fetch up to total_count candles via pagination (multiple requests).
    Deriv caps each request at 5000 candles, so we page backwards in time.
    """
    ws_url = f"wss://ws.binaryws.com/websockets/v3?app_id={app_id}"
    all_candles = []

    try:
        async with websockets.connect(ws_url, ping_interval=30) as ws:
            if api_token:
                await ws.send(json.dumps({"authorize": api_token}))
                auth = json.loads(await ws.recv())
                if "error" in auth:
                    print(f"  Auth warning: {auth['error']['message']}")

            remaining   = total_count
            end_epoch   = "latest"
            page        = 0

            while remaining > 0:
                batch_size = min(remaining, MAX_PER_REQUEST)
                candles    = await _fetch_batch(ws, symbol, granularity, batch_size, end_epoch)
                if not candles:
                    break

                all_candles = candles + all_candles   # prepend older data
                remaining  -= len(candles)
                page       += 1

                if len(candles) < batch_size:
                    break   # No more history available

                # Next page ends just before oldest candle in this batch
                oldest_epoch = candles[0]["epoch"] - granularity
                end_epoch    = str(oldest_epoch)

                if remaining > 0:
                    await asyncio.sleep(0.4)   # Polite pause

    except Exception as e:
        print(f"  Error fetching {symbol}: {type(e).__name__}: {e}")
        if not all_candles:
            return pd.DataFrame()

    if not all_candles:
        return pd.DataFrame()

    df = pd.DataFrame(all_candles)
    df["datetime"] = pd.to_datetime(df["epoch"], unit="s")
    df = df[["datetime", "open", "high", "low", "close"]].copy()
    df["volume"] = 0
    df[["open","high","low","close"]] = df[["open","high","low","close"]].astype(float)
    df = df.drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True)
    return df


async def fetch_all_pairs(timeframe: str = "H1", total_count: int = 20000,
                          api_token: str = None, output_dir: str = "data",
                          app_id: str = APP_ID) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    granularity = TIMEFRAMES.get(timeframe, 3600)
    all_data    = {}

    pages_needed = (total_count + MAX_PER_REQUEST - 1) // MAX_PER_REQUEST
    print(f"\n{'='*58}")
    print(f"  Deriv Historical Data Fetcher  (paginated)")
    print(f"  App ID    : {app_id}")
    print(f"  Timeframe : {timeframe} | Target candles: {total_count:,} ({pages_needed} pages each)")
    print(f"{'='*58}\n")

    for pair_name, symbol in MAJOR_PAIRS.items():
        print(f"  {pair_name} ({symbol})...", end=" ", flush=True)
        df = await fetch_candles(symbol, granularity, total_count, api_token, app_id)

        if not df.empty:
            filepath = os.path.join(output_dir, f"{pair_name}_{timeframe}.csv")
            df.to_csv(filepath, index=False)
            all_data[pair_name] = df
            start = df["datetime"].iloc[0].strftime("%Y-%m-%d")
            end   = df["datetime"].iloc[-1].strftime("%Y-%m-%d")
            print(f"✓  {len(df):,} candles  [{start} → {end}]")
        else:
            print("✗  No data")

        await asyncio.sleep(0.8)

    print(f"\n  Saved {len(all_data)} pairs to '{output_dir}/'")
    return all_data


def load_from_csv(pair: str, timeframe: str = "H1", data_dir: str = "data") -> pd.DataFrame:
    filepath = os.path.join(data_dir, f"{pair}_{timeframe}.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No data: {filepath}")
    return pd.read_csv(filepath, parse_dates=["datetime"])


if __name__ == "__main__":
    from deriv_fetcher import APP_ID as _ID
    cid   = input(f"app_id (Enter={_ID}): ").strip() or _ID
    token = input("API token (Enter=skip): ").strip() or None
    tf    = input("Timeframe [H1/H4/D1] (H1): ").strip() or "H1"
    cnt   = int(input("Total candles (20000): ").strip() or "20000")
    asyncio.run(fetch_all_pairs(timeframe=tf, total_count=cnt,
                                api_token=token, app_id=cid))