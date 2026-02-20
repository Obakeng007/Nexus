"""
Data Manager v2 — Deriv API as primary data source
Priority order: 1) Deriv live fetch  2) CSV cache  3) Synthetic fallback
"""
import asyncio
import os
import json
import time
import threading
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")
STATUS_FILE = os.path.join(DATA_DIR, "_fetch_status.json")
os.makedirs(DATA_DIR, exist_ok=True)

INSTRUMENTS = {
    "EURUSD": {"deriv": "frxEURUSD", "base": 1.0850, "vol": 0.006,  "spread": 0.0001, "digits": 5},
    "GBPUSD": {"deriv": "frxGBPUSD", "base": 1.2650, "vol": 0.008,  "spread": 0.0001, "digits": 5},
    "USDJPY": {"deriv": "frxUSDJPY", "base": 149.50, "vol": 0.50,   "spread": 0.010,  "digits": 3},
    "AUDUSD": {"deriv": "frxAUDUSD", "base": 0.6520, "vol": 0.005,  "spread": 0.0001, "digits": 5},
    "USDCAD": {"deriv": "frxUSDCAD", "base": 1.3600, "vol": 0.0055, "spread": 0.0001, "digits": 5},
    "USDCHF": {"deriv": "frxUSDCHF", "base": 0.8920, "vol": 0.0045, "spread": 0.0001, "digits": 5},
    "NZDUSD": {"deriv": "frxNZDUSD", "base": 0.5980, "vol": 0.0045, "spread": 0.0002, "digits": 5},
    "XAUUSD": {"deriv": "frxXAUUSD", "base": 2020.0, "vol": 12.0,   "spread": 0.30,   "digits": 2},
}

TIMEFRAMES = {"M1": 60, "M5": 300, "M15": 900, "H1": 3600, "H4": 14400, "D1": 86400}

_fetch_status: dict = {}
_status_lock  = threading.Lock()

# ── Status helpers ─────────────────────────────────────────────────────────────
def _load_status():
    global _fetch_status
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE) as f:
                _fetch_status = json.load(f)
        except Exception:
            _fetch_status = {}

def _save_status():
    try:
        with open(STATUS_FILE, "w") as f:
            json.dump(_fetch_status, f, indent=2)
    except Exception:
        pass

def set_fetch_status(key: str, value: dict):
    with _status_lock:
        _fetch_status[key] = value
        _save_status()

def get_fetch_status() -> dict:
    with _status_lock:
        return dict(_fetch_status)

# ── CSV cache helpers ──────────────────────────────────────────────────────────
def _csv_path(pair: str, timeframe: str = "H1") -> str:
    return os.path.join(DATA_DIR, f"{pair}_{timeframe}.csv")

def save_df(df: pd.DataFrame, pair: str, timeframe: str = "H1"):
    df.to_csv(_csv_path(pair, timeframe), index=False)

def load_csv(pair: str, timeframe: str = "H1") -> pd.DataFrame | None:
    path = _csv_path(pair, timeframe)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        df.columns = [c.lower().strip() for c in df.columns]
        date_col = next((c for c in ["datetime","date","time","timestamp"] if c in df.columns), None)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            if date_col != "datetime":
                df.rename(columns={date_col: "datetime"}, inplace=True)
        else:
            df.insert(0, "datetime", pd.date_range(end=datetime.utcnow(), periods=len(df), freq="h"))
        for col in ["open","high","low","close"]:
            if col not in df.columns:
                return None
        if "volume" not in df.columns:
            df["volume"] = 0
        df = df[["datetime","open","high","low","close","volume"]].dropna(subset=["close"])
        df.sort_values("datetime", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    except Exception as e:
        print(f"CSV load error {pair}: {e}")
        return None

# ── Deriv fetch ────────────────────────────────────────────────────────────────
def fetch_from_deriv(pair: str, timeframe: str = "H1", count: int = 20000,
                     api_token: str = None, app_id: str = "36544") -> pd.DataFrame:
    """Synchronous wrapper around the async Deriv fetcher."""
    try:
        from deriv_fetcher import fetch_candles
    except ImportError:
        return pd.DataFrame()  # websockets not installed

    symbol = INSTRUMENTS.get(pair, {}).get("deriv")
    if not symbol:
        return pd.DataFrame()
    granularity = TIMEFRAMES.get(timeframe, 3600)

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        df = loop.run_until_complete(fetch_candles(symbol, granularity, count, api_token, app_id))
        loop.close()
        return df
    except Exception as e:
        print(f"Deriv fetch error [{pair}]: {e}")
        return pd.DataFrame()

# ── Synthetic fallback ─────────────────────────────────────────────────────────
def generate_synthetic(pair: str, periods: int = 2000) -> pd.DataFrame:
    cfg = INSTRUMENTS.get(pair, {"base": 1.0, "vol": 0.005, "digits": 5})
    np.random.seed(hash(pair) % (2**31))
    base, vol = cfg["base"], cfg["vol"]

    prices, regime, rc = [base], 0, 0
    for _ in range(periods - 1):
        rc += 1
        if rc > np.random.randint(30, 120):
            regime = np.random.choice([-1, 0, 0, 1])
            rc = 0
        shock = np.random.normal(regime * vol * 0.05, vol)
        mr    = -0.02 * (prices[-1] - base) / base
        prices.append(max(prices[-1] * (1 + shock + mr), base * 0.5))

    end   = datetime.utcnow()
    dates = [end - timedelta(hours=periods - i) for i in range(periods)]
    rows  = []
    for i, (dt, close) in enumerate(zip(dates, prices)):
        iv    = vol * np.random.uniform(0.3, 1.5)
        open_ = prices[i-1] if i else close * (1 + np.random.normal(0, iv*0.5))
        high  = max(close*(1+abs(np.random.normal(0,iv))), open_, close)
        low   = min(close*(1-abs(np.random.normal(0,iv))), open_, close)
        rows.append({"datetime":dt,"open":round(open_,cfg["digits"]),
                     "high":round(high,cfg["digits"]),"low":round(low,cfg["digits"]),
                     "close":round(close,cfg["digits"]),"volume":0})
    return pd.DataFrame(rows)

# ── Primary entry point ────────────────────────────────────────────────────────
def load_instrument_data(pair: str, timeframe: str = "H1", periods: int = 2000) -> pd.DataFrame:
    """Load data: CSV cache → Deriv live → synthetic."""
    # Fast path: cached CSV
    df = load_csv(pair, timeframe)
    if df is not None and len(df) >= 200:
        return df.tail(periods).reset_index(drop=True)

    # Try Deriv live
    df = fetch_from_deriv(pair, timeframe, periods)
    if not df.empty:
        save_df(df, pair, timeframe)
        set_fetch_status(f"{pair}_{timeframe}", {
            "status":"ok","rows":len(df),"source":"deriv",
            "last":str(df["datetime"].iloc[-1])
        })
        return df.tail(periods).reset_index(drop=True)

    # Synthetic
    print(f"  [{pair}] Using synthetic data")
    df = generate_synthetic(pair, periods)
    set_fetch_status(f"{pair}_{timeframe}", {"status":"synthetic","rows":len(df),"source":"synthetic"})
    return df

def get_all_instruments() -> list:
    return list(INSTRUMENTS.keys())

def get_instrument_info(pair: str) -> dict:
    return INSTRUMENTS.get(pair, {})

def cache_age_hours(pair: str, timeframe: str = "H1") -> float | None:
    path = _csv_path(pair, timeframe)
    if not os.path.exists(path):
        return None
    return (time.time() - os.path.getmtime(path)) / 3600

# ── Auto-refresh scheduler ─────────────────────────────────────────────────────
_refresh_thread: threading.Thread | None = None
_refresh_stop   = threading.Event()
_refresh_config = {
    "enabled": False, "interval_h": 4, "timeframe": "H1",
    "api_token": None, "app_id": "36544", "count": 20000, "retrain": True,
}

def start_auto_refresh(interval_h: float = 4, timeframe: str = "H1",
                       api_token: str = None, app_id: str = "36544",
                       count: int = 20000, auto_retrain: bool = True):
    global _refresh_thread
    _refresh_config.update({
        "enabled":True,"interval_h":interval_h,"timeframe":timeframe,
        "api_token":api_token,"app_id":app_id,"count":count,"retrain":auto_retrain,
    })
    _refresh_stop.clear()
    if _refresh_thread and _refresh_thread.is_alive():
        return
    def _worker():
        while not _refresh_stop.wait(interval_h * 3600):
            _do_refresh()
    _refresh_thread = threading.Thread(target=_worker, daemon=True, name="DataRefresh")
    _refresh_thread.start()
    print(f"Auto-refresh started: every {interval_h}h")

def stop_auto_refresh():
    _refresh_config["enabled"] = False
    _refresh_stop.set()

def trigger_refresh_now(pair: str | None = None):
    threading.Thread(target=_do_refresh, args=(pair,), daemon=True).start()

def _do_refresh(pair: str | None = None):
    cfg   = _refresh_config
    pairs = [pair] if pair else list(INSTRUMENTS.keys())
    tf    = cfg["timeframe"]
    for p in pairs:
        key = f"{p}_{tf}"
        set_fetch_status(key, {"status":"fetching"})
        df = fetch_from_deriv(p, tf, cfg["count"], cfg["api_token"], cfg["app_id"])
        if not df.empty:
            old = load_csv(p, tf)
            if old is not None and not old.empty:
                df = pd.concat([old, df]).drop_duplicates("datetime")
                df.sort_values("datetime", inplace=True)
                df.reset_index(drop=True, inplace=True)
            save_df(df, p, tf)
            set_fetch_status(key, {
                "status":"ok","rows":len(df),"source":"deriv",
                "last":str(df["datetime"].iloc[-1]),
                "refreshed_at":datetime.utcnow().isoformat(),
            })
            print(f"  [{p}] Refreshed: {len(df)} rows")
            if cfg["retrain"]:
                _retrain_pair(p, df)
        else:
            set_fetch_status(key, {"status":"fetch_failed"})

def _retrain_pair(pair: str, df: pd.DataFrame):
    try:
        from ml_engine import train_model
        train_model(df, pair)
        print(f"  [{pair}] Retrained after data refresh")
    except Exception as e:
        print(f"  [{pair}] Retrain failed: {e}")

_load_status()