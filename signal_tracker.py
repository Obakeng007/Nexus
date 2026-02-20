"""
Signal Tracker
==============
Manages the full lifecycle of a signal:

  PENDING  → entry price not yet reached (limit-style)
  OPEN     → position is live, monitoring SL/TP on every M15 candle
  CLOSED   → SL or TP was hit  (result: WIN / LOSS)
  EXPIRED  → max bars elapsed with no hit  (result: EXPIRED)
  INVALID  → signal was superseded before entry

Each pair can only have ONE active signal at a time.
When a signal closes, the pair is immediately re-scanned for a new setup.

Storage: data/signals.json  (persists across restarts)
"""

from __future__ import annotations
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

DATA_DIR     = os.path.join(os.path.dirname(__file__), "data")
SIGNALS_FILE = os.path.join(DATA_DIR, "signals.json")

os.makedirs(DATA_DIR, exist_ok=True)

_lock = threading.Lock()


# ── Persistence ────────────────────────────────────────────────────────────────

def _load_all() -> dict:
    if not os.path.exists(SIGNALS_FILE):
        return {}
    try:
        with open(SIGNALS_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_all(signals: dict):
    try:
        with open(SIGNALS_FILE, "w") as f:
            json.dump(signals, f, indent=2, default=str)
    except Exception as e:
        print(f"[Tracker] Save error: {e}")


# ── Public API ─────────────────────────────────────────────────────────────────

def get_active_signals() -> list[dict]:
    """Return all OPEN or PENDING signals."""
    with _lock:
        all_sigs = _load_all()
    return [s for s in all_sigs.values()
            if s.get("status") in ("OPEN", "PENDING")]


def get_all_signals(limit: int = 50) -> list[dict]:
    """Return recent signals (newest first)."""
    with _lock:
        all_sigs = _load_all()
    sigs = sorted(all_sigs.values(),
                  key=lambda s: s.get("created_at", ""), reverse=True)
    return sigs[:limit]


def get_signal_for_pair(pair: str) -> Optional[dict]:
    """Return the current active signal for a pair (or None)."""
    with _lock:
        all_sigs = _load_all()
    for s in all_sigs.values():
        if s["pair"] == pair and s["status"] in ("OPEN", "PENDING"):
            return s
    return None


def open_signal(pair: str, direction: str, entry: float,
                stop_loss: float, take_profit: float,
                grade: str = "B", rr_ratio: float = 0,
                timeframe: str = "M15",
                conditions: list = None,
                current_price: float = None) -> dict:
    """
    Record a new signal. Replaces any existing PENDING/OPEN signal for the pair.
    """
    with _lock:
        all_sigs = _load_all()

        # Invalidate any previous open signal for this pair
        for sig in all_sigs.values():
            if sig["pair"] == pair and sig["status"] in ("OPEN", "PENDING"):
                sig["status"]    = "INVALID"
                sig["closed_at"] = _now()
                sig["note"]      = "Superseded by new signal"

        sid = str(uuid.uuid4())[:8]
        sig = {
            "id":            sid,
            "pair":          pair,
            "direction":     direction,
            "entry":         round(entry, 8),
            "stop_loss":     round(stop_loss, 8),
            "take_profit":   round(take_profit, 8),
            "grade":         grade,
            "rr_ratio":      round(rr_ratio, 2),
            "timeframe":     timeframe,
            "conditions":    conditions or [],
            "status":        "OPEN",
            "created_at":    _now(),
            "closed_at":     None,
            "result":        None,
            "exit_price":    None,
            "bars_open":     0,
            "max_favour":    0.0,   # max favourable excursion
            "max_adverse":   0.0,   # max adverse excursion
            "current_price": current_price or entry,
            "pnl_pips":      0.0,
            "note":          "",
        }
        all_sigs[sid] = sig
        _save_all(all_sigs)
        print(f"[Tracker] OPENED {direction} {pair}  entry={entry}  SL={stop_loss}  TP={take_profit}  grade={grade}")
        return sig


def close_signal(sig_id: str, exit_price: float, result: str,
                 note: str = "") -> Optional[dict]:
    """Manually close a signal."""
    with _lock:
        all_sigs = _load_all()
        sig = all_sigs.get(sig_id)
        if not sig:
            return None
        sig["status"]     = "CLOSED"
        sig["result"]     = result
        sig["exit_price"] = round(exit_price, 8)
        sig["closed_at"]  = _now()
        sig["note"]       = note
        sig["pnl_pips"]   = _calc_pips(sig["pair"], sig["direction"],
                                        sig["entry"], exit_price)
        _save_all(all_sigs)
        print(f"[Tracker] CLOSED {sig['direction']} {sig['pair']}  "
              f"exit={exit_price}  result={result}  pips={sig['pnl_pips']:+.1f}")
        return sig


def check_signals_on_candle(pair: str, candle_high: float,
                             candle_low: float, candle_close: float,
                             candle_time: str = "") -> Optional[dict]:
    """
    Called on every new M15 candle for a pair.
    Returns the closed signal dict if SL/TP was hit, else None.
    """
    with _lock:
        all_sigs = _load_all()

    active = None
    for sig in all_sigs.values():
        if sig["pair"] == pair and sig["status"] in ("OPEN", "PENDING"):
            active = sig
            break

    if not active:
        return None

    sid       = active["id"]
    direction = active["direction"]
    entry     = active["entry"]
    sl        = active["stop_loss"]
    tp        = active["take_profit"]

    # Track excursions
    if direction == "BUY":
        favour  = candle_high  - entry
        adverse = entry - candle_low
    else:
        favour  = entry - candle_low
        adverse = candle_high - entry

    with _lock:
        all_sigs = _load_all()
        sig = all_sigs.get(sid)
        if not sig:
            return None

        sig["bars_open"]     = sig.get("bars_open", 0) + 1
        sig["current_price"] = candle_close
        sig["max_favour"]    = max(sig.get("max_favour", 0), favour)
        sig["max_adverse"]   = max(sig.get("max_adverse", 0), adverse)
        sig["pnl_pips"]      = _calc_pips(pair, direction, entry, candle_close)

        result     = None
        exit_price = None

        if direction == "BUY":
            if candle_low <= sl:
                result = "LOSS"; exit_price = sl
            elif candle_high >= tp:
                result = "WIN";  exit_price = tp
        else:  # SELL
            if candle_high >= sl:
                result = "LOSS"; exit_price = sl
            elif candle_low <= tp:
                result = "WIN";  exit_price = tp

        # Expire after 96 bars (~24h on M15) if no hit
        if result is None and sig["bars_open"] >= 96:
            result = "EXPIRED"; exit_price = candle_close

        if result:
            sig["status"]     = "CLOSED"
            sig["result"]     = result
            sig["exit_price"] = round(exit_price, 8)
            sig["closed_at"]  = candle_time or _now()
            sig["pnl_pips"]   = _calc_pips(pair, direction, entry, exit_price)
            _save_all(all_sigs)
            print(f"[Tracker] {result}  {direction} {pair}  "
                  f"bars={sig['bars_open']}  pips={sig['pnl_pips']:+.1f}")
            return sig
        else:
            _save_all(all_sigs)
            return None


def get_stats() -> dict:
    """Summary statistics across all closed signals."""
    with _lock:
        all_sigs = _load_all()
    closed = [s for s in all_sigs.values() if s["status"] == "CLOSED"]
    if not closed:
        return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0,
                "total_pips": 0, "avg_pips": 0, "open": 0}

    wins   = [s for s in closed if s.get("result") == "WIN"]
    losses = [s for s in closed if s.get("result") == "LOSS"]
    pips   = [s.get("pnl_pips", 0) for s in closed]

    return {
        "total":      len(closed),
        "wins":       len(wins),
        "losses":     len(losses),
        "expired":    len([s for s in closed if s.get("result") == "EXPIRED"]),
        "win_rate":   round(len(wins) / len(closed) * 100, 1) if closed else 0,
        "total_pips": round(sum(pips), 1),
        "avg_pips":   round(sum(pips) / len(pips), 1) if pips else 0,
        "open":       len([s for s in all_sigs.values() if s["status"] in ("OPEN","PENDING")]),
        "best_trade": round(max(pips), 1) if pips else 0,
        "worst_trade":round(min(pips), 1) if pips else 0,
    }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _calc_pips(pair: str, direction: str,
               entry: float, exit_price: float) -> float:
    """Calculate P&L in pips."""
    diff = exit_price - entry if direction == "BUY" else entry - exit_price
    # XAU: 1 pip = 0.01; JPY pairs: 1 pip = 0.01; others: 1 pip = 0.0001
    if "XAU" in pair:
        multiplier = 100
    elif "JPY" in pair:
        multiplier = 100
    else:
        multiplier = 10000
    return round(diff * multiplier, 1)
