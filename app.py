"""
Trading System API v2 — Flask backend
- Deriv data integration
- Auto-fetch + auto-retrain every 5 hours (starts on boot)
- Proper ML signals from trained models
"""
import io
import json
import traceback
import threading
import time
import os
from datetime import datetime, timezone
from flask import Flask, jsonify, request, send_from_directory
import pandas as pd
import numpy as np
import warnings
# Suppress joblib/sklearn config-propagation noise
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from data_manager import (
    load_instrument_data, get_all_instruments, save_df, get_fetch_status,
    set_fetch_status, start_auto_refresh, stop_auto_refresh, trigger_refresh_now,
    cache_age_hours, INSTRUMENTS,
)
from features import add_features
from ml_engine import train_model, generate_signal, load_model
from backtester import run_backtest
from signal_tracker import (
    get_active_signals, get_all_signals, get_signal_for_pair,
    open_signal, close_signal, check_signals_on_candle, get_stats,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB

# ── NaN-safe JSON serialiser ───────────────────────────────────────────────────
class _NaNSafeEncoder(json.JSONEncoder):
    """Convert NaN / Inf / numpy scalars to JSON-safe values."""
    def default(self, obj):
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            return None if (np.isnan(v) or np.isinf(v)) else v
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, (pd.NA.__class__,)):
            return None
        return super().default(obj)

    def iterencode(self, o, _one_shot=False):
        # Walk the whole structure and replace NaN before encoding
        o = _clean(o)
        return super().iterencode(o, _one_shot)


def _clean(obj):
    """Recursively replace NaN/Inf/numpy types with JSON-safe equivalents."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        return _clean(obj.tolist())
    try:
        import pandas as _pd
        if obj is _pd.NA or obj is _pd.NaT:
            return None
    except Exception:
        pass
    return obj


def safe_json(data, status=200):
    """Return a Flask Response with NaN-safe JSON."""
    text = json.dumps(_clean(data), cls=_NaNSafeEncoder, allow_nan=False)
    from flask import Response
    return Response(text, status=status, mimetype="application/json")

_training_status: dict = {}
_training_lock   = threading.Lock()

# ── Auto-cycle state (shared with frontend via /api/system_status) ─────────────
_system_state = {
    "auto_refresh_interval_h": 5,
    "last_fetch_at":      None,
    "next_fetch_at":      None,
    "last_train_at":      None,
    "next_train_at":      None,
    "last_signal_at":     None,
    "scheduler_running":  False,
    # M15 refresh state
    "m15_refresh_interval_m": 15,
    "last_m15_at":        None,
    "next_m15_at":        None,
    "m15_scheduler_running": False,
}
_state_lock = threading.Lock()

# ── Cached M15 entry analysis (pair → analysis dict) ──────────────────────────
_m15_cache: dict = {}
_m15_lock = threading.Lock()


# ── Serve frontend ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("templates", "index.html")


# ── Instruments ────────────────────────────────────────────────────────────────
@app.route("/api/instruments")
def get_instruments():
    result = []
    for sym in get_all_instruments():
        model = load_model(sym)
        age   = cache_age_hours(sym)
        result.append({
            "symbol":   sym,
            "trained":  model is not None,
            "accuracy": round(model["metrics"]["accuracy"] * 100, 1) if model else None,
            "trained_at": model.get("trained_at") if model else None,
            "cache_age_h": round(age, 1) if age is not None else None,
        })
    return jsonify({"instruments": result})


# ── Single signal ──────────────────────────────────────────────────────────────
@app.route("/api/signal/<pair>")
def get_signal(pair):
    if pair not in INSTRUMENTS:
        return jsonify({"error": "Unknown instrument"}), 404
    try:
        df  = load_instrument_data(pair)
        sig = generate_signal(df, pair, refine_with_m15=True)
        # Overlay cached M15 analysis if fresher
        _overlay_m15_cache(sig, pair)
        return safe_json(sig)
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


# ── All signals ────────────────────────────────────────────────────────────────
@app.route("/api/signals/all")
def get_all_signals():
    signals = []
    active  = {s["pair"]: s for s in get_active_signals()}
    for pair in get_all_instruments():
        try:
            # M15 primary — use cached M15 analysis if available
            with _m15_lock:
                cached_buy  = _m15_cache.get(f"{pair}_BUY")
                cached_sell = _m15_cache.get(f"{pair}_SELL")

            # Pick the stronger cached direction
            sig = None
            if cached_buy and cached_sell:
                b_ok = len(cached_buy.get("conditions", []))
                s_ok = len(cached_sell.get("conditions", []))
                best = cached_buy if b_ok >= s_ok else cached_sell
                direction = "BUY" if b_ok >= s_ok else "SELL"
                if best.get("grade") in ("A","B","C") and best.get("grade") != "W":
                    sig = _m15_to_signal(pair, best, direction)

            # Fallback to H1 ML signal if no M15 cache
            if sig is None:
                df  = load_instrument_data(pair)
                sig = generate_signal(df, pair, refine_with_m15=False)
                _overlay_m15_cache(sig, pair)

            # Annotate with active tracker state
            tracker_sig = active.get(pair)
            if tracker_sig:
                sig["tracker"] = tracker_sig
                sig["has_open_signal"] = True
            else:
                sig["has_open_signal"] = False

            signals.append(sig)
        except Exception as e:
            signals.append({"instrument": pair, "pair": pair, "error": str(e)})
    return safe_json({"signals": signals})


def _m15_to_signal(pair: str, m15: dict, direction: str) -> dict:
    """Convert a cached M15 analysis dict into a full signal dict."""
    from entry_refiner import _pip_mult
    entry = m15.get("m15_entry") or m15.get("entry")
    sl    = m15.get("m15_sl")    or m15.get("stop_loss")
    tp    = m15.get("m15_tp")    or m15.get("take_profit")
    rr    = m15.get("rr_ratio",  0)
    grade = m15.get("grade",     "C")
    n     = len(m15.get("conditions", []))
    return {
        "pair":          pair,
        "instrument":    pair,
        "signal":        direction,
        "grade":         grade,
        "source":        "M15",
        "entry":         entry,
        "stop_loss":     sl,
        "take_profit":   tp,
        "rr_ratio":      rr,
        "confidence":    min(50 + n * 8, 95),
        "conditions":    m15.get("conditions", []),
        "wait_reason":   m15.get("wait_reason", ""),
        "current_price": entry,
        "m15":           m15,
        "model_trained": False,
        "entry_grade":   grade,
        "entry_type":    "M15 primary",
        "indicators": {
            "rsi":       m15.get("m15_rsi"),
            "stoch_k":   m15.get("m15_stoch_k"),
            "stoch_d":   m15.get("m15_stoch_d"),
            "macd_hist": m15.get("m15_macd_hist"),
            "bb_pos":    m15.get("m15_bb_pos"),
            "ema_cross": m15.get("m15_ema_cross"),
        }
    }


def _overlay_m15_cache(sig: dict, pair: str):
    """Inject cached M15 analysis into a signal dict (non-blocking)."""
    direction = sig.get("signal", "HOLD")
    if direction == "HOLD":
        return
    with _m15_lock:
        cached = _m15_cache.get(f"{pair}_{direction}")
    if not cached:
        return
    sig["m15"] = cached
    if cached.get("available") and cached.get("grade") in ("A", "B", "C") and cached.get("m15_entry"):
        sig["entry"]       = cached["m15_entry"]
        sig["stop_loss"]   = cached["m15_sl"]
        sig["take_profit"] = cached["m15_tp"]
        sig["rr_ratio"]    = cached["rr_ratio"]
        sig["entry_grade"] = cached["grade"]
        sig["entry_type"]  = "M15 refined"
    elif cached.get("grade") == "W":
        sig["entry_grade"] = "W"
        sig["entry_type"]  = "Wait for M15 setup"


# ── Train single ───────────────────────────────────────────────────────────────
@app.route("/api/train/<pair>", methods=["POST"])
def train_pair(pair):
    if pair not in INSTRUMENTS:
        return jsonify({"error": "Unknown instrument"}), 404
    body          = request.get_json() or {}
    forward_bars  = int(body.get("forward_bars", 5))
    atr_mult      = float(body.get("atr_multiplier", 0.5))
    periods       = int(body.get("periods", 20000))

    with _training_lock:
        _training_status[pair] = {"status": "training"}

    def _do():
        try:
            df = load_instrument_data(pair, periods=periods)
            _, metrics = train_model(df, pair, forward_bars, atr_mult)
            with _training_lock:
                _training_status[pair] = {"status": "complete", "metrics": metrics}
        except Exception as e:
            with _training_lock:
                _training_status[pair] = {"status": "error", "error": str(e)}

    threading.Thread(target=_do, daemon=True).start()
    return jsonify({"message": f"Training started for {pair}", "status": "training"})


# ── Train all ──────────────────────────────────────────────────────────────────
@app.route("/api/train/all", methods=["POST"])
def train_all():
    body         = request.get_json() or {}
    forward_bars = int(body.get("forward_bars", 5))
    atr_mult     = float(body.get("atr_multiplier", 0.5))
    periods      = int(body.get("periods", 20000))

    for pair in get_all_instruments():
        with _training_lock:
            _training_status[pair] = {"status": "training"}

    def _do():
        for pair in get_all_instruments():
            try:
                df = load_instrument_data(pair, periods=periods)
                _, metrics = train_model(df, pair, forward_bars, atr_mult)
                with _training_lock:
                    _training_status[pair] = {"status": "complete", "metrics": metrics}
            except Exception as e:
                with _training_lock:
                    _training_status[pair] = {"status": "error", "error": str(e)}

    threading.Thread(target=_do, daemon=True).start()
    return jsonify({"message": "Training all instruments"})


# ── Training status ────────────────────────────────────────────────────────────
@app.route("/api/training_status")
def training_status():
    with _training_lock:
        return jsonify(dict(_training_status))


# ── Backtest ───────────────────────────────────────────────────────────────────
@app.route("/api/backtest/<pair>", methods=["POST"])
def backtest(pair):
    if pair not in INSTRUMENTS:
        return jsonify({"error": "Unknown instrument"}), 404
    body    = request.get_json() or {}
    capital = float(body.get("initial_capital", 10000))
    risk    = float(body.get("risk_per_trade", 0.02))
    conf    = float(body.get("confidence_threshold", 55))
    atr_sl  = float(body.get("atr_sl", 2.0))
    atr_tp  = float(body.get("atr_tp", 3.0))
    periods = int(body.get("periods", 5000))

    try:
        df = load_instrument_data(pair, periods=periods)
        result = run_backtest(df, pair, initial_capital=capital, risk_per_trade=risk,
                              confidence_threshold=conf, atr_sl_multiplier=atr_sl,
                              atr_tp_multiplier=atr_tp)
        return safe_json(result)
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


# ── Chart data ─────────────────────────────────────────────────────────────────
@app.route("/api/chart_data/<pair>")
def chart_data(pair):
    if pair not in INSTRUMENTS:
        return jsonify({"error": "Unknown instrument"}), 404
    periods = int(request.args.get("periods", 300))
    try:
        df     = load_instrument_data(pair, periods=periods)
        df_ind = add_features(df.copy())

        # Compute proper Bollinger Bands before downsampling
        roll_std       = df_ind["close"].rolling(20).std()
        df_ind["bb_upper"] = df_ind.get("sma_20", df_ind["close"].rolling(20).mean()) + 2 * roll_std
        df_ind["bb_lower"] = df_ind.get("sma_20", df_ind["close"].rolling(20).mean()) - 2 * roll_std

        step = max(1, len(df_ind) // 400)
        ds   = df_ind.iloc[::step].reset_index(drop=True)

        def sl(col):
            """Serialize a column, replacing any NaN/Inf with None."""
            if col not in ds.columns:
                return []
            out = []
            for v in ds[col]:
                try:
                    f = float(v)
                    out.append(None if (np.isnan(f) or np.isinf(f)) else round(f, 8))
                except (TypeError, ValueError):
                    out.append(None)
            return out

        if "datetime" in ds.columns:
            dates = [str(d)[:19] for d in ds["datetime"]]
        else:
            dates = [str(d)[:19] for d in ds.index]

        return safe_json({
            "dates":     dates,
            "open":      sl("open"),
            "high":      sl("high"),
            "low":       sl("low"),
            "close":     sl("close"),
            "volume":    sl("volume"),
            "ema_10":    sl("ema_10"),
            "ema_20":    sl("ema_20"),
            "ema_50":    sl("ema_50"),
            "sma_20":    sl("sma_20"),
            "sma_50":    sl("sma_50"),
            "bb_upper":  sl("bb_upper"),
            "bb_lower":  sl("bb_lower"),
            "rsi":       sl("rsi_14"),
            "rsi_7":     sl("rsi_7"),
            "macd_hist": sl("macd_hist"),
            "stoch_k":   sl("stoch_k"),
            "stoch_d":   sl("stoch_d"),
            "atr_ratio": sl("atr_ratio"),
            "bb_pos":    sl("bb_pos"),
            "bb_width":  sl("bb_width"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── M15 chart data ────────────────────────────────────────────────────────────
@app.route("/api/m15_chart/<pair>")
def m15_chart(pair):
    if pair not in INSTRUMENTS:
        return jsonify({"error": "Unknown instrument"}), 404
    bars = int(request.args.get("bars", 120))
    try:
        from entry_refiner import get_m15_chart_data
        data = get_m15_chart_data(pair, bars)
        return jsonify(data)
    except Exception as e:
        return jsonify({"available": False, "error": str(e)})


# ── M15 entry analysis ─────────────────────────────────────────────────────────
@app.route("/api/m15_analysis/<pair>")
def m15_analysis(pair):
    if pair not in INSTRUMENTS:
        return jsonify({"error": "Unknown instrument"}), 404
    direction = request.args.get("direction", "BUY").upper()
    try:
        from entry_refiner import get_m15_data, analyse_m15
        df_m15 = get_m15_data(pair, 300)
        result = analyse_m15(df_m15, direction)
        return jsonify(result)
    except Exception as e:
        return jsonify({"available": False, "error": str(e)})


# ── Fetch M15 data for a pair ──────────────────────────────────────────────────
@app.route("/api/deriv/fetch_m15/<pair>", methods=["POST"])
def fetch_m15(pair):
    pair = pair.upper().strip()
    if pair not in INSTRUMENTS:
        # Return available pairs to help debug
        return jsonify({
            "error": f"Unknown instrument: {pair}",
            "available": list(INSTRUMENTS.keys())
        }), 404
    body      = request.get_json() or {}
    api_token = body.get("api_token")
    app_id    = body.get("app_id", "36544")
    count     = int(body.get("count", 2000))
    def _do():
        from data_manager import fetch_from_deriv, load_csv, save_df, set_fetch_status
        key = f"{pair}_M15"
        set_fetch_status(key, {"status": "fetching"})
        df = fetch_from_deriv(pair, "M15", count, api_token, app_id)
        if not df.empty:
            old = load_csv(pair, "M15")
            if old is not None and not old.empty:
                df = pd.concat([old.tail(5000), df]).drop_duplicates("datetime")
                df.sort_values("datetime", inplace=True)
                df.reset_index(drop=True, inplace=True)
            save_df(df, pair, "M15")
            set_fetch_status(key, {
                "status": "ok", "rows": len(df), "source": "deriv",
                "last": str(df["datetime"].iloc[-1]),
            })
            print(f"  [{pair}] M15 fetched: {len(df)} rows")
        else:
            set_fetch_status(key, {"status": "fetch_failed"})
            print(f"  [{pair}] M15 fetch failed")
    threading.Thread(target=_do, daemon=True).start()
    return jsonify({"ok": True, "message": f"Fetching M15 for {pair} ({count} bars)"})


# ── Data source / Deriv config ─────────────────────────────────────────────────
@app.route("/api/deriv/fetch/<pair>", methods=["POST"])
def deriv_fetch_pair(pair):
    """Trigger an immediate Deriv fetch for one pair."""
    if pair not in INSTRUMENTS:
        return jsonify({"error": "Unknown instrument"}), 404
    body      = request.get_json() or {}
    api_token = body.get("api_token")
    app_id    = body.get("app_id", "36544")
    timeframe = body.get("timeframe", "H1")
    count     = int(body.get("count", 20000))

    def _do():
        from data_manager import fetch_from_deriv
        set_fetch_status(f"{pair}_{timeframe}", {"status": "fetching"})
        df = fetch_from_deriv(pair, timeframe, count, api_token, app_id)
        if not df.empty:
            save_df(df, pair, timeframe)
            set_fetch_status(f"{pair}_{timeframe}", {
                "status": "ok", "rows": len(df), "source": "deriv",
                "last": str(df["datetime"].iloc[-1]),
            })
        else:
            set_fetch_status(f"{pair}_{timeframe}", {"status": "fetch_failed"})

    threading.Thread(target=_do, daemon=True).start()
    return jsonify({"message": f"Fetching {pair} from Deriv"})


@app.route("/api/deriv/fetch_all", methods=["POST"])
def deriv_fetch_all():
    """Trigger Deriv fetch for all pairs."""
    body      = request.get_json() or {}
    api_token = body.get("api_token")
    app_id    = body.get("app_id", "36544")
    timeframe = body.get("timeframe", "H1")
    count     = int(body.get("count", 20000))
    retrain   = bool(body.get("auto_retrain", True))

    trigger_refresh_now()  # uses stored config
    return jsonify({"message": "Fetching all pairs from Deriv in background"})


@app.route("/api/deriv/configure", methods=["POST"])
def deriv_configure():
    """Configure and start auto-refresh."""
    body       = request.get_json() or {}
    interval_h = float(body.get("interval_h", 4))
    timeframe  = body.get("timeframe", "H1")
    api_token  = body.get("api_token")
    app_id     = body.get("app_id", "36544")
    count      = int(body.get("count", 20000))
    retrain    = bool(body.get("auto_retrain", True))

    start_auto_refresh(interval_h, timeframe, api_token, app_id, count, retrain)
    return jsonify({"message": f"Auto-refresh configured: every {interval_h}h"})


@app.route("/api/deriv/stop", methods=["POST"])
def deriv_stop():
    stop_auto_refresh()
    return jsonify({"message": "Auto-refresh stopped"})


@app.route("/api/fetch_status")
def fetch_status():
    return jsonify(get_fetch_status())


# ── CSV Upload ─────────────────────────────────────────────────────────────────
@app.route("/api/upload/<pair>", methods=["POST"])
def upload(pair):
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    file = request.files["file"]
    tf   = request.form.get("timeframe", "H1")
    try:
        content = file.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(content))
        df.columns = [c.lower().strip() for c in df.columns]
        date_col = next((c for c in ["datetime","date","time","timestamp"] if c in df.columns), None)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            df.rename(columns={date_col: "datetime"}, inplace=True)
        save_df(df, pair, tf)
        return jsonify({"message": f"Uploaded {pair} ({tf})", "rows": len(df)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Model info ─────────────────────────────────────────────────────────────────
@app.route("/api/model_info/<pair>")
def model_info(pair):
    model = load_model(pair)
    if not model:
        return jsonify({"trained": False, "pair": pair})
    top = dict(sorted(
        zip(model["feature_cols"], model["rf"].feature_importances_),
        key=lambda x: -x[1]
    )[:10])
    return jsonify({
        "trained":      True,
        "pair":         pair,
        "metrics":      model["metrics"],
        "trained_at":   model.get("trained_at"),
        "features":     len(model["feature_cols"]),
        "top_features": {k: round(float(v), 4) for k, v in top.items()},
    })


# ── System status (used by frontend countdown) ─────────────────────────────────
@app.route("/api/system_status")
def system_status():
    with _state_lock:
        state = dict(_system_state)
    # Add per-pair model status summary
    trained = sum(1 for p in get_all_instruments() if load_model(p) is not None)
    state["models_trained"]        = trained
    state["total_pairs"]           = len(get_all_instruments())
    state["server_time"]           = datetime.now(timezone.utc).isoformat()
    with _m15_lock:
        state["m15_pairs_cached"]  = len(_m15_cache)
    return jsonify(state)

@app.route("/api/system_status", methods=["POST"])
def update_system_status():
    """Frontend can update refresh interval."""
    body = request.get_json() or {}
    with _state_lock:
        if "interval_h" in body:
            _system_state["auto_refresh_interval_h"] = float(body["interval_h"])
    _restart_scheduler()
    return jsonify({"ok": True})


# ── Signal Tracker endpoints ──────────────────────────────────────────────────

@app.route("/api/tracker/active")
def tracker_active():
    return safe_json({"signals": get_active_signals()})


@app.route("/api/tracker/history")
def tracker_history():
    limit = int(request.args.get("limit", 50))
    return safe_json({"signals": get_all_signals(limit)})


@app.route("/api/tracker/stats")
def tracker_stats():
    return safe_json(get_stats())


@app.route("/api/tracker/open", methods=["POST"])
def tracker_open():
    body = request.get_json() or {}
    pair = body.get("pair", "").upper()
    if not pair or pair not in INSTRUMENTS:
        return jsonify({"error": "Invalid pair"}), 400
    sig = open_signal(
        pair       = pair,
        direction  = body.get("direction", "BUY"),
        entry      = float(body["entry"]),
        stop_loss  = float(body["stop_loss"]),
        take_profit= float(body["take_profit"]),
        grade      = body.get("grade", "B"),
        rr_ratio   = float(body.get("rr_ratio", 0)),
        conditions = body.get("conditions", []),
    )
    return safe_json(sig)


@app.route("/api/tracker/close/<sig_id>", methods=["POST"])
def tracker_close(sig_id):
    body       = request.get_json() or {}
    exit_price = float(body.get("exit_price", 0))
    result     = body.get("result", "MANUAL")
    sig = close_signal(sig_id, exit_price, result, note="Manual close")
    if not sig:
        return jsonify({"error": "Signal not found"}), 404
    return safe_json(sig)


@app.route("/api/tracker/pair/<pair>")
def tracker_pair(pair):
    sig = get_signal_for_pair(pair.upper())
    return safe_json({"signal": sig, "has_open": sig is not None})


# ── Helpers ────────────────────────────────────────────────────────────────────
def _safe_json(obj):
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_json(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def _now_iso():
    return datetime.now(timezone.utc).isoformat()

def _future_iso(hours: float):
    from datetime import timedelta
    return (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()


# ── Auto-scheduler: fetch + train every N hours ────────────────────────────────
_scheduler_thread: threading.Thread | None = None
_scheduler_stop   = threading.Event()

def _run_cycle():
    """One full fetch→train cycle for all instruments."""
    pairs = get_all_instruments()
    interval_h = _system_state["auto_refresh_interval_h"]
    print(f"\n[SCHEDULER] Starting fetch+train cycle  {_now_iso()}")

    with _state_lock:
        _system_state["last_fetch_at"]  = _now_iso()
        _system_state["next_fetch_at"]  = _future_iso(interval_h)

    for pair in pairs:
        from data_manager import fetch_from_deriv, load_csv
        key = f"{pair}_H1"
        set_fetch_status(key, {"status": "fetching"})
        df = fetch_from_deriv(pair, "H1", 20000)
        if not df.empty:
            old = load_csv(pair, "H1")
            if old is not None and not old.empty:
                import pandas as pd
                df = pd.concat([old, df]).drop_duplicates("datetime")
                df.sort_values("datetime", inplace=True)
                df.reset_index(drop=True, inplace=True)
            save_df(df, pair, "H1")
            set_fetch_status(key, {
                "status": "ok", "rows": len(df), "source": "deriv",
                "last": str(df["datetime"].iloc[-1]),
                "refreshed_at": _now_iso(),
            })
            print(f"  [{pair}] Fetched {len(df)} rows")

            # Train immediately after fetching this pair
            try:
                with _training_lock:
                    _training_status[pair] = {"status": "training"}
                _, metrics = train_model(df, pair)
                with _training_lock:
                    _training_status[pair] = {"status": "complete", "metrics": metrics}
                print(f"  [{pair}] Trained  acc={metrics['accuracy']:.3f}")
            except Exception as e:
                with _training_lock:
                    _training_status[pair] = {"status": "error", "error": str(e)}
                print(f"  [{pair}] Train error: {e}")
        else:
            set_fetch_status(key, {"status": "fetch_failed"})
            print(f"  [{pair}] Fetch failed — training on cached data")
            # Still retrain on cached data if model is stale
            try:
                cached = load_instrument_data(pair)
                if len(cached) >= 200:
                    with _training_lock:
                        _training_status[pair] = {"status": "training"}
                    _, metrics = train_model(cached, pair)
                    with _training_lock:
                        _training_status[pair] = {"status": "complete", "metrics": metrics}
            except Exception:
                pass

    # Also top up M15 data for each pair
    print(f"[SCHEDULER] Updating M15 data…")
    for pair in pairs:
        from data_manager import fetch_from_deriv, load_csv, save_df
        df15 = fetch_from_deriv(pair, "M15", 5000)
        if not df15.empty:
            old15 = load_csv(pair, "M15")
            if old15 is not None and not old15.empty:
                df15 = pd.concat([old15, df15]).drop_duplicates("datetime")
                df15.sort_values("datetime", inplace=True)
                df15.reset_index(drop=True, inplace=True)
            save_df(df15, pair, "M15")
            print(f"  [{pair}] M15: {len(df15)} rows")
        import time as _t; _t.sleep(0.3)

    with _state_lock:
        _system_state["last_train_at"] = _now_iso()
        _system_state["next_train_at"] = _future_iso(interval_h)

    print(f"[SCHEDULER] Cycle complete  {_now_iso()}\n")


def _scheduler_loop():
    with _state_lock:
        interval_h = _system_state["auto_refresh_interval_h"]
        _system_state["scheduler_running"] = True
        _system_state["next_fetch_at"]     = _future_iso(interval_h)

    while not _scheduler_stop.wait(interval_h * 3600):
        if _scheduler_stop.is_set():
            break
        _run_cycle()
        with _state_lock:
            interval_h = _system_state["auto_refresh_interval_h"]

    with _state_lock:
        _system_state["scheduler_running"] = False


def _restart_scheduler():
    global _scheduler_thread
    _scheduler_stop.set()
    if _scheduler_thread and _scheduler_thread.is_alive():
        _scheduler_thread.join(timeout=2)
    _scheduler_stop.clear()
    _scheduler_thread = threading.Thread(
        target=_scheduler_loop, daemon=True, name="AutoScheduler"
    )
    _scheduler_thread.start()


# ── M15 auto-refresh scheduler (every 15 min) ─────────────────────────────────
_m15_stop   = threading.Event()
_m15_thread = None

def _run_m15_cycle():
    """Fetch latest M15 bars + update entry analysis for all pairs."""
    pairs = get_all_instruments()
    print(f"[M15] Refreshing entry refinement for {len(pairs)} pairs  {_now_iso()}")
    with _state_lock:
        interval_m = _system_state["m15_refresh_interval_m"]
        _system_state["last_m15_at"] = _now_iso()
        from datetime import timedelta
        _system_state["next_m15_at"] = (
            datetime.now(timezone.utc) + timedelta(minutes=interval_m)
        ).isoformat()

    for pair in pairs:
        try:
            from entry_refiner import get_m15_data, analyse_m15
            from data_manager import fetch_from_deriv, load_csv, save_df

            # Fetch fresh M15 candles (only last 300 bars — quick)
            fresh = fetch_from_deriv(pair, "M15", 300)
            if not fresh.empty:
                old = load_csv(pair, "M15")
                if old is not None and not old.empty:
                    fresh = pd.concat([old.tail(2000), fresh]).drop_duplicates("datetime")
                    fresh.sort_values("datetime", inplace=True)
                    fresh.reset_index(drop=True, inplace=True)
                save_df(fresh, pair, "M15")

            # Run entry analysis for BUY and SELL
            df_m15 = get_m15_data(pair, 300)
            for direction in ("BUY", "SELL"):
                result = analyse_m15(df_m15, direction)
                with _m15_lock:
                    _m15_cache[f"{pair}_{direction}"] = result

            model = load_model(pair)
            if model:
                direction = _get_h1_direction(pair)
                if direction and direction != "HOLD":
                    analysis = _m15_cache.get(f"{pair}_{direction}")
                    if analysis:
                        print(f"  [{pair}] M15 grade={analysis.get('grade','?')}  "
                              f"R/R={analysis.get('rr_ratio','?')}  "
                              f"type={analysis.get('entry_type','?')}")
        except Exception as e:
            print(f"  [{pair}] M15 error: {e}")

    # ── Check SL/TP on all open signals ───────────────────────────────────────
    active_sigs = get_active_signals()
    closed_pairs = set()
    for sig in active_sigs:
        p = sig["pair"]
        try:
            from data_manager import load_csv
            df15 = load_csv(p, "M15")
            if df15 is not None and not df15.empty:
                last_bar = df15.iloc[-1]
                closed = check_signals_on_candle(
                    pair        = p,
                    candle_high = float(last_bar["high"]),
                    candle_low  = float(last_bar["low"]),
                    candle_close= float(last_bar["close"]),
                    candle_time = str(last_bar.get("datetime", "")),
                )
                if closed:
                    closed_pairs.add(p)
                    result = closed["result"]
                    pips   = closed.get("pnl_pips", 0)
                    print(f"  [Tracker] {p} → {result}  {pips:+.1f} pips")
        except Exception as e:
            print(f"  [Tracker] check error {p}: {e}")

    # ── Auto-open new signals for pairs that just closed or have no signal ─────
    from entry_refiner import generate_m15_signal, get_h1_bias
    for pair in pairs:
        try:
            existing = get_signal_for_pair(pair)
            if existing:
                continue  # already has open signal
            # Get fresh M15 signal
            with _m15_lock:
                cached = _m15_cache.get(f"{pair}_BUY") or _m15_cache.get(f"{pair}_SELL")
            if not cached or cached.get("grade") not in ("A", "B"):
                continue  # only auto-open grade A or B
            direction = "BUY" if _m15_cache.get(f"{pair}_BUY", {}).get("grade","W") in ("A","B") else "SELL"
            m = _m15_cache.get(f"{pair}_{direction}", {})
            if not m.get("m15_entry"):
                continue
            open_signal(
                pair        = pair,
                direction   = direction,
                entry       = m["m15_entry"],
                stop_loss   = m["m15_sl"],
                take_profit = m["m15_tp"],
                grade       = m.get("grade","B"),
                rr_ratio    = m.get("rr_ratio", 0),
                conditions  = m.get("conditions", []),
                current_price=m.get("m15_entry"),
            )
        except Exception as e:
            print(f"  [AutoOpen] {pair}: {e}")

    print(f"[M15] Done  {_now_iso()}")


def _get_h1_direction(pair: str) -> str | None:
    """Quick H1 signal direction without full re-run."""
    try:
        df = load_instrument_data(pair, periods=100)
        from ml_engine import generate_signal
        sig = generate_signal(df, pair, refine_with_m15=False)
        return sig.get("signal")
    except Exception:
        return None


def _m15_loop():
    with _state_lock:
        interval_m = _system_state["m15_refresh_interval_m"]
        _system_state["m15_scheduler_running"] = True

    # First run immediately on start (after a short delay)
    time.sleep(5)
    _run_m15_cycle()

    while not _m15_stop.wait(interval_m * 60):
        if _m15_stop.is_set():
            break
        _run_m15_cycle()
        with _state_lock:
            interval_m = _system_state["m15_refresh_interval_m"]

    with _state_lock:
        _system_state["m15_scheduler_running"] = False


def _start_m15_scheduler():
    global _m15_thread
    _m15_stop.set()
    if _m15_thread and _m15_thread.is_alive():
        _m15_thread.join(timeout=2)
    _m15_stop.clear()
    _m15_thread = threading.Thread(
        target=_m15_loop, daemon=True, name="M15Scheduler"
    )
    _m15_thread.start()
    print("[M15] Scheduler started — refreshing every 15 min")


def _boot_scheduler():
    """Called once on server start — trains any untrained models then starts loop."""
    def _boot():
        time.sleep(2)  # Give Flask a moment to start
        # Train all pairs that have no model yet, using cached or synthetic data
        pairs = get_all_instruments()
        needs_train = [p for p in pairs if load_model(p) is None]
        if needs_train:
            print(f"\n[BOOT] Training {len(needs_train)} untrained models on startup…")
            for pair in needs_train:
                try:
                    df = load_instrument_data(pair)
                    with _training_lock:
                        _training_status[pair] = {"status": "training"}
                    _, metrics = train_model(df, pair)
                    with _training_lock:
                        _training_status[pair] = {"status": "complete", "metrics": metrics}
                    print(f"  [{pair}] Boot-trained  acc={metrics['accuracy']:.3f}")
                except Exception as e:
                    with _training_lock:
                        _training_status[pair] = {"status": "error", "error": str(e)}
            print("[BOOT] Startup training complete\n")

        # Start the recurring H1 scheduler
        _restart_scheduler()
        # Start the M15 entry-refinement scheduler
        _start_m15_scheduler()

    threading.Thread(target=_boot, daemon=True, name="BootScheduler").start()


if __name__ == "__main__":
    print("=" * 60)
    print("  🚀 NEXUS Trading System v2")
    print("  Auto-train:  every 5 hours")
    print("  M15 refresh: every 15 minutes")
    print("  Open: http://localhost:5000")
    print("=" * 60)
    # Print all registered routes on startup
    with app.app_context():
        routes = sorted([str(r) for r in app.url_map.iter_rules()])
        for r in routes:
            print(f"  {r}")
    print("=" * 60)
    _boot_scheduler()
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True, use_reloader=False)