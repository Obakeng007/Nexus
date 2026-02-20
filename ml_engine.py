"""
ML Engine v2 — uses features.py for feature engineering & labeling
- Ensemble: RandomForest + GradientBoosting
- Labels come from create_labels() in features.py (ATR-directional, proven)
- Signals are purely from model predictions, not rule-based
- Model persistence with full metadata
"""
import os
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from features import add_features, create_labels, FEATURE_COLS

warnings.filterwarnings("ignore")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


# ── Training ───────────────────────────────────────────────────────────────────
def train_model(df: pd.DataFrame, pair: str,
                forward_bars: int = 5,
                atr_multiplier: float = 0.5) -> tuple[dict, dict]:
    """
    Train RF + GB ensemble on the supplied OHLCV dataframe.
    Returns (model_data dict, metrics dict).
    """
    print(f"  Training [{pair}]  rows={len(df)} …")

    # Feature engineering + labeling
    df_feat = add_features(df.copy())
    df_feat = create_labels(df_feat, forward_bars=forward_bars, atr_multiplier=atr_multiplier)

    # Feature matrix
    avail   = [c for c in FEATURE_COLS if c in df_feat.columns]
    X       = df_feat[avail].copy().replace([np.inf, -np.inf], np.nan)
    y       = df_feat["label"]

    # Drop rows where features or label are NaN (warm-up period)
    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]

    if len(X) < 200:
        raise ValueError(f"Not enough clean rows after feature engineering: {len(X)}")

    # Label distribution
    uniq, cnts = np.unique(y, return_counts=True)
    label_map  = {-1: "SELL", 0: "HOLD", 1: "BUY"}
    class_dist = {label_map.get(int(k), str(k)): int(v) for k, v in zip(uniq, cnts)}
    print(f"    Label distribution: {class_dist}")

    # Time-series split (80 / 20)
    split   = int(len(X) * 0.8)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    scaler      = RobustScaler()
    X_tr_s      = scaler.fit_transform(X_tr)
    X_te_s      = scaler.transform(X_te)

    # ── Random Forest ──
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=10,
        min_samples_split=20, max_features="sqrt",
        class_weight="balanced", n_jobs=-1, random_state=42,
    )
    rf.fit(X_tr_s, y_tr)

    # ── Gradient Boosting ──
    gb = GradientBoostingClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.04,
        subsample=0.8, min_samples_split=20, random_state=42,
    )
    gb.fit(X_tr_s, y_tr)

    # ── Ensemble predict on test set ──
    rf_proba = rf.predict_proba(X_te_s)
    gb_proba = gb.predict_proba(X_te_s)

    # Align class arrays (both models may have different class ordering)
    classes_rf = np.array(rf.classes_)
    classes_gb = np.array(gb.classes_)

    # Build combined probability matrix aligned to rf class order
    combined = rf_proba.copy()
    for i, cls in enumerate(classes_rf):
        if cls in classes_gb:
            j = list(classes_gb).index(cls)
            combined[:, i] = (rf_proba[:, i] + gb_proba[:, j]) / 2

    pred_idx    = np.argmax(combined, axis=1)
    preds       = classes_rf[pred_idx]
    accuracy    = accuracy_score(y_te, preds)

    # ── CV score (5-fold time-series) ──
    tscv     = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    for tr_i, val_i in tscv.split(X_tr_s):
        rf_cv = RandomForestClassifier(n_estimators=100, max_depth=6,
                                       class_weight="balanced", n_jobs=-1, random_state=42)
        rf_cv.fit(X_tr_s[tr_i], y_tr.iloc[tr_i])
        cv_scores.append(accuracy_score(y_tr.iloc[val_i], rf_cv.predict(X_tr_s[val_i])))

    # ── Feature importance ──
    importance   = pd.Series(rf.feature_importances_, index=avail)
    top_features = importance.nlargest(10).round(4).to_dict()

    metrics = {
        "accuracy":          round(float(accuracy), 4),
        "cv_mean":           round(float(np.mean(cv_scores)), 4),
        "cv_std":            round(float(np.std(cv_scores)), 4),
        "train_samples":     int(len(X_tr)),
        "test_samples":      int(len(X_te)),
        "class_distribution": class_dist,
        "top_features":      top_features,
        "forward_bars":      forward_bars,
        "atr_multiplier":    atr_multiplier,
        "features_used":     len(avail),
    }

    print(f"    Accuracy={accuracy:.3f}  CV={np.mean(cv_scores):.3f}±{np.std(cv_scores):.3f}")

    model_data = {
        "rf": rf, "gb": gb, "scaler": scaler,
        "feature_cols": avail,
        "classes_rf": classes_rf.tolist(),
        "classes_gb": classes_gb.tolist(),
        "metrics": metrics,
        "pair": pair,
        "trained_at": pd.Timestamp.now("UTC").isoformat(),
    }

    path = os.path.join(MODELS_DIR, f"{pair}_model.pkl")
    with open(path, "wb") as f:
        pickle.dump(model_data, f)

    return model_data, metrics


# ── Load model ─────────────────────────────────────────────────────────────────
def load_model(pair: str) -> dict | None:
    path = os.path.join(MODELS_DIR, f"{pair}_model.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Signal generation ──────────────────────────────────────────────────────────
def generate_signal(df: pd.DataFrame, pair: str,
                    refine_with_m15: bool = True) -> dict:
    """
    Generate a live signal for the latest bar using the trained model.
    Falls back to a rule-based signal only when no model exists.
    If refine_with_m15=True, runs M15 entry refinement on BUY/SELL signals.
    """
    model_data = load_model(pair)
    df_feat    = add_features(df.copy())

    if model_data is None:
        result = _rule_based_signal(df_feat, pair)
    else:
        result = _ml_signal(df_feat, model_data, pair)

    # ── M15 entry refinement ─────────────────────────────────────────────────
    if refine_with_m15 and result.get("signal") in ("BUY", "SELL"):
        try:
            from entry_refiner import get_m15_data, analyse_m15
            df_m15 = get_m15_data(pair, 300)
            m15    = analyse_m15(df_m15, result["signal"])
            result["m15"] = m15

            # If M15 gives a graded entry, override the entry/SL/TP
            if m15.get("available") and m15.get("grade") in ("A", "B", "C") and m15.get("m15_entry"):
                result["entry"]       = m15["m15_entry"]
                result["stop_loss"]   = m15["m15_sl"]
                result["take_profit"] = m15["m15_tp"]
                result["rr_ratio"]    = m15["rr_ratio"]
                result["entry_grade"] = m15["grade"]
                result["entry_type"]  = "M15 refined"
            else:
                result["entry_grade"] = "H1"
                result["entry_type"]  = "H1 only"
                if m15.get("grade") == "W":
                    result["entry_type"] = "Wait for M15 setup"
        except Exception as e:
            result["m15"] = {"available": False, "error": str(e)}
            result["entry_grade"] = "H1"
            result["entry_type"]  = "H1 only"
    else:
        result["entry_grade"] = "H1"
        result["entry_type"]  = "H1 only"
        result["m15"] = {"available": False, "direction": result.get("signal", "HOLD")}

    return result


def _ml_signal(df_feat: pd.DataFrame, model_data: dict, pair: str) -> dict:
    """Derive signal from ML ensemble on the latest bar."""
    rf          = model_data["rf"]
    gb          = model_data["gb"]
    scaler      = model_data["scaler"]
    feat_cols   = model_data["feature_cols"]
    classes_rf  = np.array(model_data["classes_rf"])
    classes_gb  = np.array(model_data["classes_gb"])

    X = df_feat[feat_cols].copy().replace([np.inf, -np.inf], np.nan)
    X = X.dropna()
    if X.empty:
        return _rule_based_signal(df_feat, pair)

    last       = X.iloc[[-1]]
    last_s     = scaler.transform(last)

    rf_proba   = rf.predict_proba(last_s)[0]
    gb_proba   = gb.predict_proba(last_s)[0]

    # Align to rf class order
    combined = rf_proba.copy()
    for i, cls in enumerate(classes_rf):
        if cls in classes_gb:
            j = list(classes_gb).index(cls)
            combined[i] = (rf_proba[i] + gb_proba[j]) / 2

    pred_idx    = int(np.argmax(combined))
    pred_class  = int(classes_rf[pred_idx])
    confidence  = float(combined[pred_idx]) * 100

    label_map   = {1: "BUY", -1: "SELL", 0: "HOLD"}
    signal_text = label_map.get(pred_class, "HOLD")

    # Probability breakdown
    proba_dict  = {}
    for i, cls in enumerate(classes_rf):
        proba_dict[label_map.get(int(cls), str(cls))] = round(float(combined[i]) * 100, 1)

    # Make sure BUY/SELL/HOLD keys always present
    for k in ["BUY", "HOLD", "SELL"]:
        proba_dict.setdefault(k, 0.0)

    return _build_output(df_feat, pair, signal_text, confidence, proba_dict, model_data)


def _rule_based_signal(df_feat: pd.DataFrame, pair: str) -> dict:
    """Simple indicator-consensus fallback when no model is trained."""
    latest  = df_feat.dropna(subset=["rsi_14"]).iloc[-1] if not df_feat.empty else None
    if latest is None:
        return _empty_signal(pair)

    score = 0
    reasons = []

    rsi = float(latest.get("rsi_14", 50))
    if rsi < 35:
        score += 2; reasons.append(f"RSI oversold ({rsi:.1f})")
    elif rsi > 65:
        score -= 2; reasons.append(f"RSI overbought ({rsi:.1f})")

    if float(latest.get("macd_hist", 0)) > 0:
        score += 1; reasons.append("MACD histogram positive")
    else:
        score -= 1; reasons.append("MACD histogram negative")

    if int(latest.get("ema_cross", 0)) == 1:
        score += 1; reasons.append("EMA 10 > EMA 20 (bullish)")
    else:
        score -= 1; reasons.append("EMA 10 < EMA 20 (bearish)")

    stoch_k = float(latest.get("stoch_k", 50))
    if stoch_k < 20:
        score += 1; reasons.append(f"Stoch oversold ({stoch_k:.1f})")
    elif stoch_k > 80:
        score -= 1; reasons.append(f"Stoch overbought ({stoch_k:.1f})")

    if score >= 2:
        signal, conf = "BUY",  min(45 + score * 8, 72)
    elif score <= -2:
        signal, conf = "SELL", min(45 + abs(score) * 8, 72)
    else:
        signal, conf = "HOLD", 38

    proba = {"BUY": 33, "HOLD": 34, "SELL": 33}

    result = _build_output(df_feat, pair, signal, conf, proba, None)
    result["note"] = "Rule-based fallback (model not trained)"
    return result


def _build_output(df_feat: pd.DataFrame, pair: str, signal: str,
                  confidence: float, proba: dict,
                  model_data: dict | None) -> dict:
    """Build the full signal output dict with SL/TP levels."""
    from data_manager import INSTRUMENTS

    latest       = df_feat.iloc[-1]
    current      = float(latest["close"])
    atr_val      = float(latest.get("atr_14", current * 0.001))
    cfg          = INSTRUMENTS.get(pair, {})
    spread       = cfg.get("spread", current * 0.0001)
    digits       = cfg.get("digits", 5)

    if signal == "BUY":
        entry = round(current + spread, digits)
        sl    = round(entry - 2.0 * atr_val, digits)
        tp    = round(entry + 3.0 * atr_val, digits)
    elif signal == "SELL":
        entry = round(current - spread, digits)
        sl    = round(entry + 2.0 * atr_val, digits)
        tp    = round(entry - 3.0 * atr_val, digits)
    else:
        entry = sl = tp = None

    # Support / Resistance
    hi20 = float(df_feat["high"].rolling(20).max().iloc[-1])
    lo20 = float(df_feat["low"].rolling(20).min().iloc[-1])

    # Reasons from indicator state
    reasons = []
    rsi_v   = float(latest.get("rsi_14", 50))
    reasons.append(f"RSI-14: {rsi_v:.1f}" + (" (oversold)" if rsi_v < 35 else " (overbought)" if rsi_v > 65 else ""))
    macd_h = float(latest.get("macd_hist", 0))
    reasons.append(f"MACD hist: {'↑ positive' if macd_h > 0 else '↓ negative'}")
    stk = float(latest.get("stoch_k", 50))
    reasons.append(f"Stoch %K: {stk:.1f}" + (" (OB)" if stk > 80 else " (OS)" if stk < 20 else ""))
    vr = int(latest.get("volatility_regime", 0))
    reasons.append("High volatility regime" if vr == 1 else "Low volatility regime")
    if int(latest.get("is_overlap", 0)):
        reasons.append("London/NY overlap (high liquidity)")
    elif int(latest.get("is_london", 0)):
        reasons.append("London session active")
    elif int(latest.get("is_newyork", 0)):
        reasons.append("New York session active")

    indicators = {
        "rsi":       round(rsi_v, 2),
        "rsi_7":     round(float(latest.get("rsi_7", 50)), 2),
        "macd_hist": round(float(latest.get("macd_hist", 0)), 5),
        "stoch_k":   round(stk, 2),
        "stoch_d":   round(float(latest.get("stoch_d", 50)), 2),
        "bb_pos":    round(float(latest.get("bb_pos", 0.5)), 3),
        "bb_squeeze": int(latest.get("bb_squeeze", 0)),
        "atr_ratio": round(float(latest.get("atr_ratio", 1)), 3),
        "ema_cross": int(latest.get("ema_cross", 0)),
        "trend_str": round(float(latest.get("trend_strength", 0)), 4),
        "vol_regime": int(latest.get("volatility_regime", 0)),
        "hl_pos":    round(float(latest.get("hl_position", 0.5)), 3),
    }

    return {
        "pair":          pair,
        "instrument":    pair,  # backward compat
        "signal":        signal,
        "confidence":    round(confidence, 1),
        "probabilities": proba,
        "current_price": round(current, digits),
        "entry":         round(entry, digits) if entry else None,
        "stop_loss":     round(sl, digits)    if sl    else None,
        "take_profit":   round(tp, digits)    if tp    else None,
        "rr_ratio":      1.5,
        "atr":           round(atr_val, digits),
        "reasons":       reasons[:5],
        "indicators":    indicators,
        "levels": {
            "resistance": round(hi20, digits),
            "support":    round(lo20, digits),
        },
        "timestamp":       str(df_feat.iloc[-1].get("datetime", "")) if "datetime" in df_feat.columns
                           else str(pd.Timestamp.now("UTC")),
        "model_accuracy":  round(model_data["metrics"]["accuracy"] * 100, 1) if model_data else None,
        "model_trained":   model_data is not None,
    }


def _empty_signal(pair: str) -> dict:
    return {
        "pair": pair, "instrument": pair,
        "signal": "HOLD", "confidence": 0,
        "probabilities": {"BUY": 33, "HOLD": 34, "SELL": 33},
        "current_price": 0, "entry": None, "stop_loss": None, "take_profit": None,
        "reasons": ["Insufficient data"], "indicators": {}, "levels": {},
        "model_accuracy": None, "model_trained": False, "timestamp": "",
    }


# ── Batch signals for all bars (used by backtester) ────────────────────────────
def generate_signals_series(df_feat: pd.DataFrame, model_data: dict,
                             confidence_threshold: float = 55.0) -> pd.Series:
    """Return a Series of predicted labels (-1,0,1) for every bar."""
    rf          = model_data["rf"]
    gb          = model_data["gb"]
    scaler      = model_data["scaler"]
    feat_cols   = model_data["feature_cols"]
    classes_rf  = np.array(model_data["classes_rf"])
    classes_gb  = np.array(model_data["classes_gb"])

    X = df_feat[feat_cols].copy().replace([np.inf, -np.inf], np.nan).fillna(0)
    X_s = scaler.transform(X)

    rf_proba = rf.predict_proba(X_s)
    gb_proba = gb.predict_proba(X_s)

    combined = rf_proba.copy()
    for i, cls in enumerate(classes_rf):
        if cls in classes_gb:
            j = list(classes_gb).index(cls)
            combined[:, i] = (rf_proba[:, i] + gb_proba[:, j]) / 2

    pred_idx    = np.argmax(combined, axis=1)
    preds       = classes_rf[pred_idx]
    confidences = combined[np.arange(len(combined)), pred_idx] * 100

    # Apply confidence threshold
    signals = np.where(confidences >= confidence_threshold, preds, 0)
    return pd.Series(signals.astype(int), index=df_feat.index)