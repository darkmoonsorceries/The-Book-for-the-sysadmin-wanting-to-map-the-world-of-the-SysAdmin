#!/usr/bin/env python3
"""
Fitrah Engine — Deriving mathematical equations from human nature patterns in crypto markets.

Pulls full 2026 price history for Base chain tokens available on Rainbow,
then fits a behavioral model derived from fitrah axioms to the data.

The model iterates until it finds parameters that map and infer price movements.
"""

import json
import time
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    print("Installing requests...")
    os.system(f"{sys.executable} -m pip install requests")
    import requests

try:
    import numpy as np
except ImportError:
    print("Installing numpy...")
    os.system(f"{sys.executable} -m pip install numpy")
    import numpy as np

try:
    from scipy.optimize import minimize, differential_evolution
    from scipy.signal import find_peaks
except ImportError:
    print("Installing scipy...")
    os.system(f"{sys.executable} -m pip install scipy")
    from scipy.optimize import minimize, differential_evolution
    from scipy.signal import find_peaks


# =============================================================================
# 1. DATA LAYER — Pull from CoinGecko (free, no key needed)
# =============================================================================

# Rainbow Base chain tokens mapped to CoinGecko IDs
TOKENS = {
    "ETH":    "ethereum",
    "UNI":    "uniswap",
    "MKR":    "maker",
    "AAVE":   "aave",
    "COMP":   "compound-governance-token",
    "CRV":    "curve-dao-token",
    "SNX":    "synthetix-network-token",
    "YFI":    "yearn-finance",
    "SUSHI":  "sushi",
    "ZRX":    "0x",
    "SHIB":   "shiba-inu",
    "1INCH":  "1inch",
    "PENDLE": "pendle",
    "BAL":    "balancer",
    "CTSI":   "cartesi",
    "DAI":    "dai",
    "DOLA":   "dola-usd",
    "LUSD":   "liquity-usd",
}

DATA_DIR = Path(__file__).parent / "price_data"
DATA_DIR.mkdir(exist_ok=True)


def fetch_price_history(coin_id: str, vs_currency: str = "usd") -> dict | None:
    """
    Fetch daily OHLC + volume from CoinGecko for 2026 (Jan 1 to now).
    Uses the free /coins/{id}/market_chart/range endpoint.
    """
    cache_file = DATA_DIR / f"{coin_id}_2026.json"

    # Use cache if less than 1 hour old
    if cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < 3600:
            with open(cache_file) as f:
                return json.load(f)

    # Jan 1, 2026 00:00 UTC
    from_ts = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp())
    to_ts = int(datetime.now(timezone.utc).timestamp())

    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range"
    params = {
        "vs_currency": vs_currency,
        "from": from_ts,
        "to": to_ts,
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 429:
            print(f"  Rate limited on {coin_id}, waiting 60s...")
            time.sleep(60)
            resp = requests.get(url, params=params, timeout=30)

        if resp.status_code != 200:
            print(f"  Failed to fetch {coin_id}: HTTP {resp.status_code}")
            return None

        data = resp.json()

        # Cache it
        with open(cache_file, "w") as f:
            json.dump(data, f)

        return data

    except Exception as e:
        print(f"  Error fetching {coin_id}: {e}")
        return None


def parse_to_arrays(data: dict) -> dict:
    """Convert CoinGecko response to numpy arrays."""
    prices = np.array(data["prices"])        # [[timestamp_ms, price], ...]
    volumes = np.array(data["total_volumes"])

    timestamps = prices[:, 0] / 1000  # ms -> seconds
    close = prices[:, 1]
    vol = volumes[:, 1]

    # Compute returns
    returns = np.diff(close) / close[:-1]

    return {
        "timestamps": timestamps,
        "close": close,
        "volume": vol,
        "returns": returns,
        "dates": [datetime.fromtimestamp(t, tz=timezone.utc) for t in timestamps],
    }


def load_local_data() -> dict:
    """
    Load pre-generated synthetic or cached data from price_data/ directory.
    Falls back to this when API is unavailable.
    """
    all_data = {}
    for json_file in DATA_DIR.glob("*_2026.json"):
        ticker = json_file.stem.replace("_2026", "").upper()
        try:
            with open(json_file) as f:
                raw = json.load(f)
            if "prices" in raw and len(raw["prices"]) > 10:
                all_data[ticker] = parse_to_arrays(raw)
                print(f"  [local] {ticker}: {len(all_data[ticker]['close'])} data points")
        except Exception as e:
            print(f"  [local] Failed to load {json_file}: {e}")
    return all_data


def fetch_all_tokens() -> dict:
    """
    Fetch price data for all tokens.
    Strategy: try local data first, then API if needed.
    """
    # First, try loading local/synthetic data
    print("Checking for local data...")
    all_data = load_local_data()

    if all_data:
        print(f"Loaded {len(all_data)} tokens from local data.")
        return all_data

    # Fall back to CoinGecko API
    print("No local data found. Fetching from CoinGecko API...")
    total = len(TOKENS)

    for i, (ticker, coin_id) in enumerate(TOKENS.items()):
        print(f"[{i+1}/{total}] Fetching {ticker} ({coin_id})...")
        raw = fetch_price_history(coin_id)
        if raw and "prices" in raw and len(raw["prices"]) > 10:
            all_data[ticker] = parse_to_arrays(raw)
            print(f"  Got {len(all_data[ticker]['close'])} data points")
        else:
            print(f"  Skipped {ticker} — insufficient data")

        # CoinGecko free tier: ~10-30 calls/min
        if i < total - 1:
            time.sleep(4)

    return all_data


# =============================================================================
# 2. FITRAH FEATURE EXTRACTION — Turn price data into behavioral signals
# =============================================================================

def compute_fitrah_features(close: np.ndarray, volume: np.ndarray, returns: np.ndarray) -> dict:
    """
    Extract features corresponding to each fitrah axiom.
    These are the 'X' variables our model will use.
    """
    n = len(returns)
    features = {}

    # --- Axiom 1: Hifz al-Mal (Loss Aversion Asymmetry) ---
    # Measure: ratio of downside speed to upside speed
    # Humans sell faster than they buy
    window = 14
    hifz = np.zeros(n)
    for i in range(window, n):
        w = returns[i-window:i]
        up_moves = w[w > 0]
        down_moves = w[w < 0]
        avg_up = np.mean(up_moves) if len(up_moves) > 0 else 1e-10
        avg_down = np.mean(np.abs(down_moves)) if len(down_moves) > 0 else 1e-10
        hifz[i] = avg_down / avg_up  # >1 means drops faster than rises
    features["hifz_al_mal"] = hifz

    # --- Axiom 2: Mizan (Balance / Mean Reversion) ---
    # Measure: deviation from EMA equilibrium
    ema_period = 50
    ema = _ema(close[:-1], ema_period)  # align with returns length
    if len(ema) > len(returns):
        ema = ema[:len(returns)]
    elif len(ema) < len(returns):
        ema = np.pad(ema, (len(returns) - len(ema), 0), mode='edge')
    mizan = (close[1:len(returns)+1] - ema) / ema
    features["mizan"] = mizan

    # --- Axiom 3: Israf (Excess Detection) ---
    # Measure: current move relative to ATR (average true range proxy)
    atr_period = 20
    israf = np.zeros(n)
    for i in range(atr_period, n):
        atr = np.mean(np.abs(returns[i-atr_period:i])) * close[i]
        if atr > 0:
            current_move = abs(close[i+1] - close[i]) if i+1 < len(close) else 0
            israf[i] = current_move / atr
    features["israf"] = israf

    # --- Axiom 4: Gharar (Uncertainty / Volatility) ---
    # Measure: rolling volatility percentile
    vol_window = 20
    vol_lookback = 100
    gharar = np.zeros(n)
    rolling_vol = np.zeros(n)
    for i in range(vol_window, n):
        rolling_vol[i] = np.std(returns[i-vol_window:i])
    for i in range(vol_lookback, n):
        past_vols = rolling_vol[i-vol_lookback:i]
        if np.max(past_vols) > 0:
            gharar[i] = np.sum(past_vols < rolling_vol[i]) / vol_lookback
    features["gharar"] = gharar

    # --- Axiom 5: Shukr/Qana'ah (Contentment / Profit-Taking) ---
    # Measure: how far price has risen from recent low (greed indicator)
    lookback = 20
    shukr = np.zeros(n)
    for i in range(lookback, n):
        recent_low = np.min(close[i-lookback:i])
        if recent_low > 0:
            shukr[i] = (close[i+1] - recent_low) / recent_low if i+1 < len(close) else 0
    features["shukr"] = shukr

    # --- Axiom 7: Sabr (Patience / Consolidation Duration) ---
    # Measure: how long price has been in a tight range (patience zone duration)
    sabr_threshold = 0.02  # 2% range = consolidation
    sabr = np.zeros(n)
    for i in range(10, n):
        # Look back and count how many bars the range has been < threshold
        duration = 0
        for j in range(i, max(i-100, 0), -1):
            window_prices = close[j:i+2]
            if len(window_prices) < 2:
                break
            pct_range = (np.max(window_prices) - np.min(window_prices)) / np.min(window_prices)
            if pct_range < sabr_threshold:
                duration += 1
            else:
                break
        sabr[i] = duration
    features["sabr"] = sabr

    # --- Axiom 8: Tawakkul vs Tadbir (Volume conviction) ---
    # Measure: volume relative to average when price is stable vs moving
    vol_ema = _ema(volume[:-1], 50)
    if len(vol_ema) > n:
        vol_ema = vol_ema[:n]
    elif len(vol_ema) < n:
        vol_ema = np.pad(vol_ema, (n - len(vol_ema), 0), mode='edge')
    tawakkul = np.zeros(n)
    for i in range(1, n):
        if vol_ema[i] > 0:
            vol_ratio = volume[i] / vol_ema[i]
            price_move = abs(returns[i])
            if price_move < 0.01:  # stable price
                tawakkul[i] = vol_ratio  # high vol + stable = tawakkul (holding)
            else:
                tawakkul[i] = -vol_ratio  # high vol + movement = tadbir (exiting)
    features["tawakkul_tadbir"] = tawakkul

    return features


def _ema(data: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average."""
    alpha = 2 / (period + 1)
    ema = np.zeros(len(data))
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
    return ema


# =============================================================================
# 3. FITRAH MODEL — Mathematical equation to fit
# =============================================================================

def fitrah_model(features: dict, params: np.ndarray) -> np.ndarray:
    """
    The core equation: predicted next return as a weighted combination
    of fitrah behavioral signals.

    predicted_return[t] = sum(w_i * f_i(features[t])) + bias

    Where f_i are nonlinear transformations of each axiom's signal,
    parameterized to capture the behavioral dynamics.

    Parameters (16 total):
        [0]  w_hifz       - weight for loss aversion signal
        [1]  w_mizan      - weight for mean reversion signal
        [2]  w_israf      - weight for excess signal
        [3]  w_gharar     - weight for uncertainty signal
        [4]  w_shukr      - weight for profit-taking signal
        [5]  w_sabr       - weight for patience signal
        [6]  w_tawakkul   - weight for conviction signal
        [7]  bias         - constant offset
        [8]  hifz_thresh  - threshold for loss aversion activation
        [9]  mizan_decay  - how quickly mizan reversion acts
        [10] israf_clip   - excess level beyond which signal saturates
        [11] gharar_exit  - volatility percentile for full exit
        [12] shukr_target - profit level triggering contentment
        [13] sabr_min     - minimum patience duration to matter
        [14] momentum     - trend-following component weight
        [15] interaction  - mizan * sabr interaction term weight
    """
    n = len(features["mizan"])
    signal = np.full(n, params[7])  # start with bias

    # Axiom 1: Hifz al-Mal — activated when asymmetry exceeds threshold
    hifz_active = np.where(
        features["hifz_al_mal"] > params[8],
        -params[0] * (features["hifz_al_mal"] - params[8]),  # contrarian: sell panic
        0
    )
    signal += hifz_active

    # Axiom 2: Mizan — mean reversion with decay rate
    mizan_signal = -params[1] * np.tanh(features["mizan"] * params[9])
    signal += mizan_signal

    # Axiom 3: Israf — excess clipped at saturation
    israf_clipped = np.clip(features["israf"], 0, params[10])
    israf_signal = -params[2] * israf_clipped / max(params[10], 0.01)
    signal += israf_signal

    # Axiom 4: Gharar — reduce signal magnitude in high uncertainty
    gharar_damper = np.where(
        features["gharar"] > params[11],
        1 - (features["gharar"] - params[11]) / (1 - params[11] + 1e-10),
        1.0
    )
    gharar_damper = np.clip(gharar_damper, 0, 1)
    signal *= gharar_damper
    # Also add direct gharar aversion
    signal -= params[3] * np.maximum(features["gharar"] - params[11], 0)

    # Axiom 5: Shukr — take profits when gains exceed contentment threshold
    shukr_signal = np.where(
        features["shukr"] > params[12],
        -params[4] * (features["shukr"] - params[12]),
        0
    )
    signal += shukr_signal

    # Axiom 7: Sabr — patience zones amplify signals
    sabr_active = np.where(features["sabr"] > params[13], features["sabr"] / 100, 0)
    signal += params[5] * sabr_active

    # Axiom 8: Tawakkul/Tadbir — conviction direction
    signal += params[6] * np.tanh(features["tawakkul_tadbir"])

    # Momentum component (simple trend)
    signal += params[14] * features["mizan"]

    # Interaction: mizan * sabr (breakout from patience at extreme deviation)
    signal += params[15] * features["mizan"] * sabr_active

    return signal


# =============================================================================
# 4. FITTING ENGINE — Iterate until the equation maps the data
# =============================================================================

def compute_loss(params: np.ndarray, features: dict, actual_returns: np.ndarray,
                 start_idx: int = 100) -> float:
    """
    Loss function: how well does our fitrah model predict actual returns?

    Uses a combination of:
    - Direction accuracy (did we predict the right sign?)
    - Magnitude correlation
    - Sharpe ratio of theoretical trades
    """
    predicted = fitrah_model(features, params)

    # Only evaluate after warmup period
    pred = predicted[start_idx:]
    actual = actual_returns[start_idx:]

    # 1. Directional accuracy loss (most important — did we get the direction right?)
    direction_match = np.sign(pred) == np.sign(actual)
    direction_loss = -np.mean(direction_match)  # negative because we minimize

    # 2. Correlation loss
    if np.std(pred) > 1e-10 and np.std(actual) > 1e-10:
        corr = np.corrcoef(pred, actual)[0, 1]
        if np.isnan(corr):
            corr = 0
    else:
        corr = 0
    correlation_loss = -corr

    # 3. Sharpe ratio of theoretical trades (signal * actual return)
    pnl = pred * actual
    if np.std(pnl) > 1e-10:
        sharpe = np.mean(pnl) / np.std(pnl) * np.sqrt(365)
    else:
        sharpe = 0
    sharpe_loss = -sharpe

    # 4. Penalize extreme parameters (regularization)
    reg = 0.001 * np.sum(params**2)

    # Combined loss
    total_loss = 0.4 * direction_loss + 0.3 * correlation_loss + 0.3 * sharpe_loss + reg

    return total_loss


def fit_token(ticker: str, token_data: dict, max_iterations: int = 200) -> dict:
    """
    Fit the fitrah model to a single token's data.
    Uses differential evolution (global optimizer) to find the best parameters.
    """
    close = token_data["close"]
    volume = token_data["volume"]
    returns = token_data["returns"]

    if len(returns) < 120:
        print(f"  {ticker}: Not enough data ({len(returns)} points), skipping")
        return None

    print(f"  Computing fitrah features for {ticker}...")
    features = compute_fitrah_features(close, volume, returns)

    # Parameter bounds: [lower, upper] for each of 16 params
    bounds = [
        (-2, 2),      # w_hifz
        (-2, 2),      # w_mizan
        (-2, 2),      # w_israf
        (-2, 2),      # w_gharar
        (-2, 2),      # w_shukr
        (-2, 2),      # w_sabr
        (-2, 2),      # w_tawakkul
        (-0.01, 0.01),# bias
        (0.5, 3.0),   # hifz_thresh
        (0.1, 10.0),  # mizan_decay
        (0.5, 5.0),   # israf_clip
        (0.5, 0.95),  # gharar_exit
        (0.02, 0.20), # shukr_target
        (5, 50),      # sabr_min
        (-1, 1),      # momentum
        (-2, 2),      # interaction
    ]

    print(f"  Fitting model ({max_iterations} iterations)...")

    result = differential_evolution(
        compute_loss,
        bounds=bounds,
        args=(features, returns),
        maxiter=max_iterations,
        seed=42,
        tol=1e-8,
        disp=False,
        polish=True,
        workers=1,
    )

    # Evaluate the fit
    predicted = fitrah_model(features, result.x)
    start = 100

    pred = predicted[start:]
    actual = returns[start:]

    direction_accuracy = np.mean(np.sign(pred) == np.sign(actual))

    if np.std(pred) > 1e-10 and np.std(actual) > 1e-10:
        correlation = np.corrcoef(pred, actual)[0, 1]
    else:
        correlation = 0

    pnl = pred * actual
    sharpe = np.mean(pnl) / np.std(pnl) * np.sqrt(365) if np.std(pnl) > 1e-10 else 0

    # Current signal (last data point)
    current_signal = predicted[-1]

    return {
        "ticker": ticker,
        "params": result.x.tolist(),
        "loss": result.fun,
        "direction_accuracy": direction_accuracy,
        "correlation": correlation,
        "sharpe": sharpe,
        "current_signal": current_signal,
        "data_points": len(returns),
        "param_names": [
            "w_hifz", "w_mizan", "w_israf", "w_gharar", "w_shukr",
            "w_sabr", "w_tawakkul", "bias", "hifz_thresh", "mizan_decay",
            "israf_clip", "gharar_exit", "shukr_target", "sabr_min",
            "momentum", "interaction"
        ],
    }


# =============================================================================
# 5. CROSS-TOKEN UNIVERSALS — Find fitrah constants across all tokens
# =============================================================================

def find_universal_fitrah(all_results: dict) -> dict:
    """
    After fitting each token individually, look for parameters that are
    consistent across all tokens. These are the 'fitrah constants' —
    behavioral invariants that hold regardless of which token humans trade.
    """
    param_names = [
        "w_hifz", "w_mizan", "w_israf", "w_gharar", "w_shukr",
        "w_sabr", "w_tawakkul", "bias", "hifz_thresh", "mizan_decay",
        "israf_clip", "gharar_exit", "shukr_target", "sabr_min",
        "momentum", "interaction"
    ]

    all_params = np.array([r["params"] for r in all_results.values()])

    universals = {}
    for i, name in enumerate(param_names):
        values = all_params[:, i]
        mean = np.mean(values)
        std = np.std(values)
        cv = abs(std / mean) if abs(mean) > 1e-10 else float('inf')

        universals[name] = {
            "mean": float(mean),
            "std": float(std),
            "cv": float(cv),  # coefficient of variation — lower = more universal
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "is_fitrah_constant": cv < 0.5,  # consistent across tokens
        }

    return universals


# =============================================================================
# 6. OUTPUT & REPORTING
# =============================================================================

def print_results(all_results: dict, universals: dict):
    """Print a human-readable summary."""
    print("\n" + "=" * 80)
    print("FITRAH ENGINE — RESULTS")
    print("=" * 80)

    # Per-token results
    print("\n--- PER-TOKEN MODEL FIT ---\n")
    print(f"{'Token':<8} {'Dir Acc':>8} {'Corr':>8} {'Sharpe':>8} {'Signal':>10} {'Meaning':<20}")
    print("-" * 72)

    for ticker, r in sorted(all_results.items(), key=lambda x: x[1]["sharpe"], reverse=True):
        sig = r["current_signal"]
        if sig > 0.005:
            meaning = "BUY (fitrah)"
        elif sig < -0.005:
            meaning = "SELL (fitrah)"
        else:
            meaning = "HOLD (mizan)"

        print(f"{ticker:<8} {r['direction_accuracy']:>7.1%} {r['correlation']:>8.4f} "
              f"{r['sharpe']:>8.2f} {sig:>10.6f} {meaning:<20}")

    # Universal fitrah constants
    print("\n--- UNIVERSAL FITRAH CONSTANTS ---")
    print("(Parameters consistent across all tokens = innate human behavior)\n")

    print(f"{'Parameter':<20} {'Mean':>10} {'Std':>10} {'CV':>8} {'Universal?':<12}")
    print("-" * 62)

    for name, u in sorted(universals.items(), key=lambda x: x[1]["cv"]):
        is_u = "FITRAH" if u["is_fitrah_constant"] else "varies"
        print(f"{name:<20} {u['mean']:>10.4f} {u['std']:>10.4f} {u['cv']:>8.3f} {is_u:<12}")

    # The equation
    print("\n--- THE FITRAH EQUATION ---\n")
    print("predicted_return[t] =")

    fitrah_params = {k: v for k, v in universals.items() if v["is_fitrah_constant"]}
    if fitrah_params:
        for name, u in fitrah_params.items():
            print(f"    + {u['mean']:+.4f} * {name}(t)")
    else:
        print("    (No universally consistent parameters found yet — need more data)")

    print()


def save_results(all_results: dict, universals: dict):
    """Save results to JSON for further analysis."""
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tokens_analyzed": len(all_results),
        "per_token": {k: {kk: vv for kk, vv in v.items()} for k, v in all_results.items()},
        "universal_fitrah_constants": universals,
    }

    out_path = Path(__file__).parent / "fitrah_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("FITRAH ENGINE")
    print("Deriving mathematical patterns of human nature from crypto price data")
    print("Tokens: Rainbow Base chain list | Period: Jan 1 2026 — present")
    print("=" * 80)

    # Step 1: Fetch data
    print("\n[STEP 1] Fetching 2026 price history...\n")
    all_data = fetch_all_tokens()

    if not all_data:
        print("ERROR: No data fetched. Check internet connection.")
        return

    print(f"\nFetched data for {len(all_data)} tokens.\n")

    # Step 2: Fit model to each token
    print("[STEP 2] Fitting fitrah model to each token...\n")
    all_results = {}
    for ticker, data in all_data.items():
        print(f"\n--- {ticker} ---")
        result = fit_token(ticker, data)
        if result:
            all_results[ticker] = result
            print(f"  Direction accuracy: {result['direction_accuracy']:.1%}")
            print(f"  Correlation:        {result['correlation']:.4f}")
            print(f"  Sharpe:             {result['sharpe']:.2f}")
            print(f"  Current signal:     {result['current_signal']:.6f}")

    if len(all_results) < 3:
        print("Not enough tokens fitted to find universals.")
        return

    # Step 3: Find universal fitrah constants
    print("\n[STEP 3] Identifying universal fitrah constants across all tokens...\n")
    universals = find_universal_fitrah(all_results)

    # Step 4: Report
    print_results(all_results, universals)
    save_results(all_results, universals)


if __name__ == "__main__":
    main()
