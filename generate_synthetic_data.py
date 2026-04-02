#!/usr/bin/env python3
"""
Synthetic Data Generator — Reconstructs price patterns from observed screenshots.

Creates realistic price data based on the actual patterns seen in:
- DOTA (Defense of the Agents): spike-and-decay, staircase down, floor consolidation
- CLANKER (tokenbot): staircase decline from $26.40, three sabr zones, bounce at $24.11
- Plus generates behavioral archetypes: pump-dump, slow bleed, accumulation, breakout

These serve as ground truth for fitting and testing the fitrah model
until live API access is available.
"""

import json
import os
import sys

try:
    import numpy as np
except ImportError:
    os.system(f"{sys.executable} -m pip install numpy")
    import numpy as np

from pathlib import Path

DATA_DIR = Path(__file__).parent / "price_data"
DATA_DIR.mkdir(exist_ok=True)

np.random.seed(42)  # Reproducible


# =============================================================================
# PATTERN GENERATORS — Each encodes a fitrah behavior
# =============================================================================

def generate_noise(n: int, scale: float = 0.005) -> np.ndarray:
    """Market microstructure noise."""
    return np.random.normal(0, scale, n)


def generate_staircase_decline(n: int, start_price: float, steps: list,
                                sabr_durations: list) -> np.ndarray:
    """
    Axiom 7 (Sabr) pattern: price declines in discrete steps with
    consolidation (patience) at each level.

    steps: list of price levels to step down to
    sabr_durations: how many bars of patience at each level
    """
    prices = []
    current = start_price

    for i, (level, duration) in enumerate(zip(steps, sabr_durations)):
        # Sharp drop to new level (Axiom 1: hifz al-mal, fast drops)
        drop_bars = max(3, int(duration * 0.1))
        drop = np.linspace(current, level, drop_bars)
        drop += generate_noise(drop_bars, scale=abs(current - level) * 0.05)
        prices.extend(drop)

        # Consolidation at level (Axiom 7: sabr zone)
        sabr = np.full(duration, level) + generate_noise(duration, scale=level * 0.008)
        prices.extend(sabr)
        current = level

    # Pad or trim to desired length
    prices = np.array(prices[:n] if len(prices) >= n else
                      list(prices) + list(np.full(n - len(prices), prices[-1]) +
                                          generate_noise(n - len(prices), scale=current * 0.005)))
    return np.maximum(prices, start_price * 0.01)  # floor at 1% of start


def generate_spike_and_decay(n: int, base_price: float, spike_magnitude: float,
                              decay_rate: float = 0.97) -> np.ndarray:
    """
    Axiom 3 (Israf) pattern: excess spike followed by exponential decay
    back to equilibrium. Bubbles pop.
    """
    prices = np.zeros(n)
    prices[0] = base_price

    # Build up phase (slow, Axiom 7: sabr building)
    buildup = int(n * 0.15)
    for i in range(1, buildup):
        prices[i] = prices[i-1] * (1 + np.random.normal(0.002, 0.01))

    # Spike (israf / excess)
    spike_bars = int(n * 0.05)
    spike_peak = prices[buildup-1] * (1 + spike_magnitude)
    for i in range(buildup, buildup + spike_bars):
        t = (i - buildup) / spike_bars
        prices[i] = prices[buildup-1] + (spike_peak - prices[buildup-1]) * t
        prices[i] += generate_noise(1, scale=prices[i] * 0.01)[0]

    # Decay back to equilibrium (mizan restoration)
    current = spike_peak
    for i in range(buildup + spike_bars, n):
        current = base_price + (current - base_price) * decay_rate
        current += generate_noise(1, scale=current * 0.008)[0]
        prices[i] = current

    return np.maximum(prices, base_price * 0.1)


def generate_mean_reversion(n: int, equilibrium: float, volatility: float = 0.02,
                             reversion_speed: float = 0.05) -> np.ndarray:
    """
    Axiom 2 (Mizan) pattern: Ornstein-Uhlenbeck process.
    Price oscillates around equilibrium — balance always restored.
    """
    prices = np.zeros(n)
    prices[0] = equilibrium * (1 + np.random.normal(0, 0.05))

    for i in range(1, n):
        drift = reversion_speed * (equilibrium - prices[i-1])
        shock = np.random.normal(0, volatility * prices[i-1])
        prices[i] = prices[i-1] + drift + shock

    return np.maximum(prices, equilibrium * 0.1)


def generate_panic_sell(n: int, start_price: float, panic_point: float = 0.6,
                         recovery: float = 0.3) -> np.ndarray:
    """
    Axiom 1 (Hifz al-Mal) pattern: gradual rise, then sudden panic selling
    (loss aversion cascade). Partial recovery as tawakkul holders buy.
    """
    prices = np.zeros(n)
    prices[0] = start_price

    # Gradual rise
    rise_end = int(n * panic_point)
    for i in range(1, rise_end):
        prices[i] = prices[i-1] * (1 + np.random.normal(0.001, 0.008))

    peak = prices[rise_end - 1]

    # Panic drop (fast — asymmetric, Axiom 1)
    panic_bars = int(n * 0.08)
    drop_target = peak * 0.7
    for i in range(rise_end, min(rise_end + panic_bars, n)):
        t = (i - rise_end) / panic_bars
        prices[i] = peak - (peak - drop_target) * (t ** 0.5)  # concave = fast then slow
        prices[i] += generate_noise(1, scale=prices[i] * 0.015)[0]

    # Partial recovery (Axiom 8: tawakkul holders step in)
    recovery_start = rise_end + panic_bars
    recovery_target = drop_target + (peak - drop_target) * recovery
    current = drop_target
    for i in range(recovery_start, n):
        current += (recovery_target - current) * 0.02
        current += generate_noise(1, scale=current * 0.01)[0]
        prices[i] = current

    return np.maximum(prices, start_price * 0.01)


def generate_gharar_chaos(n: int, start_price: float) -> np.ndarray:
    """
    Axiom 4 (Gharar) pattern: high uncertainty regime.
    Wild swings, no clear direction — humans withdraw.
    Volume should drop (simulated separately).
    """
    prices = np.zeros(n)
    prices[0] = start_price

    # Start stable, then enter chaos, then resolve
    stable1 = int(n * 0.3)
    chaos_start = stable1
    chaos_end = int(n * 0.7)

    for i in range(1, n):
        if i < chaos_start:
            vol = 0.01
        elif i < chaos_end:
            vol = 0.04  # 4x normal volatility = gharar
        else:
            vol = 0.01  # clarity returns

        prices[i] = prices[i-1] * (1 + np.random.normal(0, vol))

    return np.maximum(prices, start_price * 0.1)


def generate_volume(n: int, price_changes: np.ndarray, base_volume: float) -> np.ndarray:
    """
    Generate volume that reflects fitrah behavior:
    - Volume spikes on big moves (fear/greed)
    - Volume drops during consolidation (sabr)
    - Volume drops during gharar (withdrawal)
    """
    volume = np.full(n, base_volume)

    for i in range(1, n):
        abs_change = abs(price_changes[i-1]) if i-1 < len(price_changes) else 0
        # Volume proportional to price movement intensity
        vol_multiplier = 1 + abs_change * 50
        # Add noise
        vol_multiplier *= np.random.lognormal(0, 0.3)
        volume[i] = base_volume * vol_multiplier

    return volume


# =============================================================================
# RECONSTRUCT OBSERVED TOKENS
# =============================================================================

def generate_dota() -> dict:
    """
    DOTA (Defense of the Agents) — from screenshots:
    - 1H: spike from ~$0.0₅290 to ~$0.0₅820, decay back to ~$0.0₅428
    - 15M: mean-reverting around $0.0₅490 with range $0.0₅383-$0.0₅705
    - 5M: sharp dump then consolidation at bottom
    - 1M: grinding lower near support at $0.0₅428

    Full pattern: pump → slow bleed → staircase down → floor
    """
    n = 2000  # ~2000 hourly candles ≈ 83 days (Jan-Apr 2026)
    base = 0.00000428

    # Phase 1: Initial accumulation (days 1-20)
    p1 = generate_mean_reversion(500, base * 0.7, volatility=0.015, reversion_speed=0.02)

    # Phase 2: Spike (the pump visible on 1H chart)
    p2 = generate_spike_and_decay(400, base * 0.7, spike_magnitude=1.8, decay_rate=0.985)

    # Phase 3: Staircase decline (visible on 5M, 15M)
    p3 = generate_staircase_decline(600, p2[-1],
        steps=[base * 1.3, base * 1.1, base * 0.95, base],
        sabr_durations=[120, 100, 80, 100])

    # Phase 4: Floor consolidation (current state)
    p4 = generate_mean_reversion(500, base, volatility=0.008, reversion_speed=0.08)

    close = np.concatenate([p1, p2, p3, p4])[:n]
    returns = np.diff(close) / close[:-1]
    volume = generate_volume(n, returns, base_volume=50000)

    # Timestamps: hourly from Jan 1 2026
    start_ts = 1767225600
    timestamps = np.array([start_ts + i * 3600 for i in range(n)], dtype=float)

    return {
        "ticker": "DOTA",
        "close": close,
        "volume": volume,
        "returns": returns,
        "timestamps": timestamps,
    }


def generate_clanker() -> dict:
    """
    CLANKER (tokenbot) — from screenshots:
    - 4H: downtrend from ~$30 to $24.60
    - 1H: pump from ~$23 to $26.40, rounded top, declining back
    - 15M/5M: three-step staircase: $26.40 → $25.20 → $24.80 → $24.11
    - 1M: bouncing at $24.11 floor, current $24.60

    Historical context: tried to be acquired by Rainbow, spiked 70% to $48
    """
    n = 2000
    base = 24.60

    # Phase 1: Post-acquisition-news spike and settle
    p1 = generate_spike_and_decay(500, 28.0, spike_magnitude=0.7, decay_rate=0.99)

    # Phase 2: Slow bleed from ~$30 area
    p2 = generate_staircase_decline(600, 30.0,
        steps=[28.0, 26.5, 25.5],
        sabr_durations=[150, 120, 100])

    # Phase 3: Recent staircase (what we saw in screenshots)
    p3 = generate_staircase_decline(500, 26.40,
        steps=[25.60, 24.80, 24.11],
        sabr_durations=[120, 100, 80])

    # Phase 4: Current bounce
    p4 = generate_mean_reversion(400, base, volatility=0.008, reversion_speed=0.06)

    close = np.concatenate([p1, p2, p3, p4])[:n]
    returns = np.diff(close) / close[:-1]
    volume = generate_volume(n, returns, base_volume=500000)

    start_ts = 1767225600
    timestamps = np.array([start_ts + i * 3600 for i in range(n)], dtype=float)

    return {
        "ticker": "CLANKER",
        "close": close,
        "volume": volume,
        "returns": returns,
        "timestamps": timestamps,
    }


def generate_archetypal_tokens() -> list:
    """
    Generate tokens representing pure fitrah archetypes to test each axiom.
    """
    n = 2000
    tokens = []

    # 1. Pure mean-reversion token (tests Axiom 2: Mizan)
    close = generate_mean_reversion(n, 100.0, volatility=0.02, reversion_speed=0.05)
    returns = np.diff(close) / close[:-1]
    tokens.append({
        "ticker": "MIZAN_TOKEN",
        "close": close,
        "volume": generate_volume(n, returns, 1000000),
        "returns": returns,
        "timestamps": np.array([1767225600 + i * 3600 for i in range(n)], dtype=float),
    })

    # 2. Panic-recovery token (tests Axiom 1: Hifz al-Mal)
    close = generate_panic_sell(n, 50.0, panic_point=0.5, recovery=0.4)
    returns = np.diff(close) / close[:-1]
    tokens.append({
        "ticker": "HIFZ_TOKEN",
        "close": close,
        "volume": generate_volume(n, returns, 800000),
        "returns": returns,
        "timestamps": np.array([1767225600 + i * 3600 for i in range(n)], dtype=float),
    })

    # 3. Bubble-pop token (tests Axiom 3: Israf)
    close = generate_spike_and_decay(n, 10.0, spike_magnitude=3.0, decay_rate=0.995)
    returns = np.diff(close) / close[:-1]
    tokens.append({
        "ticker": "ISRAF_TOKEN",
        "close": close,
        "volume": generate_volume(n, returns, 600000),
        "returns": returns,
        "timestamps": np.array([1767225600 + i * 3600 for i in range(n)], dtype=float),
    })

    # 4. High-chaos token (tests Axiom 4: Gharar)
    close = generate_gharar_chaos(n, 5.0)
    returns = np.diff(close) / close[:-1]
    tokens.append({
        "ticker": "GHARAR_TOKEN",
        "close": close,
        "volume": generate_volume(n, returns, 400000),
        "returns": returns,
        "timestamps": np.array([1767225600 + i * 3600 for i in range(n)], dtype=float),
    })

    # 5. Staircase token (tests Axiom 7: Sabr)
    close = generate_staircase_decline(n, 200.0,
        steps=[180, 160, 140, 120, 110, 105],
        sabr_durations=[200, 180, 160, 140, 120, 100])
    returns = np.diff(close) / close[:-1]
    tokens.append({
        "ticker": "SABR_TOKEN",
        "close": close,
        "volume": generate_volume(n, returns, 700000),
        "returns": returns,
        "timestamps": np.array([1767225600 + i * 3600 for i in range(n)], dtype=float),
    })

    # 6. Stable DeFi token — ETH-like (tests all axioms in natural mix)
    # Combine patterns: accumulation → pump → correction → recovery → range
    p1 = generate_mean_reversion(400, 2000, volatility=0.015, reversion_speed=0.03)
    p2_start = p1[-1]
    p2 = generate_spike_and_decay(400, p2_start, spike_magnitude=0.4, decay_rate=0.992)
    p3 = generate_panic_sell(400, p2[-1], panic_point=0.4, recovery=0.5)
    p4 = generate_mean_reversion(400, p3[-1], volatility=0.018, reversion_speed=0.04)
    p5 = generate_staircase_decline(400, p4[-1],
        steps=[p4[-1]*0.95, p4[-1]*0.90, p4[-1]*0.87],
        sabr_durations=[100, 80, 60])
    close = np.concatenate([p1, p2, p3, p4, p5])[:n]
    returns = np.diff(close) / close[:-1]
    tokens.append({
        "ticker": "ETH_SIM",
        "close": close,
        "volume": generate_volume(n, returns, 5000000),
        "returns": returns,
        "timestamps": np.array([1767225600 + i * 3600 for i in range(n)], dtype=float),
    })

    return tokens


def save_synthetic_data():
    """Generate and save all synthetic data."""
    all_tokens = {}

    print("Generating DOTA (observed)...")
    all_tokens["DOTA"] = generate_dota()

    print("Generating CLANKER (observed)...")
    all_tokens["CLANKER"] = generate_clanker()

    print("Generating archetypal tokens...")
    for token in generate_archetypal_tokens():
        print(f"  {token['ticker']}...")
        all_tokens[token["ticker"]] = token

    # Save each token as JSON (CoinGecko-compatible format)
    for ticker, data in all_tokens.items():
        # Convert to CoinGecko format: {"prices": [[ts_ms, price], ...], "total_volumes": [[ts_ms, vol], ...]}
        cg_format = {
            "prices": [[float(t * 1000), float(p)] for t, p in zip(data["timestamps"], data["close"])],
            "total_volumes": [[float(t * 1000), float(v)] for t, v in zip(data["timestamps"], data["volume"])],
            "market_caps": [[float(t * 1000), 0.0] for t in data["timestamps"]],
        }
        out_path = DATA_DIR / f"{ticker.lower()}_2026.json"
        with open(out_path, "w") as f:
            json.dump(cg_format, f)
        print(f"  Saved {out_path} ({len(data['close'])} points)")

    print(f"\nGenerated {len(all_tokens)} tokens total.")
    return all_tokens


if __name__ == "__main__":
    save_synthetic_data()
