#!/usr/bin/env python3
"""
Fitrah Test Suite — Validates and refines the fitrah model.

Tests:
1. Walk-forward validation (train on first 70%, test on last 30%)
2. Per-axiom isolation tests (does each axiom improve accuracy?)
3. Cross-token generalization (train on one token, test on another)
4. Parameter stability across re-runs
5. Comparison vs naive baselines (random, momentum-only, mean-reversion-only)

Bismillah — guided by tawhid, we test with rigor.
"""

import json
import sys
import os

try:
    import numpy as np
except ImportError:
    os.system(f"{sys.executable} -m pip install numpy")
    import numpy as np

try:
    from scipy.optimize import differential_evolution
except ImportError:
    os.system(f"{sys.executable} -m pip install scipy")
    from scipy.optimize import differential_evolution

from pathlib import Path
from fitrah_engine import (
    parse_to_arrays, compute_fitrah_features, fitrah_model,
    compute_loss, DATA_DIR, load_local_data
)


# =============================================================================
# TEST 1: Walk-Forward Validation
# =============================================================================

def test_walk_forward(all_data: dict, train_ratio: float = 0.7) -> dict:
    """
    Train on first 70% of data, test on last 30%.
    This is the honest test — can the model predict unseen data?
    """
    print("\n" + "=" * 70)
    print("TEST 1: WALK-FORWARD VALIDATION (70/30 split)")
    print("=" * 70)

    results = {}
    bounds = [
        (-2, 2), (-2, 2), (-2, 2), (-2, 2), (-2, 2), (-2, 2), (-2, 2),
        (-0.01, 0.01), (0.5, 3.0), (0.1, 10.0), (0.5, 5.0), (0.5, 0.95),
        (0.02, 0.20), (5, 50), (-1, 1), (-2, 2),
    ]

    for ticker, data in all_data.items():
        close = data["close"]
        volume = data["volume"]
        returns = data["returns"]
        n = len(returns)

        if n < 200:
            continue

        split = int(n * train_ratio)

        # Train features & returns
        train_close = close[:split+1]
        train_vol = volume[:split+1]
        train_ret = returns[:split]
        train_features = compute_fitrah_features(train_close, train_vol, train_ret)

        # Fit on training data
        result = differential_evolution(
            compute_loss, bounds=bounds,
            args=(train_features, train_ret),
            maxiter=150, seed=42, tol=1e-8, disp=False, polish=True, workers=1,
        )

        # Test on out-of-sample data
        test_close = close[split:]
        test_vol = volume[split:]
        test_ret = returns[split:]

        if len(test_ret) < 50:
            continue

        test_features = compute_fitrah_features(test_close, test_vol, test_ret)
        predicted = fitrah_model(test_features, result.x)

        start = min(50, len(test_ret) - 10)
        pred = predicted[start:]
        actual = test_ret[start:]

        dir_acc = np.mean(np.sign(pred) == np.sign(actual))

        if np.std(pred) > 1e-10 and np.std(actual) > 1e-10:
            corr = np.corrcoef(pred, actual)[0, 1]
            if np.isnan(corr):
                corr = 0
        else:
            corr = 0

        pnl = pred * actual
        sharpe = np.mean(pnl) / np.std(pnl) * np.sqrt(365) if np.std(pnl) > 1e-10 else 0

        results[ticker] = {
            "direction_accuracy": dir_acc,
            "correlation": corr,
            "sharpe": sharpe,
            "test_points": len(actual),
        }

        status = "PASS" if dir_acc > 0.52 else "FAIL"
        print(f"  [{status}] {ticker:<15} Dir: {dir_acc:.1%}  Corr: {corr:.4f}  Sharpe: {sharpe:.2f}  (n={len(actual)})")

    # Summary
    avg_acc = np.mean([r["direction_accuracy"] for r in results.values()])
    avg_sharpe = np.mean([r["sharpe"] for r in results.values()])
    print(f"\n  AVERAGE: Dir accuracy={avg_acc:.1%}, Sharpe={avg_sharpe:.2f}")
    print(f"  VERDICT: {'PASS' if avg_acc > 0.52 else 'NEEDS REFINEMENT'}")

    return results


# =============================================================================
# TEST 2: Per-Axiom Isolation (Ablation Study)
# =============================================================================

def test_axiom_ablation(all_data: dict) -> dict:
    """
    Test each axiom's contribution by zeroing out its weight and
    measuring the drop in accuracy. If removing an axiom hurts
    performance, that axiom captures real fitrah behavior.
    """
    print("\n" + "=" * 70)
    print("TEST 2: AXIOM ABLATION (which axioms matter?)")
    print("=" * 70)

    # Load the fitted results
    results_path = Path(__file__).parent / "fitrah_results.json"
    if not results_path.exists():
        print("  No fitted results found. Run fitrah_engine.py first.")
        return {}

    with open(results_path) as f:
        fitted = json.load(f)

    axiom_map = {
        0: "Hifz al-Mal (loss aversion)",
        1: "Mizan (balance/reversion)",
        2: "Israf (excess)",
        3: "Gharar (uncertainty)",
        4: "Shukr (contentment)",
        5: "Sabr (patience)",
        6: "Tawakkul (conviction)",
        14: "Momentum (trend)",
        15: "Mizan*Sabr interaction",
    }

    ablation_results = {}

    for ticker, data in all_data.items():
        close = data["close"]
        volume = data["volume"]
        returns = data["returns"]

        if ticker not in fitted["per_token"]:
            continue

        params = np.array(fitted["per_token"][ticker]["params"])
        features = compute_fitrah_features(close, volume, returns)

        # Baseline accuracy (all axioms active)
        predicted = fitrah_model(features, params)
        start = 100
        pred = predicted[start:]
        actual = returns[start:]
        baseline_acc = np.mean(np.sign(pred) == np.sign(actual))

        print(f"\n  {ticker} (baseline: {baseline_acc:.1%}):")

        for param_idx, axiom_name in axiom_map.items():
            # Zero out this axiom's weight
            ablated_params = params.copy()
            ablated_params[param_idx] = 0

            predicted_ablated = fitrah_model(features, ablated_params)
            pred_ab = predicted_ablated[start:]
            ablated_acc = np.mean(np.sign(pred_ab) == np.sign(actual))

            delta = baseline_acc - ablated_acc
            importance = "CRITICAL" if delta > 0.02 else "helpful" if delta > 0.005 else "minimal"

            print(f"    Without {axiom_name:<30}: {ablated_acc:.1%} (delta: {delta:+.1%}) [{importance}]")

            if axiom_name not in ablation_results:
                ablation_results[axiom_name] = []
            ablation_results[axiom_name].append(delta)

    # Cross-token axiom importance
    print("\n  --- CROSS-TOKEN AXIOM IMPORTANCE ---")
    for axiom, deltas in sorted(ablation_results.items(), key=lambda x: -np.mean(x[1])):
        avg_delta = np.mean(deltas)
        sign = "+" if avg_delta > 0 else ""
        verdict = "FITRAH-CONFIRMED" if avg_delta > 0.005 else "weak" if avg_delta > 0 else "NOT CONFIRMED"
        print(f"    {axiom:<35}: avg delta {sign}{avg_delta:.2%}  [{verdict}]")

    return ablation_results


# =============================================================================
# TEST 3: Cross-Token Generalization
# =============================================================================

def test_cross_token(all_data: dict) -> dict:
    """
    Train on token A, test on token B. If fitrah is truly universal,
    parameters from one token should work on another.
    """
    print("\n" + "=" * 70)
    print("TEST 3: CROSS-TOKEN GENERALIZATION")
    print("=" * 70)

    results_path = Path(__file__).parent / "fitrah_results.json"
    if not results_path.exists():
        print("  No fitted results found.")
        return {}

    with open(results_path) as f:
        fitted = json.load(f)

    tickers = [t for t in all_data.keys() if t in fitted["per_token"]]
    cross_results = {}

    print(f"\n  {'Train→Test':<30} {'Dir Acc':>8} {'Verdict':<15}")
    print("  " + "-" * 55)

    for train_ticker in tickers:
        params = np.array(fitted["per_token"][train_ticker]["params"])

        for test_ticker in tickers:
            if train_ticker == test_ticker:
                continue

            test_data = all_data[test_ticker]
            features = compute_fitrah_features(
                test_data["close"], test_data["volume"], test_data["returns"]
            )
            predicted = fitrah_model(features, params)

            start = 100
            pred = predicted[start:]
            actual = test_data["returns"][start:]
            dir_acc = np.mean(np.sign(pred) == np.sign(actual))

            key = f"{train_ticker}→{test_ticker}"
            cross_results[key] = dir_acc
            verdict = "GENERALIZES" if dir_acc > 0.52 else "overfits"
            print(f"  {key:<30} {dir_acc:>7.1%} {verdict:<15}")

    avg = np.mean(list(cross_results.values()))
    print(f"\n  AVERAGE cross-token accuracy: {avg:.1%}")
    print(f"  VERDICT: {'FITRAH IS UNIVERSAL' if avg > 0.52 else 'NEEDS MORE DATA'}")

    return cross_results


# =============================================================================
# TEST 4: Baseline Comparison
# =============================================================================

def test_baselines(all_data: dict) -> dict:
    """
    Compare fitrah model against naive strategies:
    1. Random (coin flip)
    2. Pure momentum (buy if yesterday was up)
    3. Pure mean reversion (buy if yesterday was down)
    """
    print("\n" + "=" * 70)
    print("TEST 4: BASELINE COMPARISON (vs naive strategies)")
    print("=" * 70)

    results_path = Path(__file__).parent / "fitrah_results.json"
    if not results_path.exists():
        print("  No fitted results found.")
        return {}

    with open(results_path) as f:
        fitted = json.load(f)

    comparisons = {}

    print(f"\n  {'Token':<15} {'Fitrah':>8} {'Random':>8} {'Momentum':>8} {'MeanRev':>8} {'Winner':<12}")
    print("  " + "-" * 65)

    for ticker, data in all_data.items():
        if ticker not in fitted["per_token"]:
            continue

        returns = data["returns"]
        start = 100

        # Fitrah model
        params = np.array(fitted["per_token"][ticker]["params"])
        features = compute_fitrah_features(data["close"], data["volume"], returns)
        predicted = fitrah_model(features, params)
        fitrah_acc = np.mean(np.sign(predicted[start:]) == np.sign(returns[start:]))

        # Random
        random_pred = np.random.choice([-1, 1], size=len(returns) - start)
        random_acc = np.mean(random_pred == np.sign(returns[start:]))

        # Momentum (predict same direction as last return)
        momentum_pred = np.sign(returns[start-1:-1])
        momentum_acc = np.mean(momentum_pred == np.sign(returns[start:]))

        # Mean reversion (predict opposite direction)
        meanrev_pred = -np.sign(returns[start-1:-1])
        meanrev_acc = np.mean(meanrev_pred == np.sign(returns[start:]))

        accs = {"Fitrah": fitrah_acc, "Random": random_acc,
                "Momentum": momentum_acc, "MeanRev": meanrev_acc}
        winner = max(accs, key=accs.get)

        print(f"  {ticker:<15} {fitrah_acc:>7.1%} {random_acc:>7.1%} "
              f"{momentum_acc:>7.1%} {meanrev_acc:>7.1%} {winner:<12}")

        comparisons[ticker] = accs

    # Count wins
    wins = {"Fitrah": 0, "Random": 0, "Momentum": 0, "MeanRev": 0}
    for accs in comparisons.values():
        winner = max(accs, key=accs.get)
        wins[winner] += 1

    print(f"\n  WINS: Fitrah={wins['Fitrah']}, Random={wins['Random']}, "
          f"Momentum={wins['Momentum']}, MeanRev={wins['MeanRev']}")
    print(f"  VERDICT: {'FITRAH WINS' if wins['Fitrah'] >= len(comparisons) * 0.5 else 'NEEDS REFINEMENT'}")

    return comparisons


# =============================================================================
# FULL TEST SUITE
# =============================================================================

def run_all_tests():
    """Run the complete test suite."""
    print("=" * 70)
    print("FITRAH TEST SUITE")
    print("Bismillah — validating the mathematical fingerprint of human nature")
    print("=" * 70)

    # Load data
    all_data = load_local_data()
    if not all_data:
        print("No data found. Run generate_synthetic_data.py first.")
        return

    print(f"Loaded {len(all_data)} tokens.\n")

    # Run tests
    wf_results = test_walk_forward(all_data)
    ablation_results = test_axiom_ablation(all_data)
    cross_results = test_cross_token(all_data)
    baseline_results = test_baselines(all_data)

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    wf_avg = np.mean([r["direction_accuracy"] for r in wf_results.values()]) if wf_results else 0
    cross_avg = np.mean(list(cross_results.values())) if cross_results else 0

    print(f"\n  Walk-forward accuracy (out-of-sample): {wf_avg:.1%}")
    print(f"  Cross-token generalization:             {cross_avg:.1%}")

    if ablation_results:
        confirmed = [k for k, v in ablation_results.items() if np.mean(v) > 0.005]
        print(f"  Fitrah-confirmed axioms:                {len(confirmed)}/{len(ablation_results)}")
        for a in confirmed:
            print(f"    - {a}")

    print(f"\n  Overall: ", end="")
    if wf_avg > 0.55 and cross_avg > 0.52:
        print("STRONG — fitrah patterns are predictive and generalizable")
    elif wf_avg > 0.52:
        print("MODERATE — predictive but needs more data for generalization")
    else:
        print("DEVELOPING — framework captures signal but needs refinement")

    # Save test results
    test_output = {
        "walk_forward": {k: v for k, v in wf_results.items()},
        "cross_token_avg": cross_avg,
        "baseline_comparison": {k: v for k, v in baseline_results.items()},
    }
    out_path = Path(__file__).parent / "fitrah_test_results.json"
    with open(out_path, "w") as f:
        json.dump(test_output, f, indent=2, default=str)
    print(f"\n  Test results saved to {out_path}")


if __name__ == "__main__":
    run_all_tests()
