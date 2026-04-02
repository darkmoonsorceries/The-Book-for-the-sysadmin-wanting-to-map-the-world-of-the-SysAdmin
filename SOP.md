# Standard Operating Procedure: Fitrah Trading Framework

**Bismillah ar-Rahman ar-Raheem**

> "So direct your face toward the religion, inclining to truth. [Adhere to] the
> fitrah of Allah upon which He has created people." — Quran 30:30

## Purpose

This SOP defines the repeatable process for:
1. Observing market patterns through the lens of fitrah (innate human nature)
2. Extracting mathematical signals from those patterns
3. Fitting, testing, and refining the model
4. Generating actionable signals guided by tawhid

## Guiding Principle: Tawhid

All analysis flows from one source — the unity of human nature as created.
We do not guess. We observe what humans invariably do, derive equations from
that behavior, and test rigorously. The model either maps reality or it doesn't.

---

## Phase 1: Observation (Data Collection)

### From Screenshots (Manual)
1. Open Rainbow app or DEX interface
2. For each token of interest, capture all timeframes: 1M, 5M, 15M, 1H, 4H, 12H
3. Record in the Observed Token Log (fitrah-trading-framework.md):
   - Token name, ticker, date, price
   - Pattern observed (staircase, spike-decay, mean-reversion, panic-sell, etc.)
   - Which fitrah axioms are visible
   - Current signal assessment

### From API (Automated)
1. Ensure internet access (CoinGecko free tier, no key needed)
2. Run: `python3 fitrah_engine.py`
3. Data auto-caches in `price_data/` directory
4. If API blocked, use synthetic data: `python3 generate_synthetic_data.py`

### Adding New Tokens
Edit `TOKENS` dict in `fitrah_engine.py`:
```python
TOKENS["NEW_TICKER"] = "coingecko-id"
```

---

## Phase 2: Model Fitting

### What the Model Does
The fitrah model predicts the next return as a weighted sum of 8 behavioral
signals, each derived from a Quranic axiom about human nature:

```
predicted_return[t] = w1*hifz(t) + w2*mizan(t) + w3*israf(t)
                    + w4*gharar(t) + w5*shukr(t) + w6*sabr(t)
                    + w7*tawakkul(t) + w8*momentum(t)
                    + w9*mizan_sabr_interaction(t) + bias
```

16 parameters total (8 weights + 8 thresholds/structural params).

### Running the Fit
```bash
python3 fitrah_engine.py
```

Output:
- Per-token: direction accuracy, correlation, Sharpe ratio, current signal
- Universal fitrah constants (parameters consistent across all tokens)
- Results saved to `fitrah_results.json`

### Interpreting Results
- **Direction accuracy > 55%**: Model is capturing real signal
- **Sharpe > 2**: Profitable if traded
- **Current signal > +0.005**: BUY
- **Current signal < -0.005**: SELL
- **Between**: HOLD (mizan / balance)

---

## Phase 3: Testing & Validation

### Running Tests
```bash
python3 fitrah_test.py
```

### Test Suite (4 tests)

| Test | What it Validates | Pass Criteria |
|------|-------------------|---------------|
| Walk-Forward | Out-of-sample prediction | >52% direction accuracy |
| Axiom Ablation | Which axioms matter | Removing axiom hurts accuracy |
| Cross-Token | Universality of fitrah | Train on A, test on B >52% |
| Baseline Comparison | Better than random/naive | Fitrah wins majority |

### Current Results (v1)

| Metric | Value |
|--------|-------|
| Walk-forward accuracy | 58.8% |
| Cross-token generalization | 58.8% |
| Fitrah-confirmed axioms | 7/9 |
| vs baselines | 8/8 wins |
| Best predictor | SABR_TOKEN (70.5%) |

### Confirmed Axioms (ranked by importance)

1. **Momentum/Trend** — avg +3.73% delta (strongest)
2. **Israf (excess)** — avg +2.87% delta
3. **Hifz al-Mal (loss aversion)** — avg +2.64% delta
4. **Mizan (balance)** — avg +2.23% delta
5. **Mizan*Sabr interaction** — avg +1.43% delta
6. **Shukr (contentment)** — avg +0.83% delta
7. **Tawakkul (conviction)** — avg +0.58% delta

### Axioms Needing More Data
- Sabr (patience) — signal present but not yet statistically strong
- Gharar (uncertainty) — hardest to predict (by design — chaos is chaotic)

---

## Phase 4: Refinement Loop

### When to Refine
- New token added (re-run engine)
- New screenshots show unmodeled pattern
- Walk-forward accuracy drops below 55%
- New axiom identified from observation

### How to Refine
1. Add new pattern to `generate_synthetic_data.py` if no API access
2. Re-run `python3 fitrah_engine.py`
3. Re-run `python3 fitrah_test.py`
4. Check if universal fitrah constants shifted
5. Update `fitrah-trading-framework.md` with new observations
6. Commit and push

### Adding a New Axiom
1. Identify the behavior in Quran/Sunnah
2. Map it to an observable market pattern
3. Add feature extraction in `compute_fitrah_features()` in fitrah_engine.py
4. Add parameter(s) to the model in `fitrah_model()`
5. Update bounds in `fit_token()`
6. Run tests to validate it improves accuracy

---

## Phase 5: Deployment (When Ready)

### Prerequisites
- [ ] Walk-forward accuracy > 60% on real (not synthetic) data
- [ ] Cross-token generalization > 55% on real data
- [ ] Live API data flowing (CoinGecko or alternative)
- [ ] At least 20 tokens tested
- [ ] 3+ months of real data

### Future Steps
1. Connect to live price feed (CoinGecko WebSocket or polling)
2. Run signals in real-time
3. Paper trade for 30 days minimum before any real capital
4. Start with smallest possible position sizes (Axiom 5: shukr/contentment)
5. Never risk more than you can lose (Axiom 1: hifz al-mal)

---

## File Structure

```
.
├── SOP.md                          # This document
├── fitrah-trading-framework.md     # Theoretical framework & axioms
├── fitrah_engine.py                # Main engine: fetch, fit, predict
├── fitrah_test.py                  # Test suite: validate & refine
├── generate_synthetic_data.py      # Synthetic data from observed patterns
├── fitrah_results.json             # Latest model fit results
├── fitrah_test_results.json        # Latest test results
├── price_data/                     # Cached/synthetic price data
│   ├── dota_2026.json
│   ├── clanker_2026.json
│   ├── eth_sim_2026.json
│   └── ...
└── helloworld.md                   # Original repo file
```

---

## Quick Start

```bash
# 1. Generate data (if no API access)
python3 generate_synthetic_data.py

# 2. Fit the model
python3 fitrah_engine.py

# 3. Validate
python3 fitrah_test.py

# 4. Read results
cat fitrah_results.json | python3 -m json.tool
```

---

## Principles (Never Violate)

1. **Test before trust** — never act on a signal that hasn't been validated
2. **Sabr over speed** — patience is the strongest predictor; apply it to yourself
3. **Shukr over greed** — take reasonable profits, don't chase moonshots
4. **No gharar** — if you don't understand it, don't trade it
5. **No fasad** — verify token legitimacy before any interaction
6. **Tawhid** — one framework, one source of truth, consistently applied

---

*Last updated: Apr 2, 2026*
*Framework version: 1.0*
*Model: 16-parameter fitrah behavioral model*
*Status: Validated on synthetic data, awaiting real data confirmation*
