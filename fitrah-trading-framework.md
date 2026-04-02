# Fitrah-Based Trading Framework

## Premise

If fiat currencies collapse, the remaining driver of value in decentralized markets
is **human fitrah** — the innate disposition described in the Quran (30:30):

> "So direct your face toward the religion, inclining to truth. [Adhere to] the
> fitrah of Allah upon which He has created people. No change should there be in
> the creation of Allah."

Markets are human behavior made visible. If we model the invariant behaviors that
fitrah produces, we get a trading framework that holds regardless of what currency
or token exists — because it's derived from what humans *are*, not what they
temporarily believe.

---

## Axioms Derived from Fitrah

### Axiom 1: Hifz al-Mal (Preservation of Wealth)

**Source:** Preservation of wealth is one of the five maqasid al-shariah
(objectives of Islamic law).

**Market implication:** Humans feel losses ~2x more than equivalent gains. This is
not a Western behavioral economics discovery — it is a consequence of fitrah. The
instinct to preserve what you have is innate.

**Rule:** Drops are faster and sharper than rises. Therefore:
- Use tighter stop-losses than take-profits (asymmetric exits)
- After a sharp drop, expect a bounce (oversold reversion)
- After a slow rise, don't assume continuation (exhaustion is gradual)

```
if price_drop_speed > 2 * average_rise_speed:
    signal = "oversold_bounce_likely"
    enter_long(size=conservative)
    stop_loss = entry - (0.5 * drop_magnitude)
    take_profit = entry + (1.5 * drop_magnitude)
```

### Axiom 2: Mizan (Balance)

**Source:** Quran 55:7-9 — "And the heaven He raised and imposed the balance. That
you not transgress within the balance. And establish weight in justice and do not
make deficient the balance."

**Market implication:** Everything seeks equilibrium. Price deviates from fair value
and returns. This is not a hypothesis — it's a description of created reality.

**Rule:** Calculate a moving equilibrium and trade deviations from it.

```
equilibrium = exponential_moving_average(price, period=50)
deviation = (price - equilibrium) / equilibrium

if deviation > +threshold:      # Price above balance
    signal = "fade_short"       # Mizan will restore
elif deviation < -threshold:    # Price below balance
    signal = "fade_long"        # Mizan will restore
else:
    signal = "hold"             # Within balance, no action
```

### Axiom 3: Israf (Excess) Self-Corrects

**Source:** Quran 7:31 — "Eat and drink, but be not excessive. Indeed, He likes not
those who commit excess."

**Market implication:** Bubbles are israf. Every excess in price is corrected. The
bigger the excess, the harder the correction.

**Rule:** Measure how "excessive" a move is relative to normal range. The more
excessive, the higher confidence in reversion.

```
normal_range = average_true_range(period=20)
current_move = abs(price - open_today)

excess_ratio = current_move / normal_range

if excess_ratio > 2.5:
    confidence = "high_reversion"
    # The more excessive the move, the stronger the fade
    position_size = base_size * min(excess_ratio / 2, 3)
```

### Axiom 4: Gharar (Uncertainty) Avoidance

**Source:** The Prophet (peace be upon him) prohibited transactions involving
excessive gharar (uncertainty/ambiguity).

**Market implication:** When uncertainty spikes (measured by volatility), humans
withdraw. Liquidity dries up. Prices become unreliable. This is fitrah — the
innate aversion to what you cannot understand or predict.

**Rule:** Do not trade in chaos. Wait for clarity.

```
volatility = standard_deviation(returns, period=20)
vol_percentile = percentile_rank(volatility, lookback=100)

if vol_percentile > 90:
    action = "exit_all"         # Too much gharar
    reason = "uncertainty_exceeds_tolerance"
elif vol_percentile > 75:
    action = "reduce_size"      # Elevated gharar
else:
    action = "normal_trading"   # Acceptable clarity
```

### Axiom 5: Shukr and Qana'ah (Gratitude and Contentment)

**Source:** Quran 14:7 — "If you are grateful, I will surely increase you [in
favor]."

**Market implication:** Take what the market gives you. Greed (wanting more than
what's been given) leads to holding too long and losing gains. Contentment with
reasonable profit is both spiritually and financially sound.

**Rule:** Fixed, disciplined take-profit levels. Never chase "the moon."

```
# Be content with reasonable gains
take_profit_1 = entry * 1.02    # Take 50% off at 2%
take_profit_2 = entry * 1.05    # Take remaining at 5%
max_hold_time = 24 * hours      # Don't overstay

# Never re-enter immediately after taking profit
# (gratitude = accepting what was given, not demanding more)
cooldown_after_profit = 1 * hour
```

### Axiom 6: Fasad (Corruption) Reveals Itself

**Source:** Quran 30:41 — "Corruption has appeared throughout the land and sea
because of what the hands of people have earned."

**Market implication:** Scams, rug-pulls, and fraudulent tokens inevitably collapse.
The signs are always there before the collapse — low real usage, concentrated
holdings, no genuine community.

**Rule:** Pre-filter before any trade. Do not enter what shows signs of fasad.

```
def passes_fasad_check(token):
    if top_10_holders_own > 80%:
        return False            # Concentrated = manipulable
    if daily_active_users < 100:
        return False            # No real community
    if liquidity_locked == False:
        return False            # Rug-pull risk
    if contract_verified == False:
        return False            # Hidden corruption
    return True
```

---

## Combined Algorithm

```python
def fitrah_signal(token, price_data):
    """
    Trading framework derived from human fitrah.
    Models invariant human behavior, not temporary market conditions.
    """

    # Axiom 6: First, check for fasad (corruption)
    if not passes_fasad_check(token):
        return "DO_NOT_TRADE"

    # Axiom 4: Check gharar (uncertainty) level
    vol_state = assess_gharar(price_data)
    if vol_state == "excessive":
        return "EXIT_ALL"

    # Axiom 2: Calculate mizan (equilibrium)
    equilibrium = calculate_equilibrium(price_data)
    deviation = get_deviation(price_data.current, equilibrium)

    # Axiom 3: Check for israf (excess)
    excess = measure_excess(price_data)

    # Axiom 1: Check for hifz al-mal (loss-aversion driven oversold)
    drop_speed = measure_drop_speed(price_data)

    # Generate signal
    if drop_speed > 2 * average_rise_speed(price_data):
        signal = "BUY"          # Oversold by fitrah-driven panic
        size = "conservative"
    elif deviation > threshold and excess > 2.5:
        signal = "SELL"         # Israf + above mizan = fade
        size = "scaled_to_excess"
    elif deviation < -threshold:
        signal = "BUY"          # Below mizan = reversion expected
        size = "normal"
    else:
        signal = "HOLD"         # Within balance

    # Axiom 5: Always apply shukr (contentment) exits
    apply_take_profit_discipline(signal)

    return signal
```

---

## Why This Holds When Currencies Collapse

Traditional quant models are trained on historical price data denominated in fiat.
If USD or EUR collapse, those models break — their training data becomes
meaningless.

This framework doesn't depend on any currency or historical dataset. It depends on:

1. Humans fear loss more than they desire gain (fitrah)
2. Excess always corrects (mizan)
3. Greed overextends and reverts (israf)
4. Uncertainty causes withdrawal (gharar avoidance)
5. Contentment preserves gains (shukr/qana'ah)
6. Corruption self-destructs (fasad)

These are properties of human nature as created. They don't change with the
currency. Whether people trade in gold, Bitcoin, seashells, or a post-collapse
crypto token — they will still behave according to their fitrah.

The algorithm doesn't predict *price*. It predicts *human behavior* — and prices
are just human behavior made numerical.

---

## Important Disclaimers

- This is an intellectual framework, not financial advice
- All trading carries risk of loss
- Islamic scholars differ on whether cryptocurrency trading itself is halal
- Consult qualified scholars for fiqh rulings on specific transactions
- The Quranic verses cited are for deriving behavioral axioms, not for
  claiming divine endorsement of any trading strategy
