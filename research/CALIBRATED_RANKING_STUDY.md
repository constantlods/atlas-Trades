# Calibrated Ranking Study

**Date:** 2026-09-27
**Status:** Research layer on top of an unchanged Ross/ICC baseline. No live strategy, filter, or threshold was changed as a result of this study.
**Data:** 32,348 historical candidates (full ~13,500-symbol tradable universe, last 250 trading days, both directions, no cherry-picking) discovered by a ~15.5-hour, memory-safe batched collection run — 770 causally-valid ICC signals, 714 with a computable primary label. This expands the prior study's 124-signal sample by 6.2x; the true signal conversion rate on this larger, more representative sample (2.4%) turned out lower than the earlier small sample suggested (4.3%), which is itself a disclosed finding, not a target met (1,000+ signals would have required processing nearly 4x more candidates than was practical in one run).

**Reproduce with:** `research/scripts/large_scale_collector.py` (collection, batched/checkpointed) then `research/scripts/calibrated_ranking_study.py` (calibration/ranking analysis) in this repo.

---

## The headline question

*"Can Atlas identify a subset of Ross + ICC signals whose probability of a favorable executable outcome is substantially higher than the base population, and does that relationship survive walk-forward testing?"*

**Partially — and only for a specific tier and model, not uniformly.** Every result below is from an expanding-window walk-forward split (train on everything chronologically before a fold, test on the fold, never the reverse) with predictions pooled across folds before any bucket/ranking analysis — never a single train/test split.

| Selection | n (pooled out-of-sample) | Actual success rate |
|---|---|---|
| **ALL signals** | 500 | **21.0%** [17.7%, 24.8%] |
| Top 25% (random forest, calibrated) | 121 | 30.6% |
| Top 10% (random forest, calibrated) | 49 | 34.7% |
| **Top 5% (random forest, calibrated)** | 25 | **36.0%** |
| High-confidence (predicted >40%) | 3 | too few to report |
| **MEDIUM tier (25–40% predicted, logistic)** | **151** | **35.1%** [27.9%, 43.0%] |

The random-forest ranking shows a real, monotonic, out-of-sample lift (21% → 30% → 35% → 36% narrowing to the top 5%). **This does not replicate across models** — logistic regression's top 5% actually *underperformed* the base rate (20% and 16% for its raw and calibrated variants), and its own "HIGH confidence" tier (>40%) scored *worse* than its lower tiers (8.7% vs. 19–35%). This inconsistency across models is the single most important finding: the ranking signal is not robust to model choice, which is exactly the failure mode expected of an effect that isn't real.

## Signal vs. execution feature separation

Per the research brief driving this study, only SIGNAL features (relative volume, % gain, average volume, price, the full set of re-derived ICC structural features — indication strength, correction depth/duration, confirmation strength/duration, breakout candle size, pullback/confirmation volume, volume expansion, VWAP distance, hour of day) were fed to the model. EXECUTION features (spread%, execution tier, expected net R, shortability status) were tracked and reported separately and never used as model inputs.

## Calibration (mandatory check)

All three models (logistic raw, logistic calibrated via `CalibratedClassifierCV`/sigmoid, random forest calibrated via isotonic) are reasonably calibrated in the 0–40% predicted range — calibration error (|predicted − actual|) ranges only 0.001–0.11 across every reliably-sized bucket (n≥15). **No model ever produces a reliably-sized prediction above 40%** — every bucket from 40–100% falls below the 15-sample reliability floor for all three models. There is no populated "70–80% bucket" to test against the illustrative 75% example from the original brief. Honest answer: **the calibration question about high-confidence predictions cannot be answered yet, because the models never produce high-confidence predictions on this data.**

### Full calibration table (logistic_raw, pooled n=500)

| Predicted range | n | Predicted mean | Actual rate | Calibration error |
|---|---|---|---|---|
| 0.0–0.1 | 77 | 0.064 | 0.104 | 0.040 |
| 0.1–0.2 | 145 | 0.160 | 0.179 | 0.019 |
| 0.2–0.3 | 196 | 0.247 | 0.224 | 0.023 |
| 0.3–0.4 | 59 | 0.336 | 0.424 | 0.088 |
| 0.4–1.0 | 23 total, split across 6 buckets | — | — | below reliability floor throughout |

## The one genuinely clean result: the MEDIUM abstain tier

| Tier (logistic_raw) | n | Actual rate | 95% CI |
|---|---|---|---|
| NO_TRADE (<15%) | 129 | 9.3% | [5.4%, 15.6%] |
| LOW (15–25%) | 197 | 19.3% | [14.4%, 25.4%] |
| **MEDIUM (25–40%)** | **151** | **35.1%** | **[27.9%, 43.0%]** |
| HIGH (>40%) | 23 | 8.7% (inverted, small n) | [2.4%, 26.8%] |

NO_TRADE → LOW → MEDIUM is a clean, monotonic, reasonably-sized separation. This is the most defensible "identifiable subset" finding in the study — notably a *middle* band, not the top, and it does not extend to "HIGH."

## Critical caveat: direction-split reduces the apparent effect

Ranking bullish and bearish separately (as required) shrinks random forest's apparent lift substantially:

| Direction | n | Base rate | Top-20% (within direction) rate |
|---|---|---|---|
| Bullish | 359 | 22.0% [18.0%, 26.6%] | 23.9% (n=71) |
| Bearish | 141 | 18.4% [12.9%, 25.6%] | 25.0% (n=28) |

Compare to the **pooled** (both directions combined) top-20% result of 30.3%. A large share of the pooled ranking's apparent value looks like the model partly learning to separate bullish/bearish base rates rather than fine-grained within-direction ranking skill. Within each direction alone the lift is real but much smaller (roughly +2 to +7 points, not +9 to +15).

## Economic value by tier (random forest, MFE/MAE in R)

| Tier | Mean MFE (R) | Mean MAE (R) |
|---|---|---|
| NO_TRADE | 0.38 | 0.42 |
| LOW | 0.79 | 0.61 |
| MEDIUM | 0.78 | 0.70 |

Higher tiers show more upside *and* more downside — a real risk/reward tradeoff, not a free improvement.

## Small-account economics (random forest tiers, $30–$1,000)

Executability improves modestly with tier — NO_TRADE plateaus at 4/121 executable by $500; MEDIUM reaches 24/172 — but stays a small fraction of the tier throughout. Account size was never close to the limiting factor at this signal-quality level; liquidity/shortability dominates.

## Bearish integrity

43 of 200 bearish signals in the modeling set have a *determined* executable-success verdict; 157 remain honestly `SHORTABILITY_UNKNOWN` — never coerced to true or false, matching the tri-state convention used throughout this research phase.

## Direct answers

1. **Does Atlas reliably identify a materially-better subset?** Weak evidence for one model (random forest, pooled) and one tier (MEDIUM, logistic) — not a consistent finding across models.
2. **Does it survive walk-forward testing?** Yes in the sense that every number above *is* the out-of-sample walk-forward result — but the inconsistency between models on the identical out-of-sample data is itself evidence the effect is fragile.
3. **Does it survive direction-separation?** Substantially weaker once bullish/bearish are ranked separately, as required.
4. **Is 714 enough?** Better than 124 (the prior study's sample), but the clearest tell in this study — no model ever populates a high-confidence bucket — says the sample still is not large enough to test the part of the original question that matters most: whether high predicted confidence actually means high actual success.

**No live strategy, filter, or threshold was changed as part of this study.**
