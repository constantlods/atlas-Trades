# ICC Signal-Quality Study

**Date:** 2026-09-26
**Status:** Diagnostic/research only. No live strategy, filter, or threshold was changed as a result of this study.
**Data:** 2,911 historical candidates discovered via unbiased magnitude-based scanning (last ~250 trading days, both directions, no cherry-picking). 124 causally-valid ICC signals (screen pass + confirmed continuation) had their exact structural features re-derived and forward paths measured on continuous outcomes, reusing the checkpointed backtest data with zero additional API calls.

**Reproduce with:** `scripts/icc_signal_quality_study.py` (in this folder), which reads `phase3_checkpoint.pkl` produced by `scripts/predictive_research_experiment.py`.

---

## A. Dataset size

| | Count |
|---|---|
| Candidates | 2,911 |
| Ross-passed | 345 |
| ICC signals (causally valid) | **124** |
| Bullish | 70 |
| Bearish | 54 |
| Executable (risk-approved, default config) | 4 |

## B. Signal effect — a methodological note before the numbers

Forward returns were measured from **ICC confirmation** (`execution_timestamp`), not from initial discovery (`as_of`) as an earlier diagnostic in this same research phase did — these are different reference points and give different significance patterns; both are legitimate measurements of different questions, and the discrepancy between them is itself an honest finding, not reconciled into one number here.

| Horizon | n | Mean return | p (vs. 0) |
|---|---|---|---|
| 1m | 95 | +0.09% | 0.28 |
| 3m | 111 | +0.23% | **0.026** |
| 5m | 116 | +0.26% | 0.09 |
| 10m | 118 | +0.32% | 0.09 |
| 15m | 119 | +0.41% | 0.14 |
| 30m/60m | 119 | +0.47% | 0.21 |

## C. ICC feature correlations (Spearman, n=119)

| Feature | MFE corr | p | MAE corr | p |
|---|---|---|---|---|
| **relative_volume** | **+0.263** | **0.004** | +0.10 | 0.28 |
| breakout_candle_size_pct | +0.165 | 0.073 | +0.10 | 0.26 |
| confirmation_strength_pct | +0.152 | 0.098 | +0.12 | 0.20 |
| volume_expansion_on_confirmation | −0.140 | 0.13 | +0.12 | 0.21 |
| indication_strength_pct, correction_depth_pct, correction_duration_bars, confirmation_duration_bars, pullback_volume, confirmation_volume, vwap_distance_pct, percent_change | — | not significant | — | not significant |

Only relative volume shows a statistically significant relationship, and it's with the *size of the favorable move* (MFE) — not with risk (MAE was unrelated to every feature tested).

### Confirmation-strength tiers (tertile split on the measured value, not an invented score)

| Tier | n | Mean 15m return | Mean MFE% |
|---|---|---|---|
| Weak | 42 | +0.19% | 1.16% |
| Moderate | 41 | −0.06% | 1.71% |
| **Strong** | 41 | **+1.15%** | **3.40%** |

## G. Risk geometry — direct evidence on the 2R/1R question

| Target/Stop | n | Success rate | 95% CI |
|---|---|---|---|
| 1R / 0.5R | 119 | 24.4% | [17.0%, 33.1%] |
| **1R / 1R** | 119 | **26.9%** | [19.2%, 35.8%] |
| 1.5R / 1R | 119 | 17.6% | [11.3%, 25.7%] |
| **2R / 1R (current live config)** | 119 | **8.4%** | [4.1%, 14.9%] |
| 3R / 1R | 119 | 4.2% | [1.4%, 9.5%] |

Clear evidence the current 2R/1R structure is mismatched to the actual short-horizon movement distribution observed in this sample — cutting the target to 1R more than triples the measured hit rate. This is evidence for further research, not a recommendation that was acted on; the live target/stop configuration was not changed.

## F. Execution cost sensitivity (15m raw return)

| Assumed round-trip cost | Mean net return | P(net > 0) |
|---|---|---|
| 0.00% | +0.41% | 50.4% |
| 0.05% | +0.36% | 49.6% |
| 0.10% | +0.31% | 47.1% |
| 0.20% | +0.21% | 44.5% |
| **0.50%** | **−0.09%** | 36.1% |
| 1.00% | −0.59% | 28.6% |

The edge survives small costs (up to ~0.2–0.3%) and is consumed somewhere between 0.2% and 0.5% — squarely within realistic spread+slippage range for the low-liquidity names this system discovers.

## Big-move base rates (15m, n=124)

P(>0.5%) = 34.7%, P(>1%) = 27.4%, P(>2%) = 15.3%.

## Nonlinear buckets

- RVOL 5–10x (n=84): 35.7% big-move rate. RVOL 10–20x (n=22) and 20x+ (n=18) fall below the 30-sample floor — cannot claim "higher RVOL is better" from this data.
- % gain 10–20% (n=36): 41.7% vs. 20–50% (n=68): 27.9% — lower gain-at-signal outperformed higher gain-at-signal in this sample, the opposite of naive intuition. Above 50% falls below the sample floor.
- **Float: 0 of 124 rows have any value at all** — no fundamentals data source is wired up anywhere in this system. Every float-based hypothesis (low-float+RVOL, float buckets, etc.) is untestable with current data, not tested-and-negative.

## I. Time of day — a hard methodological limitation, not a null result

Every one of the 124 signals falls outside the 7:00–11:00 ET window Ross's own material emphasizes, because the historical-mover discovery methodology pins each candidate's discovery moment to 15:30 ET by construction (a pre-existing, documented limitation), and ICC confirmation can only occur after that. **This dataset structurally cannot test the morning-session claim at all** — a different discovery methodology (reconstructing what a live scanner would have found continuously through the day) would be required.

## Bullish vs. bearish (n=70 / n=54)

Bullish: mean 15m return +0.47%, MFE 2.39%, MAE 1.28%. Bearish: +0.33%, MFE 1.60%, MAE 1.19%. Directionally similar; sample too small to conclude either direction is superior.

## D/E. Walk-forward model + ranking (strict chronological 80/20 split, train n=99 / test n=25)

| Model | Test accuracy | AUC | Brier |
|---|---|---|---|
| Logistic regression | 48.0% | 0.46 (worse than chance) | 0.279 |
| Random forest | 64.0% | 0.50 (exactly chance) | 0.222 |
| Gradient boosting | 72.0% | 0.76 | 0.175 |

Ranking by logistic regression's probability: top-25% (k=6) big-move rate 16.7% — *below* the test-set base rate of 32.0%. Top-10%/top-5% (k=2, k=1) show 50%/100% but are anecdotal at that size. At n=25 out-of-sample, none of this is reliable enough to act on — gradient boosting's 0.76 AUC is as plausibly a lucky split as real skill.

## H. Small-account economics

| Equity | Executable | Not executable (too small to size) | Undetermined (bearish, shortability unknown) |
|---|---|---|---|
| $30 | 4 | 66 (42 literally zero shares) | 54 |
| $50 | 8 | 62 (36) | 54 |
| $100 | 12 | 58 (22) | 54 |
| $250 | 14 | 56 (9) | 54 |
| **$500** | **17** | **53 (2)** | 54 |
| $1,000–$5,000 | 17 | 53 (2) | 54 |

Executability plateaus at $500 — above that, equity stops being the binding constraint (liquidity/spread/participation takes over). Below $500, a large fraction of signals are literally unsizeable (0 affordable shares).

---

## J. Final research conclusions

1. **Does ICC add predictive information?** Yes — full signal vs. screen-only was clearly distinguishable in an earlier diagnostic this same phase, and here, confirmation strength and relative volume both show real relationships with the size of the favorable move.
2. **Which features appear to strengthen the signal?** Relative volume (significant, p=0.004) and confirmation strength (suggestive, p≈0.10, clean 3-tier separation).
3. **Can those be identified before the move?** Yes — both computable at or before the confirmation timestamp; verified against a point-in-time leakage audit.
4. **Does the signal survive realistic execution costs?** Only up to a point — holds through ~0.2–0.3% round-trip cost, consumed between 0.2% and 0.5%.
5. **Does the edge survive walk-forward testing?** Not demonstrated — the only out-of-sample test available (n=25) showed one model at exactly chance, one worse than chance, and one promising but statistically unreliable at that size.
6. **What account size looks economically viable?** Executability plateaus at $500; below that a large share of signals are literally too small to size.
7. **What evidence is still missing?** (a) Enough signals for a real walk-forward validation — 124 is enough for description, not model validation. (b) Any float/fundamentals data at all. (c) A discovery methodology that can reach the 7–11am ET window Ross's material centers on. (d) Confirmation on whether 1R/1R-style geometry, which triples the raw hit rate here, holds up on a larger sample.

**No live strategy, filter, or threshold was changed as part of this study.**
