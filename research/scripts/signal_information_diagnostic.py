"""Diagnoses whether the SIGNAL itself (Ross/bearish screen pass, and
separately, confirmed ICC continuation on top of it) carries predictive
information about forward price movement -- independent of execution
costs, position sizing, or any threshold tuning. Reuses the checkpointed
records from scripts/predictive_research_experiment.py's Phase 3 --
zero new API calls, no re-fetching.

Groups compared (mutually exclusive):
  SCREEN_ONLY   -- signal_direction is set (Ross or bearish screen passed)
                   but ICC never reached a causally-tradeable continuation
                   (signal_valid=False). Has record.outcome (computed
                   whenever signal_direction is set + bars exist,
                   regardless of ICC stage -- see backtesting/runner.py).
  FULL_SIGNAL   -- signal_direction is set AND signal_valid=True (ICC
                   continuation causally confirmed). Has record.outcome.

Both groups' record.outcome (backtesting.outcome.SignalOutcome) already
exists on the pickled records -- this script only reads/aggregates it,
computes nothing forward-looking that wasn't already computed
point-in-time-safe by the real pipeline.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "research_checkpoints" / "predictive_experiment"

with open(OUT_DIR / "phase3_checkpoint.pkl", "rb") as fh:
    checkpoint = pickle.load(fh)
records = checkpoint["records"]

HORIZONS = (1, 5, 15, 30, 60)


def favorable_move_percent(outcome, horizon):
    """Direction-aware forward return: positive = price moved the way
    the signal pointed. Identical convention to backtesting/narrative.py's
    own _favorable_move_percent -- reused here as the same formula, not
    re-derived differently."""
    if outcome is None or outcome.signal_price <= 0:
        return None
    price = outcome.price_at(horizon)
    if price is None:
        return None
    raw = (price - outcome.signal_price) / outcome.signal_price * 100
    return raw if outcome.direction == "bullish" else -raw


def group_records(records):
    screen_only, full_signal = [], []
    for r in records:
        if r.signal_direction is None or r.outcome is None:
            continue
        (full_signal if r.signal_valid else screen_only).append(r)
    return screen_only, full_signal


def summarize(records, label):
    n = len(records)
    print(f"\n--- {label} (n={n}) ---")
    result = {"n": n, "horizons": {}}
    if n == 0:
        return result
    for h in HORIZONS:
        values = [v for r in records if (v := favorable_move_percent(r.outcome, h)) is not None]
        if len(values) < 3:
            print(f"  {h:>2}m: n={len(values)} -- too few for a t-test")
            result["horizons"][h] = {"n": len(values), "mean": None, "p_value_vs_zero": None}
            continue
        arr = np.array(values)
        t_stat, p_value = stats.ttest_1samp(arr, 0.0)
        mean, sem = arr.mean(), stats.sem(arr)
        ci = stats.t.interval(0.95, len(arr) - 1, loc=mean, scale=sem) if sem > 0 else (mean, mean)
        print(f"  {h:>2}m: n={len(values):3d}  mean={mean:+.3f}%  95% CI=[{ci[0]:+.3f}%, {ci[1]:+.3f}%]  "
              f"p(vs 0)={p_value:.3f}{'  *' if p_value < 0.05 else ''}")
        result["horizons"][h] = {"n": len(values), "mean": float(mean), "ci": [float(ci[0]), float(ci[1])], "p_value_vs_zero": float(p_value)}

    mfe = [r.outcome.mfe_percent for r in records if r.outcome.mfe_percent is not None]
    mae = [r.outcome.mae_percent for r in records if r.outcome.mae_percent is not None]
    if mfe:
        print(f"  MFE%: mean={np.mean(mfe):.3f}  median={np.median(mfe):.3f}")
        print(f"  MAE%: mean={np.mean(mae):.3f}  median={np.median(mae):.3f}")
    result["mfe_mean"] = float(np.mean(mfe)) if mfe else None
    result["mae_mean"] = float(np.mean(mae)) if mae else None
    return result


def compare_groups(a, a_label, b, b_label, horizon=15):
    a_vals = [v for r in a if (v := favorable_move_percent(r.outcome, horizon)) is not None]
    b_vals = [v for r in b if (v := favorable_move_percent(r.outcome, horizon)) is not None]
    if len(a_vals) < 3 or len(b_vals) < 3:
        print(f"\n{a_label} vs {b_label} @ {horizon}m: too few observations for a comparison")
        return None
    t_stat, p_value = stats.ttest_ind(a_vals, b_vals, equal_var=False)
    u_stat, p_value_mw = stats.mannwhitneyu(a_vals, b_vals, alternative="two-sided")
    print(f"\n{a_label} (n={len(a_vals)}, mean={np.mean(a_vals):+.3f}%) vs "
          f"{b_label} (n={len(b_vals)}, mean={np.mean(b_vals):+.3f}%) @ {horizon}m")
    print(f"  Welch t-test p={p_value:.3f}   Mann-Whitney U p={p_value_mw:.3f}")
    return {"horizon": horizon, "a_n": len(a_vals), "a_mean": float(np.mean(a_vals)),
            "b_n": len(b_vals), "b_mean": float(np.mean(b_vals)),
            "t_test_p": float(p_value), "mann_whitney_p": float(p_value_mw)}


print("=" * 72)
print("SIGNAL INFORMATION DIAGNOSTIC")
print("=" * 72)

screen_only, full_signal = group_records(records)
print(f"\nSCREEN_ONLY: {len(screen_only)} candidates (screen passed, ICC never confirmed)")
print(f"FULL_SIGNAL: {len(full_signal)} candidates (screen passed AND ICC continuation confirmed)")

results = {}
results["screen_only"] = summarize(screen_only, "SCREEN_ONLY (all horizons, direction-aware, favorable=+)")
results["full_signal"] = summarize(full_signal, "FULL_SIGNAL (all horizons, direction-aware, favorable=+)")

comparisons = []
for h in HORIZONS:
    c = compare_groups(full_signal, "FULL_SIGNAL", screen_only, "SCREEN_ONLY", horizon=h)
    if c:
        comparisons.append(c)
results["full_vs_screen_only_comparisons"] = comparisons

# Direction split within FULL_SIGNAL
bullish = [r for r in full_signal if r.signal_direction == "bullish"]
bearish = [r for r in full_signal if r.signal_direction == "bearish"]
print("\n" + "=" * 72)
print("FULL_SIGNAL split by direction")
print("=" * 72)
results["full_signal_bullish"] = summarize(bullish, "FULL_SIGNAL / bullish")
results["full_signal_bearish"] = summarize(bearish, "FULL_SIGNAL / bearish")

# ICC stage dose-response (within screen-passed population overall)
print("\n" + "=" * 72)
print("DOSE-RESPONSE: forward outcome by furthest ICC stage reached")
print("=" * 72)
by_stage = {"indication": [], "correction": [], "continuation": [], "none": []}
for r in screen_only + full_signal:
    stage = r.icc_stage or "none"
    by_stage.setdefault(stage, []).append(r)
results["by_icc_stage"] = {}
for stage in ("none", "indication", "correction", "continuation"):
    results["by_icc_stage"][stage] = summarize(by_stage.get(stage, []), f"ICC stage = {stage}")

(OUT_DIR / "signal_information_diagnostic.json").write_text(json.dumps(results, indent=2, default=str))
print(f"\nsaved {OUT_DIR / 'signal_information_diagnostic.json'}")
