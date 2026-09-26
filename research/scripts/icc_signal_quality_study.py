"""ICC signal-quality study: what distinguishes ICC signals that produce
large/tradeable moves from those that produce small/untradeable ones.

Purely diagnostic -- reuses the checkpointed records/bars from
scripts/predictive_research_experiment.py's Phase 3 (zero new API calls)
plus already-TESTED functions (predictive_ranking.labels.compute_labels
for R-multiple sensitivity, execution.account_sim.simulate_account_sizes
for small-account economics, predictive_ranking.statistics.
conditional_probability + ResearchCriteria for sample-size-gated bucket
stats, backtesting.runner._earliest_confirmed for reproducing the EXACT
same ICC setup selection the original backtest used). New computation is
limited to: ICC structural feature extraction, extended continuous
outcomes, and cost/R-multiple sensitivity sweeps -- none of it touches
ross_strategy/icc_strategy/risk_manager thresholds.

Does NOT modify any tested module. Does NOT change any live config.
"""
import json
import pickle
import sys
from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analytics.report import compute_performance  # noqa: E402
from backtesting.research_criteria import ResearchCriteria  # noqa: E402
from backtesting.runner import _earliest_confirmed  # noqa: E402
from execution.account_sim import simulate_account_sizes  # noqa: E402
from icc_strategy import detect_icc_setups  # noqa: E402
from market_data.base import AccountStatus, MarketDataError, MarketDataProvider, Quote  # noqa: E402
from market_structure import MarketStructureConfig, analyze_structure  # noqa: E402
from market_structure.vwap import session_vwap  # noqa: E402
from predictive_ranking.labels import LabelConfig, compute_labels  # noqa: E402
from predictive_ranking.statistics import conditional_probability  # noqa: E402
from risk_manager import RiskConfig  # noqa: E402
from trade_plan.models import TradePlan  # noqa: E402

import pandas as pd  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "research_checkpoints" / "predictive_experiment"
EASTERN = ZoneInfo("America/New_York")
STRUCTURE_CONFIG = MarketStructureConfig()  # BacktestConfig()'s own default -- must match what produced these records

with open(OUT_DIR / "phase3_checkpoint.pkl", "rb") as fh:
    checkpoint = pickle.load(fh)
records = checkpoint["records"]
bars_by_symbol = checkpoint["bars_by_symbol"]

full_signal = [r for r in records if r.signal_valid]
print(f"FULL_SIGNAL candidates: {len(full_signal)}")

# ---------------------------------------------------------------------------
# A. Re-derive each record's exact ICCSetup (same selection logic runner.py
#    used) and extract structural features -- SIGNAL features only.
# ---------------------------------------------------------------------------


def find_matching_setup(record):
    bars = bars_by_symbol.get(record.symbol, [])
    if not bars:
        return None, bars
    structure = analyze_structure(record.symbol, bars, STRUCTURE_CONFIG)
    setups = detect_icc_setups(record.symbol, bars, structure, STRUCTURE_CONFIG)
    confirmed = [s for s in setups if s.stage == "continuation"]
    tradeable = [s for s in confirmed if s.continuation.timestamp >= record.as_of]
    if not tradeable:
        return None, bars
    direction_agreeing = [s for s in tradeable if s.direction == record.signal_direction]
    setup = _earliest_confirmed(direction_agreeing or tradeable)
    return setup, bars


def extract_icc_features(record, setup, bars):
    ind, corr, cont = setup.indication, setup.correction, setup.continuation
    vwap = session_vwap(bars)

    indication_strength = abs(ind.breakout_close - ind.level) / ind.level * 100 if ind.level else None
    correction_depth = abs(ind.level - corr.extreme_point.price) / ind.level * 100 if ind.level else None
    correction_duration_bars = corr.extreme_point.bar_index - ind.bar_index
    confirmation_strength = (
        abs(cont.confirmation_close - corr.reaction_level) / corr.reaction_level * 100
        if corr.reaction_level else None
    )
    confirmation_duration_bars = cont.bar_index - corr.extreme_point.bar_index
    breakout_bar = bars[ind.bar_index]
    breakout_candle_size_pct = (breakout_bar.high - breakout_bar.low) / breakout_bar.low * 100 if breakout_bar.low else None

    correction_window = bars[ind.bar_index:corr.extreme_point.bar_index + 1]
    confirmation_window = bars[corr.extreme_point.bar_index + 1:cont.bar_index + 1]
    pullback_volume = float(np.mean([b.volume for b in correction_window])) if correction_window else None
    confirmation_volume = float(np.mean([b.volume for b in confirmation_window])) if confirmation_window else None
    volume_expansion = (
        confirmation_volume / pullback_volume if pullback_volume and confirmation_volume else None
    )

    vwap_at_confirmation = vwap[cont.bar_index] if cont.bar_index < len(vwap) else None
    vwap_distance_pct = (
        (cont.confirmation_close - vwap_at_confirmation) / vwap_at_confirmation * 100
        if vwap_at_confirmation else None
    )

    et_time = cont.timestamp.astimezone(EASTERN)
    return {
        "indication_strength_pct": indication_strength,
        "correction_depth_pct": correction_depth,
        "correction_duration_bars": correction_duration_bars,
        "confirmation_strength_pct": confirmation_strength,
        "confirmation_duration_bars": confirmation_duration_bars,
        "breakout_candle_size_pct": breakout_candle_size_pct,
        "pullback_volume": pullback_volume,
        "confirmation_volume": confirmation_volume,
        "volume_expansion_on_confirmation": volume_expansion,
        "vwap_distance_pct": vwap_distance_pct,
        "hour_et": et_time.hour,
        "minute_et": et_time.minute,
    }


rows = []
for r in full_signal:
    setup, bars = find_matching_setup(r)
    if setup is None:
        continue
    feats = extract_icc_features(r, setup, bars)
    row = {
        "symbol": r.symbol.split("__")[0], "direction": r.signal_direction,
        "relative_volume": r.relative_volume, "percent_change": r.percent_change,
        "avg_volume": r.avg_volume, "float_shares": r.float_shares, "price": r.price,
        "has_catalyst": r.has_catalyst, "market_regime": r.market_regime,
        "plan_entry": r.plan_entry, "plan_stop": r.plan_stop, "plan_target": r.plan_target,
        "plan_risk_per_share": r.plan_risk_per_share, "execution_timestamp": r.execution_timestamp,
        "spread_percent": r.spread_percent,
    }
    row.update(feats)
    row["_record"] = r
    row["_bars"] = bars
    rows.append(row)

print(f"ICC structural features extracted for {len(rows)}/{len(full_signal)} FULL_SIGNAL records")
df = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# B/C. Extended continuous outcomes: forward returns at more horizons,
#      MFE/MAE in R and %, time-to-MFE/MAE, time-to-partial-R.
# ---------------------------------------------------------------------------

FORWARD_HORIZONS = (1, 3, 5, 10, 15, 30, 60)
R_CHECKPOINTS = (0.5, 1.0, 1.5, 2.0)


def extended_outcome(row):
    r = row["_record"]
    bars = row["_bars"]
    entry = row["plan_entry"]
    risk_per_share = row["plan_risk_per_share"]
    direction = row["direction"]
    if entry is None or risk_per_share is None or not risk_per_share:
        return {}

    window = [b for b in bars if b.timestamp > r.execution_timestamp]
    out = {}
    for h in FORWARD_HORIZONS:
        cutoff = r.execution_timestamp + timedelta(minutes=h)
        candidates_h = [b for b in window if b.timestamp <= cutoff]
        if not candidates_h:
            out[f"return_{h}m_pct"] = None
            continue
        price = candidates_h[-1].close
        raw = (price - entry) / entry * 100
        out[f"return_{h}m_pct"] = raw if direction == "bullish" else -raw

    if not window:
        return out
    highs = np.array([b.high for b in window])
    lows = np.array([b.low for b in window])
    times = [b.timestamp for b in window]
    if direction == "bullish":
        mfe_idx = int(np.argmax(highs))
        mae_idx = int(np.argmin(lows))
        mfe_price, mae_price = highs[mfe_idx], lows[mae_idx]
        mfe_pct = (mfe_price - entry) / entry * 100
        mae_pct = (entry - mae_price) / entry * 100
    else:
        mfe_idx = int(np.argmin(lows))
        mae_idx = int(np.argmax(highs))
        mfe_price, mae_price = lows[mfe_idx], highs[mae_idx]
        mfe_pct = (entry - mfe_price) / entry * 100
        mae_pct = (mae_price - entry) / entry * 100
    out["mfe_pct"] = max(mfe_pct, 0.0)
    out["mae_pct"] = max(mae_pct, 0.0)
    out["mfe_r"] = out["mfe_pct"] / 100 * entry / risk_per_share
    out["mae_r"] = out["mae_pct"] / 100 * entry / risk_per_share
    out["time_to_mfe_min"] = (times[mfe_idx] - r.execution_timestamp).total_seconds() / 60
    out["time_to_mae_min"] = (times[mae_idx] - r.execution_timestamp).total_seconds() / 60
    out["mfe_before_mae"] = out["time_to_mfe_min"] <= out["time_to_mae_min"]

    for rc in R_CHECKPOINTS:
        target_level = entry + rc * risk_per_share if direction == "bullish" else entry - rc * risk_per_share
        hit_time = None
        for b in window:
            touched = b.high >= target_level if direction == "bullish" else b.low <= target_level
            if touched:
                hit_time = (b.timestamp - r.execution_timestamp).total_seconds() / 60
                break
        out[f"time_to_{rc}R_min"] = hit_time
    return out


extended = [extended_outcome(row) for _, row in df.iterrows()]
ext_df = pd.DataFrame(extended)
full_df = pd.concat([df.drop(columns=["_record", "_bars"]), ext_df], axis=1)
full_df.to_csv(OUT_DIR / "icc_signal_quality_dataset.csv", index=False)
print(f"saved {OUT_DIR / 'icc_signal_quality_dataset.csv'} ({len(full_df)} rows, {len(full_df.columns)} columns)")

results = {"n_full_signal_with_features": len(full_df)}

# ---------------------------------------------------------------------------
# B. Signal effect (screen-only vs full-signal already established;
#    here: full-signal average forward return by horizon, direction split)
# ---------------------------------------------------------------------------
print("\n--- B. FORWARD RETURN BY HORIZON (all FULL_SIGNAL, direction-aware) ---")
results["forward_returns"] = {}
for h in FORWARD_HORIZONS:
    col = f"return_{h}m_pct"
    vals = full_df[col].dropna().values
    if len(vals) < 5:
        print(f"  {h:>2}m: n={len(vals)} -- too few")
        results["forward_returns"][h] = {"n": len(vals), "mean": None}
        continue
    t_stat, p = stats.ttest_1samp(vals, 0.0)
    print(f"  {h:>2}m: n={len(vals):3d} mean={vals.mean():+.3f}% p={p:.3f}{'  *' if p<0.05 else ''}")
    results["forward_returns"][h] = {"n": int(len(vals)), "mean": float(vals.mean()), "p_value": float(p)}

# ---------------------------------------------------------------------------
# C. ICC feature correlation with MFE / MAE
# ---------------------------------------------------------------------------
print("\n--- C. ICC FEATURE CORRELATION WITH MFE% / MAE% (Spearman) ---")
feature_cols = [
    "indication_strength_pct", "correction_depth_pct", "correction_duration_bars",
    "confirmation_strength_pct", "confirmation_duration_bars", "breakout_candle_size_pct",
    "pullback_volume", "confirmation_volume", "volume_expansion_on_confirmation",
    "vwap_distance_pct", "relative_volume", "percent_change",
]
corr_results = {}
for col in feature_cols:
    sub = full_df[[col, "mfe_pct", "mae_pct"]].dropna()
    if len(sub) < 10:
        corr_results[col] = {"n": len(sub), "mfe_corr": None, "mae_corr": None}
        continue
    mfe_r, mfe_p = stats.spearmanr(sub[col], sub["mfe_pct"])
    mae_r, mae_p = stats.spearmanr(sub[col], sub["mae_pct"])
    print(f"  {col:35s} n={len(sub):3d}  MFE corr={mfe_r:+.3f} (p={mfe_p:.3f})  MAE corr={mae_r:+.3f} (p={mae_p:.3f})")
    corr_results[col] = {"n": int(len(sub)), "mfe_corr": float(mfe_r), "mfe_p": float(mfe_p),
                          "mae_corr": float(mae_r), "mae_p": float(mae_p)}
results["icc_feature_correlations"] = corr_results

json.dump(results, open(OUT_DIR / "icc_signal_quality_partial.json", "w"), indent=2, default=str)
print(f"\n[checkpoint saved] {OUT_DIR / 'icc_signal_quality_partial.json'}")

# ---------------------------------------------------------------------------
# item 12. R-multiple / stop-target geometry sensitivity -- reuses the
#          ALREADY-TESTED predictive_ranking.labels.compute_labels with a
#          grid of (target_r, stop_r), nothing new implemented here.
# ---------------------------------------------------------------------------
print("\n--- G. R-MULTIPLE / STOP-TARGET GEOMETRY SENSITIVITY (research only) ---")
r_grid = [(1.0, 0.5), (1.5, 1.0), (2.0, 1.0), (3.0, 1.0), (1.0, 1.0)]
r_sensitivity = {}
for target_r, stop_r in r_grid:
    config = LabelConfig(target_r_multiple=target_r, stop_r_multiple=stop_r, horizon_minutes=60)
    labels = [compute_labels(r, bars_by_symbol.get(r.symbol, []), config) for r in full_signal]
    theo = [l.theoretical_success for l in labels if l is not None and l.theoretical_success is not None]
    key = f"{target_r}R_target_{stop_r}R_stop"
    if len(theo) < 10:
        print(f"  {key:20s} n={len(theo)} -- too few")
        r_sensitivity[key] = {"n": len(theo), "success_rate": None}
        continue
    rate = float(np.mean(theo))
    ci_low, ci_high = stats.binomtest(sum(theo), len(theo)).proportion_ci(confidence_level=0.95)
    print(f"  {key:20s} n={len(theo):3d}  success_rate={rate:.3f}  95% CI=[{ci_low:.3f},{ci_high:.3f}]")
    r_sensitivity[key] = {"n": len(theo), "success_rate": rate, "ci": [float(ci_low), float(ci_high)]}
results["r_multiple_sensitivity"] = r_sensitivity

# ---------------------------------------------------------------------------
# item 11. Execution cost sensitivity on raw forward return (15m primary)
# ---------------------------------------------------------------------------
print("\n--- F. EXECUTION COST SENSITIVITY (15m raw return minus assumed round-trip cost) ---")
cost_assumptions = [0.00, 0.05, 0.10, 0.20, 0.50, 1.00]
cost_sensitivity = {}
raw_15m = full_df["return_15m_pct"].dropna()
for cost in cost_assumptions:
    net = raw_15m - cost
    positive_rate = float((net > 0).mean())
    print(f"  cost={cost:.2f}%: n={len(net):3d}  mean_net={net.mean():+.3f}%  P(net>0)={positive_rate:.3f}")
    cost_sensitivity[cost] = {"n": int(len(net)), "mean_net": float(net.mean()), "p_positive": positive_rate}
results["cost_sensitivity_15m"] = cost_sensitivity

# ---------------------------------------------------------------------------
# item 7. "Big move" subset definitions, sample-size gated
# ---------------------------------------------------------------------------
print("\n--- Big-move subset definitions (15m horizon, gated by n>=30) ---")
criteria = ResearchCriteria(min_sample_size=30)
big_move_defs = {
    "A: >0.5%": full_df["return_15m_pct"] > 0.5,
    "B: >1%": full_df["return_15m_pct"] > 1.0,
    "C: >2%": full_df["return_15m_pct"] > 2.0,
}
big_move_results = {}
for name, mask in big_move_defs.items():
    n = int(mask.notna().sum())
    rate = float(mask[mask.notna()].mean()) if n >= criteria.min_sample_size else None
    print(f"  {name}: n={n}  rate={rate}")
    big_move_results[name] = {"n": n, "rate": rate}
results["big_move_definitions"] = big_move_results

# ---------------------------------------------------------------------------
# item 6. Nonlinear bucket analysis (RVOL, % move, float) via the tested
#         conditional_probability + ResearchCriteria gate
# ---------------------------------------------------------------------------
print("\n--- Nonlinear buckets: RVOL vs P(15m return > 0.5%) ---")
full_df["big_move_0_5"] = full_df["return_15m_pct"] > 0.5
rvol_buckets = [(0, 2), (2, 5), (5, 10), (10, 20), (20, float("inf"))]
bucket_results = {}
for lo, hi in rvol_buckets:
    mask = (full_df["relative_volume"] >= lo) & (full_df["relative_volume"] < hi)
    cp = conditional_probability(full_df, mask, "big_move_0_5", f"RVOL [{lo},{hi})", criteria)
    print(f"  RVOL [{lo:>4},{hi:<4}): n={cp.n:3d}  p={cp.p_success}")
    bucket_results[f"rvol_{lo}_{hi}"] = {"n": cp.n, "p": cp.p_success, "ci": [cp.ci_low, cp.ci_high]}
results["rvol_buckets"] = bucket_results

print("\n--- Nonlinear buckets: % gain at signal vs P(15m return > 0.5%) ---")
pct_buckets = [(0, 5), (5, 10), (10, 20), (20, 50), (50, float("inf"))]
pct_bucket_results = {}
for lo, hi in pct_buckets:
    mask = (full_df["percent_change"].abs() >= lo) & (full_df["percent_change"].abs() < hi)
    cp = conditional_probability(full_df, mask, "big_move_0_5", f"|%chg| [{lo},{hi})", criteria)
    print(f"  |%chg| [{lo:>4},{hi:<4}): n={cp.n:3d}  p={cp.p_success}")
    pct_bucket_results[f"pct_{lo}_{hi}"] = {"n": cp.n, "p": cp.p_success, "ci": [cp.ci_low, cp.ci_high]}
results["pct_change_buckets"] = pct_bucket_results

float_known = full_df["float_shares"].notna().sum()
print(f"\nFloat buckets: float_shares is non-null for {float_known}/{len(full_df)} rows -- "
      f"{'skipping, no fundamentals data source wired up' if float_known == 0 else 'proceeding'}")
results["float_data_available_n"] = int(float_known)

# ---------------------------------------------------------------------------
# item 8. Confirmation-strength grouping (tertile split, not invented)
# ---------------------------------------------------------------------------
print("\n--- Confirmation strength (tertile split) vs forward outcome ---")
conf_strength = full_df["confirmation_strength_pct"].dropna()
if len(conf_strength) >= 30:
    tertiles = conf_strength.quantile([1 / 3, 2 / 3]).values
    full_df["confirmation_tier"] = pd.cut(
        full_df["confirmation_strength_pct"], bins=[-np.inf, tertiles[0], tertiles[1], np.inf],
        labels=["weak", "moderate", "strong"],
    )
    conf_tier_results = {}
    for tier in ["weak", "moderate", "strong"]:
        sub = full_df[full_df["confirmation_tier"] == tier]
        vals = sub["return_15m_pct"].dropna()
        mfe_vals = sub["mfe_pct"].dropna()
        print(f"  {tier:10s}: n={len(sub):3d}  mean_15m_return={vals.mean() if len(vals) else None}  "
              f"mean_MFE%={mfe_vals.mean() if len(mfe_vals) else None}")
        conf_tier_results[tier] = {
            "n": int(len(sub)), "mean_15m_return": float(vals.mean()) if len(vals) else None,
            "mean_mfe_pct": float(mfe_vals.mean()) if len(mfe_vals) else None,
        }
    results["confirmation_strength_tiers"] = conf_tier_results
else:
    print(f"  n={len(conf_strength)} -- too few for a tertile split")
    results["confirmation_strength_tiers"] = None

# ---------------------------------------------------------------------------
# item 9. Time of day (ET)
# ---------------------------------------------------------------------------
print("\n--- Time of day (ET) ---")
time_buckets = [(7, 8), (8, 9), (9, 9.5), (9.5, 10), (10, 10.5), (10.5, 11)]
tod_results = {}
full_df["hour_frac_et"] = full_df["hour_et"] + full_df["minute_et"] / 60
for lo, hi in time_buckets:
    mask = (full_df["hour_frac_et"] >= lo) & (full_df["hour_frac_et"] < hi)
    sub = full_df[mask]
    vals = sub["return_15m_pct"].dropna()
    label = f"{int(lo):02d}:{int((lo%1)*60):02d}-{int(hi):02d}:{int((hi%1)*60):02d}"
    print(f"  {label}: n={len(sub):3d}  mean_15m_return={vals.mean() if len(vals) else None}")
    tod_results[label] = {"n": int(len(sub)), "mean_15m_return": float(vals.mean()) if len(vals) else None}
other = full_df[~full_df["hour_frac_et"].apply(lambda h: any(lo <= h < hi for lo, hi in time_buckets))]
print(f"  outside 7:00-11:00: n={len(other)}")
tod_results["outside_7_11"] = {"n": int(len(other))}
results["time_of_day"] = tod_results

# ---------------------------------------------------------------------------
# item 10. Bullish vs bearish, separately, full detail
# ---------------------------------------------------------------------------
print("\n--- Bullish vs Bearish (full detail) ---")
direction_results = {}
for direction in ("bullish", "bearish"):
    sub = full_df[full_df["direction"] == direction]
    vals = sub["return_15m_pct"].dropna()
    mfe = sub["mfe_pct"].dropna()
    mae = sub["mae_pct"].dropna()
    print(f"  {direction}: n={len(sub):3d}  mean_15m={vals.mean() if len(vals) else None}  "
          f"mean_MFE%={mfe.mean() if len(mfe) else None}  mean_MAE%={mae.mean() if len(mae) else None}")
    direction_results[direction] = {
        "n": int(len(sub)), "mean_15m_return": float(vals.mean()) if len(vals) else None,
        "mean_mfe_pct": float(mfe.mean()) if len(mfe) else None, "mean_mae_pct": float(mae.mean()) if len(mae) else None,
    }
results["by_direction"] = direction_results

json.dump(results, open(OUT_DIR / "icc_signal_quality_full.json", "w"), indent=2, default=str)
print(f"\n[full results saved] {OUT_DIR / 'icc_signal_quality_full.json'}")

# ---------------------------------------------------------------------------
# item 14/15. Walk-forward model on the EXTENDED (signal + ICC-structural)
#             feature set, chronological train/test split (out-of-sample
#             only) -- n=124 is too small for backtesting.runner.
#             run_walk_forward's multi-window scheme to say anything with
#             more than one window, so a single strict chronological
#             80/20 split is used instead, sorted by execution_timestamp.
#             Target: big_move_0_5 (P(15m return > 0.5%)) -- chosen because
#             it has full n=124 coverage and a more balanced base rate
#             (~35%) than the 2R/1R label (~8%), which would be too
#             imbalanced to fit anything meaningful at this sample size.
# ---------------------------------------------------------------------------
print("\n--- D/E. WALK-FORWARD MODEL + RANKING (chronological 80/20 split, out-of-sample only) ---")
model_df = full_df.dropna(subset=["big_move_0_5"]).sort_values("execution_timestamp").reset_index(drop=True)
split_idx = int(len(model_df) * 0.8)
train_df, test_df = model_df.iloc[:split_idx], model_df.iloc[split_idx:]
print(f"train n={len(train_df)}  test n={len(test_df)}  "
      f"(train base rate={train_df['big_move_0_5'].mean():.3f}, test base rate={test_df['big_move_0_5'].mean():.3f})")

MODEL_FEATURES = [
    "relative_volume", "percent_change", "avg_volume", "price",
    "indication_strength_pct", "correction_depth_pct", "correction_duration_bars",
    "confirmation_strength_pct", "confirmation_duration_bars", "breakout_candle_size_pct",
    "pullback_volume", "confirmation_volume", "volume_expansion_on_confirmation", "vwap_distance_pct",
]

model_results = {"train_n": len(train_df), "test_n": len(test_df),
                  "train_base_rate": float(train_df["big_move_0_5"].mean()),
                  "test_base_rate": float(test_df["big_move_0_5"].mean())}

if len(test_df) >= 5 and train_df["big_move_0_5"].nunique() > 1:
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    X_train = imputer.fit_transform(train_df[MODEL_FEATURES])
    X_test = imputer.transform(test_df[MODEL_FEATURES])
    y_train = train_df["big_move_0_5"].astype(int).values
    y_test = test_df["big_move_0_5"].astype(int).values

    for name, clf in (
        ("logistic_regression", LogisticRegression(max_iter=1000)),
        ("random_forest", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("gradient_boosting", GradientBoostingClassifier(random_state=42)),
    ):
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)
        acc = float((preds == y_test).mean())
        try:
            auc = float(roc_auc_score(y_test, probs)) if len(set(y_test)) > 1 else None
        except ValueError:
            auc = None
        brier = float(brier_score_loss(y_test, probs))
        print(f"  {name}: test_acc={acc:.3f}  auc={auc}  brier={brier:.3f}")
        model_results[name] = {"accuracy": acc, "auc": auc, "brier": brier}

        if name == "logistic_regression":
            ranked_idx = np.argsort(-probs)
            for pct, label in ((0.25, "top_25pct"), (0.10, "top_10pct"), (0.05, "top_5pct")):
                k = max(1, int(len(ranked_idx) * pct))
                top_idx = ranked_idx[:k]
                top_rate = float(y_test[top_idx].mean())
                print(f"    {label} (k={k}): out-of-sample big-move rate={top_rate:.3f} "
                      f"(vs all-test base rate={y_test.mean():.3f})")
                model_results[label] = {"k": int(k), "rate": top_rate, "all_test_base_rate": float(y_test.mean())}
else:
    print("  test set too small or train set single-class -- cannot fit/evaluate out-of-sample")
    model_results["skipped"] = True
results["walk_forward_model"] = model_results

# ---------------------------------------------------------------------------
# item 17. Small-account economics at custom equity levels, reusing the
#          TESTED execution.account_sim.simulate_account_sizes unchanged --
#          fed from the cached plan_entry/stop/target/risk_per_share, a
#          synthetic negligible-spread quote (same convention
#          backtesting.runner._BacktestMarketData already uses), and
#          RiskConfig()'s own defaults (unmodified).
# ---------------------------------------------------------------------------
print("\n--- H. SMALL ACCOUNT ECONOMICS ---")


class _StudyMarketData(MarketDataProvider):
    def __init__(self, entry_price):
        self._entry_price = entry_price

    def is_market_open(self):
        return True

    def get_quote(self, symbol):
        from datetime import datetime, timezone
        return Quote(symbol=symbol, bid_price=self._entry_price * 0.9995,
                     ask_price=self._entry_price * 1.0005, timestamp=datetime.now(timezone.utc))

    def get_latest_bar(self, symbol):
        raise MarketDataError("not used")

    def get_account_status(self):
        raise MarketDataError("not used")

    def get_recent_bars(self, symbol, minutes):
        raise MarketDataError("not used")


equity_levels = (30.0, 50.0, 100.0, 250.0, 500.0, 1_000.0, 2_500.0, 5_000.0)
risk_config = RiskConfig()  # unmodified defaults
account_results = {level: {"executable": 0, "not_executable": 0, "undetermined": 0, "too_small_n": 0} for level in equity_levels}

for row in rows:
    r = row["_record"]
    if r.plan_entry is None or r.plan_stop is None or r.plan_target is None or r.plan_risk_per_share is None:
        continue
    plan = TradePlan(
        symbol=r.symbol, direction=r.signal_direction, entry=r.plan_entry, stop=r.plan_stop, target=r.plan_target,
        risk_per_share=r.plan_risk_per_share, reward_per_share=abs(r.plan_target - r.plan_entry),
        reward_risk_ratio=abs(r.plan_target - r.plan_entry) / r.plan_risk_per_share,
        invalidation_reason="signal_quality_study small-account check",
    )
    market_data = _StudyMarketData(r.plan_entry)
    sim_results = simulate_account_sizes(
        plan, market_data, risk_config, r.avg_volume or 0.0,
        equity_levels=equity_levels, shortability_status=r.shortability_status, include_standardized_baseline=False,
    )
    for res in sim_results:
        bucket = account_results[res.equity]
        if res.executable is True:
            bucket["executable"] += 1
        elif res.executable is False:
            bucket["not_executable"] += 1
            if res.shares_affordable == 0:
                bucket["too_small_n"] += 1
        else:
            bucket["undetermined"] += 1

for level in equity_levels:
    b = account_results[level]
    total = b["executable"] + b["not_executable"] + b["undetermined"]
    print(f"  ${level:>8,.0f}: executable={b['executable']:3d}  not_executable={b['not_executable']:3d} "
          f"(too-small-to-size={b['too_small_n']:3d})  undetermined={b['undetermined']:3d}  total={total}")
results["small_account"] = {str(k): v for k, v in account_results.items()}

json.dump(results, open(OUT_DIR / "icc_signal_quality_full.json", "w"), indent=2, default=str)
print(f"\n[FINAL full results saved] {OUT_DIR / 'icc_signal_quality_full.json'}")
