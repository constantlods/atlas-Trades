"""Calibrated probability + ranking study on the expanded dataset
(32,348 candidates, 770 causally-valid ICC signals). Reuses the same
checkpointed bars/records (zero new API calls) and the same
ICC-structural-feature extraction approach as scripts/
icc_signal_quality_study.py (duplicated here in small form rather than
imported, since that module runs its own analysis at import time).

SIGNAL features only feed the model -- spread_percent, execution_tier,
expected_net_r, and shortability_status are EXECUTION features and are
tracked/reported separately, never as model inputs (research request
item 1).

All calibration/ranking results come from an EXPANDING-WINDOW walk-
forward split (train on everything chronologically before the fold,
test on the fold, never the reverse) -- pooled out-of-sample predictions
across folds are what get bucketed for calibration, not single-fold or
in-sample numbers.
"""
import json
import pickle
import sys
from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from execution.account_sim import simulate_account_sizes  # noqa: E402
from icc_strategy import detect_icc_setups  # noqa: E402
from market_data.base import MarketDataError, MarketDataProvider, Quote  # noqa: E402
from market_structure import MarketStructureConfig, analyze_structure  # noqa: E402
from market_structure.vwap import session_vwap  # noqa: E402
from predictive_ranking.labels import LabelConfig, compute_labels  # noqa: E402
from backtesting.runner import _earliest_confirmed  # noqa: E402
from risk_manager import RiskConfig  # noqa: E402
from trade_plan.models import TradePlan  # noqa: E402

IN_DIR = Path(__file__).resolve().parent.parent / "data" / "research_checkpoints" / "large_scale_experiment"
OUT_DIR = IN_DIR
EASTERN = ZoneInfo("America/New_York")
STRUCTURE_CONFIG = MarketStructureConfig()

with open(IN_DIR / "phase3_checkpoint.pkl", "rb") as fh:
    checkpoint = pickle.load(fh)
records = checkpoint["records"]
bars_by_symbol = checkpoint["bars_by_symbol"]
full_signal = [r for r in records if r.signal_valid]
print(f"Total records: {len(records)}   FULL_SIGNAL: {len(full_signal)}")


# ---------------------------------------------------------------------------
# ICC structural feature extraction (same logic as icc_signal_quality_study.py)
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
    return _earliest_confirmed(direction_agreeing or tradeable), bars


def extract_icc_features(setup, bars):
    ind, corr, cont = setup.indication, setup.correction, setup.continuation
    vwap = session_vwap(bars)
    indication_strength = abs(ind.breakout_close - ind.level) / ind.level * 100 if ind.level else None
    correction_depth = abs(ind.level - corr.extreme_point.price) / ind.level * 100 if ind.level else None
    confirmation_strength = (
        abs(cont.confirmation_close - corr.reaction_level) / corr.reaction_level * 100 if corr.reaction_level else None
    )
    breakout_bar = bars[ind.bar_index]
    breakout_candle_size_pct = (breakout_bar.high - breakout_bar.low) / breakout_bar.low * 100 if breakout_bar.low else None
    correction_window = bars[ind.bar_index:corr.extreme_point.bar_index + 1]
    confirmation_window = bars[corr.extreme_point.bar_index + 1:cont.bar_index + 1]
    pullback_volume = float(np.mean([b.volume for b in correction_window])) if correction_window else None
    confirmation_volume = float(np.mean([b.volume for b in confirmation_window])) if confirmation_window else None
    volume_expansion = confirmation_volume / pullback_volume if pullback_volume and confirmation_volume else None
    vwap_at_confirmation = vwap[cont.bar_index] if cont.bar_index < len(vwap) else None
    vwap_distance_pct = (
        (cont.confirmation_close - vwap_at_confirmation) / vwap_at_confirmation * 100 if vwap_at_confirmation else None
    )
    et_time = cont.timestamp.astimezone(EASTERN)
    return {
        "indication_strength_pct": indication_strength, "correction_depth_pct": correction_depth,
        "correction_duration_bars": corr.extreme_point.bar_index - ind.bar_index,
        "confirmation_strength_pct": confirmation_strength,
        "confirmation_duration_bars": cont.bar_index - corr.extreme_point.bar_index,
        "breakout_candle_size_pct": breakout_candle_size_pct, "pullback_volume": pullback_volume,
        "confirmation_volume": confirmation_volume, "volume_expansion_on_confirmation": volume_expansion,
        "vwap_distance_pct": vwap_distance_pct, "hour_et": et_time.hour, "minute_et": et_time.minute,
    }


rows = []
for r in full_signal:
    setup, bars = find_matching_setup(r)
    if setup is None:
        continue
    feats = extract_icc_features(setup, bars)
    row = {
        "symbol": r.symbol.split("__")[0], "direction": r.signal_direction,
        "relative_volume": r.relative_volume, "percent_change": r.percent_change,
        "avg_volume": r.avg_volume, "price": r.price, "has_catalyst": r.has_catalyst,
        "market_regime": r.market_regime, "spread_percent": r.spread_percent,
        "execution_tier": r.execution_tier, "expected_net_r": r.expected_net_r,
        "shortability_status": r.shortability_status,
        "plan_entry": r.plan_entry, "plan_stop": r.plan_stop, "plan_target": r.plan_target,
        "plan_risk_per_share": r.plan_risk_per_share, "execution_timestamp": r.execution_timestamp,
        "avg_daily_volume_for_sizing": r.avg_volume,
    }
    row.update(feats)
    row["_record"] = r
    rows.append(row)

df = pd.DataFrame(rows)
print(f"ICC features extracted for {len(df)}/{len(full_signal)} FULL_SIGNAL records")
print(f"market_regime populated for {df['market_regime'].notna().sum()}/{len(df)} rows (0 expected -- not wired to a live SPY fetch)")

# ---------------------------------------------------------------------------
# Primary label: +1R before -1R within 15 minutes (research request's own
# default framing), plus the alternative R-targets/horizons grid.
# ---------------------------------------------------------------------------
PRIMARY_CONFIG = LabelConfig(target_r_multiple=1.0, stop_r_multiple=1.0, horizon_minutes=15)
primary_labels = [compute_labels(row["_record"], bars_by_symbol.get(row["_record"].symbol, []), PRIMARY_CONFIG) for _, row in df.iterrows()]
df["theoretical_success"] = [l.theoretical_success if l else None for l in primary_labels]
df["executable_success"] = [l.executable_success if l else None for l in primary_labels]
df["mfe_r"] = [l.mfe_r if l else None for l in primary_labels]
df["mae_r"] = [l.mae_r if l else None for l in primary_labels]

usable = df.dropna(subset=["theoretical_success"]).sort_values("execution_timestamp").reset_index(drop=True)
print(f"\nUsable for modeling (primary label computable): {len(usable)}")
print(f"Primary label (+1R/-1R @ 15m) base rate: {usable['theoretical_success'].astype(bool).mean():.3f}")

SIGNAL_FEATURES = [
    "relative_volume", "percent_change", "avg_volume", "price",
    "indication_strength_pct", "correction_depth_pct", "correction_duration_bars",
    "confirmation_strength_pct", "confirmation_duration_bars", "breakout_candle_size_pct",
    "pullback_volume", "confirmation_volume", "volume_expansion_on_confirmation",
    "vwap_distance_pct", "hour_et",
]
EXECUTION_FEATURES = ["spread_percent", "execution_tier", "expected_net_r", "shortability_status"]
print(f"\nSIGNAL features used by the model ({len(SIGNAL_FEATURES)}): {SIGNAL_FEATURES}")
print(f"EXECUTION features tracked separately, NEVER fed to the model: {EXECUTION_FEATURES}")

results = {"n_usable": len(usable), "base_rate": float(usable["theoretical_success"].astype(bool).mean())}

# ---------------------------------------------------------------------------
# Expanding-window walk-forward: train on everything chronologically before
# each fold, test on the fold, pool out-of-sample predictions across folds.
# ---------------------------------------------------------------------------
INITIAL_TRAIN_FRAC = 0.30
N_FOLDS = 6
n = len(usable)
initial_train = int(n * INITIAL_TRAIN_FRAC)
remaining = n - initial_train
fold_size = max(1, remaining // N_FOLDS)

folds = []
start = initial_train
while start < n:
    end = min(start + fold_size, n)
    folds.append((0, start, start, end))  # (train_start, train_end, test_start, test_end)
    start = end
print(f"\nWalk-forward folds: {len(folds)}, initial_train={initial_train}, ~{fold_size}/fold")

y_all = usable["theoretical_success"].astype(bool).astype(int).values
imputer_template = lambda: SimpleImputer(strategy="median", keep_empty_features=True)

pooled = {"logistic_raw": [], "logistic_calibrated": [], "random_forest_calibrated": []}

for fold_idx, (_, train_end, test_start, test_end) in enumerate(folds):
    train_df = usable.iloc[:train_end]
    test_df = usable.iloc[test_start:test_end]
    if train_df["theoretical_success"].astype(bool).nunique() < 2 or len(test_df) == 0:
        print(f"  fold {fold_idx}: train_n={len(train_df)} test_n={len(test_df)} -- skipped (single class or empty test)")
        continue

    imputer = imputer_template()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(imputer.fit_transform(train_df[SIGNAL_FEATURES]))
    X_test = scaler.transform(imputer.transform(test_df[SIGNAL_FEATURES]))
    y_train = train_df["theoretical_success"].astype(bool).astype(int).values
    y_test = test_df["theoretical_success"].astype(bool).astype(int).values

    models = {"logistic_raw": LogisticRegression(max_iter=1000)}
    cv_folds = min(3, int(np.bincount(y_train).min())) if len(np.bincount(y_train)) > 1 else 0
    if cv_folds >= 2:
        models["logistic_calibrated"] = CalibratedClassifierCV(LogisticRegression(max_iter=1000), method="sigmoid", cv=cv_folds)
        models["random_forest_calibrated"] = CalibratedClassifierCV(
            RandomForestClassifier(n_estimators=100, random_state=42), method="isotonic", cv=cv_folds
        )

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)[:, 1]
        except Exception as exc:
            print(f"  fold {fold_idx} [{name}]: fit/predict failed ({exc}) -- skipped")
            continue
        for i, (prob, actual) in enumerate(zip(probs, y_test)):
            pooled[name].append({
                "fold": fold_idx, "prob": float(prob), "actual": int(actual),
                "direction": test_df.iloc[i]["direction"], "row_idx": int(test_df.index[i]),
            })
    print(f"  fold {fold_idx}: train_n={len(train_df)} test_n={len(test_df)} test_base_rate={y_test.mean():.3f}")

for name, preds in pooled.items():
    print(f"\n{name}: {len(preds)} pooled out-of-sample predictions")
results["pooled_oos_counts"] = {name: len(preds) for name, preds in pooled.items()}


def wilson_ci(successes, n_, z=1.96):
    if n_ == 0:
        return (0.0, 0.0)
    phat = successes / n_
    z2 = z * z
    denom = 1 + z2 / n_
    center = (phat + z2 / (2 * n_)) / denom
    margin = (z * ((phat * (1 - phat) / n_ + z2 / (4 * n_ * n_)) ** 0.5)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


MIN_BUCKET_N = 15

# ---------------------------------------------------------------------------
# CALIBRATION: 10 probability buckets, pooled out-of-sample, per model
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("CALIBRATION (pooled out-of-sample predictions, 10 buckets)")
print("=" * 72)
calibration_results = {}
bucket_edges = [i / 10 for i in range(11)]
for name, preds in pooled.items():
    if not preds:
        continue
    pdf = pd.DataFrame(preds)
    print(f"\n--- {name} ---")
    model_cal = []
    for i in range(10):
        lo, hi = bucket_edges[i], bucket_edges[i + 1]
        mask = (pdf["prob"] >= lo) & (pdf["prob"] < hi if hi < 1.0 else pdf["prob"] <= hi)
        sub = pdf[mask]
        n_bucket = len(sub)
        if n_bucket < MIN_BUCKET_N:
            print(f"  [{lo:.1f},{hi:.1f}): n={n_bucket:3d} -- below reliability floor ({MIN_BUCKET_N}), not reported")
            model_cal.append({"range": [lo, hi], "n": n_bucket, "predicted_mean": None, "actual_rate": None})
            continue
        actual_rate = float(sub["actual"].mean())
        predicted_mean = float(sub["prob"].mean())
        ci_low, ci_high = wilson_ci(int(sub["actual"].sum()), n_bucket)
        cal_error = abs(predicted_mean - actual_rate)
        print(f"  [{lo:.1f},{hi:.1f}): n={n_bucket:3d}  predicted={predicted_mean:.3f}  actual={actual_rate:.3f}  "
              f"95% CI=[{ci_low:.3f},{ci_high:.3f}]  |calib_error|={cal_error:.3f}")
        model_cal.append({"range": [lo, hi], "n": n_bucket, "predicted_mean": predicted_mean,
                           "actual_rate": actual_rate, "ci": [ci_low, ci_high], "calibration_error": cal_error})
    calibration_results[name] = model_cal
results["calibration"] = calibration_results

# ---------------------------------------------------------------------------
# RANKING: within each fold, rank test-fold predictions by probability,
# take top-k / top-pct, aggregate across folds. Out-of-sample by
# construction (fold ranks only ever use that fold's own test predictions).
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("RANKING (top-k / top-pct vs ALL, aggregated across out-of-sample folds)")
print("=" * 72)
ranking_results = {}
for name, preds in pooled.items():
    if not preds:
        continue
    pdf = pd.DataFrame(preds)
    print(f"\n--- {name} ---")
    all_n, all_rate = len(pdf), float(pdf["actual"].mean())
    ci = wilson_ci(int(pdf["actual"].sum()), all_n)
    print(f"  ALL:        n={all_n:4d}  actual_rate={all_rate:.3f}  95% CI=[{ci[0]:.3f},{ci[1]:.3f}]")
    model_ranking = {"all": {"n": all_n, "rate": all_rate, "ci": list(ci)}}

    for k in (1, 3, 5, 10, 20):
        picks = []
        for fold_id in pdf["fold"].unique():
            fold_df = pdf[pdf["fold"] == fold_id].sort_values("prob", ascending=False)
            picks.append(fold_df.head(k))
        picked = pd.concat(picks) if picks else pd.DataFrame(columns=pdf.columns)
        n_picked = len(picked)
        rate = float(picked["actual"].mean()) if n_picked else None
        ci_p = wilson_ci(int(picked["actual"].sum()), n_picked) if n_picked >= MIN_BUCKET_N else (None, None)
        flag = "" if n_picked >= MIN_BUCKET_N else "  (below reliability floor)"
        print(f"  TOP {k:<3d} (per fold): n={n_picked:4d}  actual_rate={rate}{flag}")
        model_ranking[f"top_{k}_per_fold"] = {"n": n_picked, "rate": rate, "ci": list(ci_p)}

    for pct, label in ((0.25, "top_25pct"), (0.10, "top_10pct"), (0.05, "top_5pct")):
        picks = []
        for fold_id in pdf["fold"].unique():
            fold_df = pdf[pdf["fold"] == fold_id].sort_values("prob", ascending=False)
            k = max(1, int(len(fold_df) * pct))
            picks.append(fold_df.head(k))
        picked = pd.concat(picks) if picks else pd.DataFrame(columns=pdf.columns)
        n_picked = len(picked)
        rate = float(picked["actual"].mean()) if n_picked else None
        ci_p = wilson_ci(int(picked["actual"].sum()), n_picked) if n_picked >= MIN_BUCKET_N else (None, None)
        flag = "" if n_picked >= MIN_BUCKET_N else "  (below reliability floor)"
        print(f"  {label:<12s}: n={n_picked:4d}  actual_rate={rate}{flag}")
        model_ranking[label] = {"n": n_picked, "rate": rate, "ci": list(ci_p)}
    ranking_results[name] = model_ranking
results["ranking"] = ranking_results

# ---------------------------------------------------------------------------
# ABSTAIN TIERS (configurable thresholds, evaluated out-of-sample on the
# pooled predictions -- NOT tuned to look good)
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("ABSTAIN TIERS (fixed, pre-specified thresholds -- NOT tuned post-hoc)")
print("=" * 72)
TIER_THRESHOLDS = [(0.0, 0.15, "NO_TRADE"), (0.15, 0.25, "LOW"), (0.25, 0.40, "MEDIUM"), (0.40, 1.01, "HIGH")]
tier_results = {}
for name, preds in pooled.items():
    if not preds:
        continue
    pdf = pd.DataFrame(preds)
    print(f"\n--- {name} ---")
    model_tiers = {}
    for lo, hi, label in TIER_THRESHOLDS:
        sub = pdf[(pdf["prob"] >= lo) & (pdf["prob"] < hi)]
        n_tier = len(sub)
        if n_tier < MIN_BUCKET_N:
            print(f"  {label:<9s} [{lo:.2f},{hi:.2f}): n={n_tier:4d} -- below reliability floor")
            model_tiers[label] = {"n": n_tier, "rate": None}
            continue
        rate = float(sub["actual"].mean())
        ci = wilson_ci(int(sub["actual"].sum()), n_tier)
        print(f"  {label:<9s} [{lo:.2f},{hi:.2f}): n={n_tier:4d}  actual_rate={rate:.3f}  95% CI=[{ci[0]:.3f},{ci[1]:.3f}]")
        model_tiers[label] = {"n": n_tier, "rate": rate, "ci": list(ci)}
    tier_results[name] = model_tiers
results["abstain_tiers"] = tier_results

json.dump(results, open(OUT_DIR / "calibrated_ranking_partial.json", "w"), indent=2, default=str)
print(f"\n[checkpoint saved] {OUT_DIR / 'calibrated_ranking_partial.json'}")

# ---------------------------------------------------------------------------
# ECONOMIC VALUE per abstain tier (random_forest_calibrated -- the model
# whose ranking showed the cleanest monotonic pattern above)
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("ECONOMIC VALUE per tier (random_forest_calibrated, MFE/MAE in R, cost-sensitivity)")
print("=" * 72)
rf_preds = pd.DataFrame(pooled["random_forest_calibrated"])
econ_results = {}
if len(rf_preds):
    for lo, hi, label in TIER_THRESHOLDS:
        sub = rf_preds[(rf_preds["prob"] >= lo) & (rf_preds["prob"] < hi)]
        if len(sub) < MIN_BUCKET_N:
            print(f"  {label:<9s}: n={len(sub)} -- below reliability floor")
            econ_results[label] = {"n": len(sub)}
            continue
        row_data = usable.loc[sub["row_idx"]]
        mfe = row_data["mfe_r"].dropna()
        mae = row_data["mae_r"].dropna()
        spread = row_data["spread_percent"].dropna()
        print(f"  {label:<9s}: n={len(sub):4d}  mean_MFE_R={mfe.mean():.3f}  mean_MAE_R={mae.mean():.3f}  "
              f"mean_spread%={spread.mean() if len(spread) else None}")
        econ_results[label] = {
            "n": len(sub), "mean_mfe_r": float(mfe.mean()) if len(mfe) else None,
            "mean_mae_r": float(mae.mean()) if len(mae) else None,
            "mean_spread_pct": float(spread.mean()) if len(spread) else None,
        }
results["economic_value_by_tier_rf"] = econ_results

# ---------------------------------------------------------------------------
# BULLISH vs BEARISH -- pooled OOS ranking, random_forest_calibrated only
# (the cleanest model above), kept SEPARATE per research request item 15.
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("BULLISH vs BEARISH (random_forest_calibrated, pooled OOS)")
print("=" * 72)
direction_results = {}
for direction in ("bullish", "bearish"):
    sub = rf_preds[rf_preds["direction"] == direction]
    n_dir = len(sub)
    if n_dir < MIN_BUCKET_N:
        print(f"  {direction}: n={n_dir} -- below reliability floor")
        direction_results[direction] = {"n": n_dir}
        continue
    rate = float(sub["actual"].mean())
    ci = wilson_ci(int(sub["actual"].sum()), n_dir)
    # top 20% within this direction only
    top20 = sub.sort_values("prob", ascending=False).head(max(1, int(n_dir * 0.2)))
    top20_rate = float(top20["actual"].mean())
    print(f"  {direction}: n={n_dir:4d}  all_rate={rate:.3f}  95% CI=[{ci[0]:.3f},{ci[1]:.3f}]  "
          f"top_20pct (n={len(top20)}) rate={top20_rate:.3f}")
    direction_results[direction] = {"n": n_dir, "all_rate": rate, "ci": list(ci),
                                     "top_20pct_n": len(top20), "top_20pct_rate": top20_rate}
results["by_direction_rf"] = direction_results

# Bearish executable success reminder (shortability tri-state, never coerced)
bearish_rows = usable[usable["direction"] == "bearish"]
exec_determined = bearish_rows["executable_success"].notna().sum()
exec_undetermined = bearish_rows["executable_success"].isna().sum()
print(f"\nBearish executable_success: {exec_determined} determined, {exec_undetermined} undetermined "
      f"(shortability unknown, never coerced to true/false)")
results["bearish_executable_status"] = {"determined": int(exec_determined), "undetermined": int(exec_undetermined)}

# ---------------------------------------------------------------------------
# SMALL ACCOUNT by abstain tier (random_forest_calibrated)
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("SMALL ACCOUNT ECONOMICS by tier (random_forest_calibrated)")
print("=" * 72)


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


equity_levels = (30.0, 50.0, 100.0, 250.0, 500.0, 1_000.0)
risk_config = RiskConfig()
small_account_results = {}
for lo, hi, label in TIER_THRESHOLDS:
    sub = rf_preds[(rf_preds["prob"] >= lo) & (rf_preds["prob"] < hi)]
    if len(sub) < MIN_BUCKET_N:
        continue
    tier_rows = usable.loc[sub["row_idx"]]
    tier_account = {level: {"executable": 0, "not_executable": 0, "undetermined": 0} for level in equity_levels}
    for _, row in tier_rows.iterrows():
        r = row["_record"]
        if r.plan_entry is None or r.plan_risk_per_share is None:
            continue
        plan = TradePlan(
            symbol=r.symbol, direction=r.signal_direction, entry=r.plan_entry, stop=r.plan_stop,
            target=r.plan_target, risk_per_share=r.plan_risk_per_share,
            reward_per_share=abs(r.plan_target - r.plan_entry),
            reward_risk_ratio=abs(r.plan_target - r.plan_entry) / r.plan_risk_per_share,
            invalidation_reason="calibrated_ranking_study small-account check",
        )
        sim = simulate_account_sizes(
            plan, _StudyMarketData(r.plan_entry), risk_config, r.avg_volume or 0.0,
            equity_levels=equity_levels, shortability_status=r.shortability_status, include_standardized_baseline=False,
        )
        for res in sim:
            bucket = tier_account[res.equity]
            if res.executable is True:
                bucket["executable"] += 1
            elif res.executable is False:
                bucket["not_executable"] += 1
            else:
                bucket["undetermined"] += 1
    print(f"  {label} (n={len(sub)}):")
    for level in equity_levels:
        b = tier_account[level]
        print(f"    ${level:>6,.0f}: executable={b['executable']:3d}  not_executable={b['not_executable']:3d}  undetermined={b['undetermined']:3d}")
    small_account_results[label] = {str(k): v for k, v in tier_account.items()}
results["small_account_by_tier"] = small_account_results

json.dump(results, open(OUT_DIR / "calibrated_ranking_full.json", "w"), indent=2, default=str)
print(f"\n[FINAL full results saved] {OUT_DIR / 'calibrated_ranking_full.json'}")
