"""Large-scale historical research experiment for the predictive ranking
layer (see PROJECT_HANDOFF.md and the two design plans this session
produced). Orchestration only -- every step below calls an already-built,
already-tested function from backtesting/, predictive_ranking/,
analytics/, execution/. No thresholds are tuned here for the baseline;
every config object uses its existing DEFAULT values (RossScanConfig(),
BearishScanConfig(), RiskConfig(), BacktestConfig()) -- the whole point of
this run is an untouched baseline. Filter-tightening variants (Phase 9)
are the one deliberate exception, and are reported as a SEPARATE,
clearly-labeled section, never blended into the baseline numbers.

Usage:
    .venv/bin/python scripts/predictive_research_experiment.py [--max-hits N] [--lookback-days N]

Every phase's raw output is checkpointed to
data/research_checkpoints/predictive_experiment/ as it completes, so a
failure partway through does not lose earlier phases.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analytics import (  # noqa: E402
    breakdown_by_regime,
    breakdown_by_track,
    funnel_by_account_size,
    funnel_counts,
)
from backtesting import (  # noqa: E402
    BacktestConfig,
    HistoricalBarFetcher,
    HistoricalMoverConfig,
    HistoricalMoverScanner,
    candidates_from_historical_movers,
    run_backtest,
    run_walk_forward,
)
from backtesting.research_criteria import ResearchCriteria  # noqa: E402
from bearish_strategy import BearishScanConfig  # noqa: E402
from config import load_settings  # noqa: E402
from analytics.report import compute_performance  # noqa: E402
from market_data.base import Bar, MarketDataError  # noqa: E402
from predictive_ranking import (  # noqa: E402
    ALLOWED_SOURCE_FIELDS,
    build_dataset,
    build_predictive_research_report,
    compare_baseline_vs_ranked,
    compute_feature_stability,
    compute_window_importances,
    conditional_probability,
    fit_model,
    gradient_boosting_factory,
    logistic_regression_factory,
    predict_proba,
    random_forest_factory,
)
from predictive_ranking.model import WindowMetrics  # noqa: E402
from ross_strategy import RossScanConfig  # noqa: E402
from scanner import AlpacaScannerDataSource  # noqa: E402
from sklearn.metrics import brier_score_loss, precision_score, recall_score, roc_auc_score  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "research_checkpoints" / "predictive_experiment"
OUT_DIR.mkdir(parents=True, exist_ok=True)

_SYMBOL_SEP = "__"  # joins real ticker + hit day into one synthetic "symbol" -- see build_candidates_and_bars


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}", flush=True)


def _default(obj):
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


def save_json(name: str, obj) -> None:
    path = OUT_DIR / name
    path.write_text(json.dumps(obj, default=_default, indent=2))
    log(f"saved {path} ({path.stat().st_size:,} bytes)")


class SuffixStrippingScannerDataSource:
    """Wraps the REAL AlpacaScannerDataSource so this script can key
    candidates/bars_by_symbol by "<real_ticker>__<hit_day>" instead of
    real_ticker alone. Necessary ONLY because a single ticker can be a
    historical mover on MULTIPLE different days, and run_backtest's
    bars_by_symbol dict is keyed by symbol alone -- it can hold just one
    day's bars per real key. Every method strips the suffix before
    calling the real data source with the ACTUAL ticker. Research-script
    -local wrapper, not a change to scanner/ itself.
    """

    def __init__(self, inner):
        self._inner = inner

    @staticmethod
    def _real_symbol(suffixed_symbol: str) -> str:
        return suffixed_symbol.split(_SYMBOL_SEP, 1)[0]

    def top_gainer_symbols(self, top):
        return self._inner.top_gainer_symbols(top)

    def top_loser_symbols(self, top):
        return self._inner.top_loser_symbols(top)

    def most_active_symbols(self, top):
        return self._inner.most_active_symbols(top)

    def get_daily_metrics(self, symbol, lookback_days, as_of=None):
        return self._inner.get_daily_metrics(self._real_symbol(symbol), lookback_days, as_of=as_of)

    def get_latest_catalyst(self, symbol, lookback_hours, as_of=None):
        return self._inner.get_latest_catalyst(self._real_symbol(symbol), lookback_hours, as_of=as_of)


class CachingScannerDataSource:
    """Memoizes get_daily_metrics/get_latest_catalyst by (symbol, as_of,
    lookback) -- this script calls run_backtest/run_walk_forward many
    times (Phase 3's single pass, Phase 6's three models' worth of
    walk-forward windows, Phase 7/8's per-window train fits, Phase 9's
    three filter tiers) against the SAME underlying candidate list every
    time. Without this cache, the real Alpaca API would be re-queried for
    the same (symbol, as_of) many times over across phases -- this cache
    makes every call after the first one free, cutting total runtime by
    roughly the number of phases that re-touch the same candidates.
    Research-script-local, not a change to scanner/ itself (this is NOT
    point-in-time-unsafe: the cache key already includes as_of, so it can
    never return one moment's data for a different moment's request).
    """

    def __init__(self, inner):
        self._inner = inner
        self._metrics_cache = {}
        self._catalyst_cache = {}

    def top_gainer_symbols(self, top):
        return self._inner.top_gainer_symbols(top)

    def top_loser_symbols(self, top):
        return self._inner.top_loser_symbols(top)

    def most_active_symbols(self, top):
        return self._inner.most_active_symbols(top)

    def get_daily_metrics(self, symbol, lookback_days, as_of=None):
        key = (symbol, lookback_days, as_of)
        if key not in self._metrics_cache:
            self._metrics_cache[key] = self._inner.get_daily_metrics(symbol, lookback_days, as_of=as_of)
        return self._metrics_cache[key]

    def get_latest_catalyst(self, symbol, lookback_hours, as_of=None):
        key = (symbol, lookback_hours, as_of)
        if key not in self._catalyst_cache:
            self._catalyst_cache[key] = self._inner.get_latest_catalyst(symbol, lookback_hours, as_of=as_of)
        return self._catalyst_cache[key]


def prefetch_metrics_and_catalysts(candidates, scanner_ds, lookback_days, catalyst_lookback_hours, max_workers=20):
    """Warms scanner_ds's cache CONCURRENTLY before run_backtest's own
    sequential loop touches it. run_backtest itself is a plain,
    easy-to-audit for-loop -- deliberately not modified here, per this
    experiment's own "do not modify the predictive system" instruction
    -- and ~3 sequential real HTTP round-trips per candidate would make
    a multi-thousand-candidate run take hours. This function calls the
    exact same public methods run_backtest already calls
    (get_daily_metrics/get_latest_catalyst) concurrently and ahead of
    time, so every later sequential call inside run_backtest is a cache
    hit. A per-candidate fetch failure here is silently skipped -- the
    same candidate will simply be fetched again (and handled the same
    defense-in-depth way run_backtest already handles it) when
    run_backtest's own sequential loop reaches it.
    """
    def _fetch_one(candidate):
        try:
            scanner_ds.get_daily_metrics(candidate.symbol, lookback_days, as_of=candidate.as_of)
        except Exception:
            pass
        try:
            scanner_ds.get_latest_catalyst(candidate.symbol, catalyst_lookback_hours, as_of=candidate.as_of)
        except Exception:
            pass

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_fetch_one, c) for c in candidates]
        for _ in as_completed(futures):
            done += 1
            if done % 500 == 0:
                log(f"  prefetched {done}/{len(candidates)} ({time.time() - t0:.0f}s elapsed)")
    log(f"prefetch done: {len(candidates)} candidates in {time.time() - t0:.0f}s")


def real_symbol(suffixed: str) -> str:
    return suffixed.split(_SYMBOL_SEP, 1)[0]


def asdict_shallow(record):
    return {f: getattr(record, f) for f in record.__dataclass_fields__}


def build_candidates_and_bars(hits, fetcher: HistoricalBarFetcher, max_hits):
    selected = hits[:max_hits] if max_hits else hits
    candidates = candidates_from_historical_movers(selected)

    bars_by_symbol: dict[str, list[Bar]] = {}
    suffixed_candidates = []
    fetch_failures = 0
    t0 = time.time()
    for i, (hit, candidate) in enumerate(zip(selected, candidates)):
        suffixed_symbol = f"{hit.symbol}{_SYMBOL_SEP}{hit.day.isoformat()}"
        try:
            bars = fetcher.fetch_day(hit.symbol, candidate.as_of)
        except MarketDataError:
            fetch_failures += 1
            continue
        if not bars:
            fetch_failures += 1
            continue
        bars_by_symbol[suffixed_symbol] = bars
        suffixed_candidates.append(type(candidate)(symbol=suffixed_symbol, as_of=candidate.as_of))
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            log(f"  fetched {i + 1}/{len(selected)} ({elapsed:.0f}s elapsed, {elapsed / (i + 1):.3f}s/hit)")

    log(f"intraday fetch done: {len(suffixed_candidates)} usable, {fetch_failures} failed/empty, "
        f"{time.time() - t0:.0f}s total")
    return suffixed_candidates, bars_by_symbol


def run_leakage_audit(df) -> dict:
    forbidden = {
        "outcome", "trade", "theoretical_pl", "executable_pl", "mfe_percent", "mae_percent",
        "account_size_results", "participation_rates", "rejection_chain",
        "plan_entry", "plan_stop", "plan_target", "plan_risk_per_share",
    }
    present_forbidden = forbidden.intersection(df.columns)
    return {
        "dataset_columns": list(df.columns),
        "allowed_source_fields": sorted(ALLOWED_SOURCE_FIELDS),
        "forbidden_fields_present_in_dataset": sorted(present_forbidden),
        "audit_passed": len(present_forbidden) == 0,
    }


def compute_conditional_probabilities(df, criteria) -> list:
    import pandas as pd

    conditions = [("all valid signals (base rate)", pd.Series([True] * len(df), index=df.index))]
    if "relative_volume" in df:
        conditions.append(("relative_volume > 5x", df["relative_volume"] > 5))
        conditions.append(("relative_volume > 8x", df["relative_volume"] > 8))
    if "float_shares" in df:
        conditions.append(("float_shares < 5,000,000", df["float_shares"] < 5_000_000))
    if "icc_direction" in df:
        conditions.append(("icc_direction == bullish", df["icc_direction"] == "bullish"))
        conditions.append(("icc_direction == bearish", df["icc_direction"] == "bearish"))
    if "aligned_with_trend" in df:
        conditions.append(("aligned_with_trend == True", df["aligned_with_trend"] == True))  # noqa: E712
    if "relative_volume" in df and "float_shares" in df:
        conditions.append(
            ("relative_volume > 5x AND float_shares < 5,000,000",
             (df["relative_volume"] > 5) & (df["float_shares"] < 5_000_000))
        )
    if "relative_volume" in df and "icc_direction" in df:
        conditions.append(
            ("relative_volume > 5x AND icc_direction == bullish",
             (df["relative_volume"] > 5) & (df["icc_direction"] == "bullish"))
        )

    results = []
    for label_column in ("theoretical_success", "executable_success"):
        for name, mask in conditions:
            results.append(conditional_probability(df, mask, label_column, f"{label_column} | {name}", criteria))
    return results


def run_filter_tightening(candidates, scanner_ds, bars_by_symbol) -> dict:
    tiers = {
        "baseline": (RossScanConfig(), BearishScanConfig()),
        "moderate": (
            RossScanConfig(min_relative_volume=8.0, min_percent_gain=15.0),
            BearishScanConfig(min_relative_volume=8.0, min_percent_loss=15.0),
        ),
        "strict": (
            RossScanConfig(min_relative_volume=12.0, min_percent_gain=25.0),
            BearishScanConfig(min_relative_volume=12.0, min_percent_loss=25.0),
        ),
    }
    results = {}
    for name, (ross_cfg, bearish_cfg) in tiers.items():
        cfg = replace(BacktestConfig(), ross_config=ross_cfg, bearish_config=bearish_cfg)
        records, _portfolio = run_backtest(candidates, scanner_ds, bars_by_symbol, cfg)
        df = build_dataset(records, bars_by_symbol)
        theo = df["theoretical_success"].dropna().astype(bool)
        results[name] = {
            "candidates": len(records),
            "signals": sum(1 for r in records if r.signal_valid),
            "executable": sum(1 for r in records if r.trade_valid),
            "theoretical_success_n": int(len(theo)),
            "theoretical_success_rate": float(theo.mean()) if len(theo) else None,
        }
    return results


def bearish_theoretical_vs_executable(df) -> dict:
    bearish = df[df["signal_direction"] == "bearish"]
    theo = bearish["theoretical_success"].dropna().astype(bool)
    return {
        "n_bearish": int(len(bearish)),
        "theoretical_success_n": int(len(theo)),
        "theoretical_success_rate": float(theo.mean()) if len(theo) else None,
        "executable_success_determined_count": int(bearish["executable_success"].notna().sum()),
        "executable_success_undetermined_count": int(bearish["executable_success"].isna().sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-hits", type=int, default=None)
    parser.add_argument("--lookback-days", type=int, default=250)
    parser.add_argument("--universe-size", type=int, default=3000)
    parser.add_argument("--min-abs-percent-change", type=float, default=15.0)
    args = parser.parse_args()

    settings = load_settings()

    log("PHASE 1: historical mover discovery scan (unbiased, magnitude-only, both directions)")
    mover_config = HistoricalMoverConfig(
        lookback_trading_days=args.lookback_days, universe_size=args.universe_size,
        min_abs_percent_change=args.min_abs_percent_change,
    )
    scanner = HistoricalMoverScanner(settings)
    hits = scanner.scan(mover_config)
    log(f"discovered {len(hits)} (symbol, day) hits, {len(set(h.symbol for h in hits))} distinct symbols, "
        f"{len(set(h.day for h in hits))} distinct days")
    save_json("phase1_hits.json", hits)
    save_json("phase1_config.json", {"mover_config": mover_config, "args": vars(args)})
    if not hits:
        log("no hits found -- stopping (nothing to build a dataset from)")
        return

    log("PHASE 2: fetching per-hit intraday bars")
    fetcher = HistoricalBarFetcher(settings)
    candidates, bars_by_symbol = build_candidates_and_bars(hits, fetcher, args.max_hits)
    save_json("phase2_summary.json", {
        "hits_considered": len(hits) if args.max_hits is None else min(len(hits), args.max_hits),
        "usable_candidates": len(candidates),
    })
    if not candidates:
        log("no usable candidates after intraday fetch -- stopping")
        return

    scanner_ds = CachingScannerDataSource(SuffixStrippingScannerDataSource(AlpacaScannerDataSource(settings)))
    backtest_config = BacktestConfig()  # every default -- no tuning for the baseline

    log("PHASE 2b: concurrent cache prefetch (script-local speed optimization, not a system change)")
    prefetch_metrics_and_catalysts(
        candidates, scanner_ds, backtest_config.lookback_days, backtest_config.catalyst_lookback_hours,
    )

    log("PHASE 3: running the real (UNTOUCHED, default-config) backtest pipeline")
    t0 = time.time()
    records, _portfolio = run_backtest(candidates, scanner_ds, bars_by_symbol, backtest_config)
    log(f"run_backtest done in {time.time() - t0:.0f}s -- {len(records)} records produced")

    # Checkpoint the expensive raw inputs (fetched bars + backtested
    # records) to disk RIGHT AFTER the ~3-hour fetch+backtest stage --
    # everything from here on (dataset assembly, statistics, modeling,
    # ranking, ablation, filter tiers) is cheap and re-runnable from this
    # pickle without ever re-touching the network again.
    import pickle
    with open(OUT_DIR / "phase3_checkpoint.pkl", "wb") as fh:
        pickle.dump({"records": records, "bars_by_symbol": bars_by_symbol, "candidates": candidates}, fh)
    log(f"checkpointed raw records+bars to {OUT_DIR / 'phase3_checkpoint.pkl'}")

    # funnel_counts doesn't read symbol identity, so the suffixed-vs-real
    # distinction doesn't matter for it -- kept as a plain list, not a
    # dict, so nothing here collapses/loses any record.
    fixed_records = [type(r)(**{**asdict_shallow(r), "symbol": real_symbol(r.symbol)}) for r in records]
    save_json("phase3_funnel.json", funnel_counts(fixed_records))

    log("PHASE 4: building the predictive dataset + leakage audit")
    # IMPORTANT: build_dataset (and the compute_labels bar lookup inside
    # it) must run against the SUFFIXED symbol keys, exactly as fetched --
    # 653 of 904 real tickers in this dataset are historical movers on
    # MORE THAN ONE day (one ticker 62 times), so collapsing
    # bars_by_symbol down to real-ticker keys before this point would
    # silently overwrite most symbols' bars with whichever day's fetch
    # happened to be last in iteration order, corrupting the forward-
    # looking label for every OTHER occurrence of that ticker. The real
    # ticker name is cosmetic and is only substituted AFTER every label
    # is computed, on the resulting DataFrame column -- a plain string
    # column, not a dict key, so it can never collide/overwrite a row.
    df = build_dataset(records, bars_by_symbol)
    df["symbol"] = df["symbol"].apply(real_symbol)
    log(f"dataset: {len(df)} rows")
    df.to_csv(OUT_DIR / "phase4_dataset.csv", index=False)
    save_json("phase4_leakage_audit.json", run_leakage_audit(df))

    log("PHASE 5: baseline conditional-probability statistics")
    criteria = ResearchCriteria(min_sample_size=30)
    conditional_probs = compute_conditional_probabilities(df, criteria)
    save_json("phase5_conditional_probabilities.json", conditional_probs)

    log("PHASE 6+7+8: walk-forward windows computed ONCE, all 3 models + feature stability + ranking "
        "fit against the SAME precomputed train/test data (avoids re-running run_backtest once per "
        "model/analysis, which would otherwise repeat the expensive structure/ICC computation "
        "redundantly -- a script-local efficiency choice, not a change to what is measured)")
    n = len(candidates)
    train_size = max(20, n // 8)
    test_size = max(10, n // 16)
    log(f"walk-forward windows: train_size={train_size} test_size={test_size} (n={n} total candidates)")

    windows_for_walkforward = run_walk_forward(
        candidates, train_size, test_size, scanner_ds, bars_by_symbol, backtest_config,
    )
    log(f"{len(windows_for_walkforward)} walk-forward windows")

    model_factories = {
        "logistic_regression": logistic_regression_factory,
        "random_forest": random_forest_factory,
        "gradient_boosting": gradient_boosting_factory,
    }
    window_metrics_by_model = {name: [] for name in model_factories}
    importances = []
    ranking_results = []

    for i, w in enumerate(windows_for_walkforward):
        t0 = time.time()
        train_records, _ = run_backtest(w["train"], scanner_ds, bars_by_symbol, backtest_config)
        train_df = build_dataset(train_records, bars_by_symbol)
        test_df = build_dataset(w["records"], bars_by_symbol)
        usable_test = test_df.dropna(subset=["theoretical_success"])
        signal_count = sum(1 for r in w["records"] if r.signal_valid)

        importances.append(compute_window_importances(train_df, "theoretical_success"))

        performance = compute_performance(w["portfolio"].closed_trades, starting_equity=backtest_config.starting_equity)

        for name, factory in model_factories.items():
            fitted = fit_model(train_df, "theoretical_success", factory)
            if fitted is None or len(usable_test) == 0:
                window_metrics_by_model[name].append(WindowMetrics(
                    window_index=i, train_n=len(train_df.dropna(subset=["theoretical_success"])),
                    test_n=len(usable_test), accuracy=None, precision=None, recall=None, roc_auc=None,
                    brier_score=None, net_expectancy=None, max_drawdown_percent=None, signal_count=signal_count,
                ))
                continue
            probs = predict_proba(fitted, usable_test)
            y_true = usable_test["theoretical_success"].astype(bool).astype(int).values
            preds = (probs >= 0.5).astype(int)
            window_metrics_by_model[name].append(WindowMetrics(
                window_index=i, train_n=len(train_df.dropna(subset=["theoretical_success"])),
                test_n=len(usable_test), accuracy=float((preds == y_true).mean()),
                precision=float(precision_score(y_true, preds, zero_division=0)),
                recall=float(recall_score(y_true, preds, zero_division=0)),
                roc_auc=float(roc_auc_score(y_true, probs)) if len(set(y_true)) > 1 else None,
                brier_score=float(brier_score_loss(y_true, probs)),
                net_expectancy=performance.expectancy, max_drawdown_percent=performance.max_drawdown_percent,
                signal_count=signal_count,
            ))
            if name == "logistic_regression":
                comparison = compare_baseline_vs_ranked(
                    w["records"], fitted, starting_equity=backtest_config.starting_equity, top_k_values=(1, 3, 5, 10),
                )
                ranking_results.append({"window_index": i, "comparison": comparison})

        log(f"  window {i + 1}/{len(windows_for_walkforward)} done in {time.time() - t0:.0f}s "
            f"(train_n={len(w['train'])}, test_n={len(w['records'])})")

    stability = compute_feature_stability(importances)
    save_json("phase6_window_metrics.json", window_metrics_by_model)
    save_json("phase7_feature_stability.json", stability)
    save_json("phase8_ranking_comparison.json", ranking_results)

    log("PHASE 9: filter-tightening simulation (baseline / moderate / strict) -- SEPARATE from the baseline above")
    filter_results = run_filter_tightening(candidates, scanner_ds, bars_by_symbol)
    save_json("phase9_filter_tightening.json", filter_results)

    log("PHASE 10: small-account / track / regime / bearish breakdowns")
    save_json("phase10_funnel_by_account_size.json", funnel_by_account_size(fixed_records))
    save_json("phase10_breakdown_by_track.json", breakdown_by_track(fixed_records))
    save_json("phase10_breakdown_by_regime.json", breakdown_by_regime(fixed_records))
    save_json("phase10_bearish_theoretical_vs_executable.json", bearish_theoretical_vs_executable(df))

    log("Building final narrative report")
    report_text = build_predictive_research_report(
        df=df, conditional_probabilities=conditional_probs,
        window_metrics_by_model=window_metrics_by_model, feature_stability=stability,
        ablation_results={}, ranking_comparison=None, criteria=criteria,
    )
    (OUT_DIR / "FINAL_REPORT.txt").write_text(report_text)
    log(f"saved {OUT_DIR / 'FINAL_REPORT.txt'}")
    log("DONE")


if __name__ == "__main__":
    main()
