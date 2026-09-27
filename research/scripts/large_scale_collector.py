"""Large-scale, memory-safe collector for the expanded historical dataset.

Processes historical-mover hits in BATCHES through fetch -> prefetch ->
backtest, discarding each batch's bars once its records are produced --
EXCEPT for records that reach signal_valid=True, whose bars are retained
(they're needed later for ICC structural feature extraction). This is a
memory-safety measure only: processing 32k+ candidates' full-day 1-minute
bars simultaneously would need several GB this machine doesn't reliably
have free. It does not change the discovery methodology, the backtest
pipeline, or which candidates get included -- every hit from Phase 1 is
still processed, just in sequential chunks.

Produces the same checkpoint shape scripts/predictive_research_experiment.py
already uses (records, bars_by_symbol, candidates) so every existing
downstream script (icc_signal_quality_study.py, signal_information_
diagnostic.py, the dataset/statistics/model sections of predictive_
research_experiment.py) can point at it unchanged.

Progress is checkpointed after EVERY batch, not just at the end, so a
crash partway through a ~15-hour run loses at most one batch's work.
"""
import argparse
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtesting import BacktestConfig, HistoricalBarFetcher, HistoricalMoverConfig, HistoricalMoverScanner  # noqa: E402
from backtesting.runner import run_backtest  # noqa: E402
from config import load_settings  # noqa: E402
from scanner import AlpacaScannerDataSource  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predictive_research_experiment import (  # noqa: E402
    CachingScannerDataSource,
    SuffixStrippingScannerDataSource,
    build_candidates_and_bars,
    prefetch_metrics_and_catalysts,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "research_checkpoints" / "large_scale_experiment"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback-days", type=int, default=250)
    parser.add_argument("--universe-size", type=int, default=None)
    parser.add_argument("--min-abs-percent-change", type=float, default=15.0)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--max-hits", type=int, default=None, help="testing only: cap total hits processed")
    args = parser.parse_args()

    settings = load_settings()

    log("PHASE 1: historical mover discovery scan (unbiased, full universe, both directions)")
    mover_config = HistoricalMoverConfig(
        lookback_trading_days=args.lookback_days, universe_size=args.universe_size,
        min_abs_percent_change=args.min_abs_percent_change,
    )
    scanner = HistoricalMoverScanner(settings)
    hits = scanner.scan(mover_config)
    log(f"discovered {len(hits)} (symbol, day) hits, {len(set(h.symbol for h in hits))} distinct symbols, "
        f"{len(set(h.day for h in hits))} distinct days")
    if args.max_hits:
        hits = hits[:args.max_hits]
        log(f"(testing) capped to {len(hits)} hits")
    with open(OUT_DIR / "all_hits.pkl", "wb") as fh:
        pickle.dump(hits, fh)

    fetcher = HistoricalBarFetcher(settings)
    scanner_ds = CachingScannerDataSource(SuffixStrippingScannerDataSource(AlpacaScannerDataSource(settings)))
    backtest_config = BacktestConfig()  # every default -- no tuning

    all_records = []
    retained_bars = {}
    batch_size = args.batch_size
    n_batches = (len(hits) + batch_size - 1) // batch_size
    overall_t0 = time.time()

    for batch_num in range(n_batches):
        batch_hits = hits[batch_num * batch_size: (batch_num + 1) * batch_size]
        log(f"=== BATCH {batch_num + 1}/{n_batches} ({len(batch_hits)} hits) ===")

        t0 = time.time()
        candidates, bars_by_symbol = build_candidates_and_bars(batch_hits, fetcher, None)
        log(f"  fetch done in {time.time() - t0:.0f}s -- {len(candidates)} usable candidates")

        t0 = time.time()
        prefetch_metrics_and_catalysts(candidates, scanner_ds, backtest_config.lookback_days, backtest_config.catalyst_lookback_hours)
        log(f"  prefetch done in {time.time() - t0:.0f}s")

        t0 = time.time()
        records, _portfolio = run_backtest(candidates, scanner_ds, bars_by_symbol, backtest_config)
        log(f"  backtest done in {time.time() - t0:.0f}s -- {len(records)} records, "
            f"{sum(1 for r in records if r.signal_valid)} signal_valid")

        all_records.extend(records)
        for r in records:
            if r.signal_valid:
                retained_bars[r.symbol] = bars_by_symbol[r.symbol]
        del bars_by_symbol  # free this batch's full bar set now that records/retained_bars are captured

        # Checkpoint after EVERY batch -- a crash later loses at most the
        # batch in progress, not the whole run.
        with open(OUT_DIR / "phase3_checkpoint.pkl", "wb") as fh:
            pickle.dump({"records": all_records, "bars_by_symbol": retained_bars, "candidates": None}, fh)
        elapsed = time.time() - overall_t0
        signal_valid_so_far = sum(1 for r in all_records if r.signal_valid)
        log(f"  checkpoint saved: {len(all_records)} total records, {signal_valid_so_far} signal_valid so far "
            f"({elapsed:.0f}s elapsed total, ~{elapsed / (batch_num + 1) * (n_batches - batch_num - 1):.0f}s remaining)")

    log(f"DONE -- {len(all_records)} total records, "
        f"{sum(1 for r in all_records if r.signal_valid)} signal_valid, "
        f"{time.time() - overall_t0:.0f}s total")


if __name__ == "__main__":
    main()
