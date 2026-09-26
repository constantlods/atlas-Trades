"""A/B/C liquidity-floor experiment, reusing the checkpointed raw data from
scripts/predictive_research_experiment.py's Phase 3 (no re-fetching).

A: no execution gate at all (min_avg_daily_volume=1.0, max_participation_percent=100.0)
B: today's hard floor only (min_avg_daily_volume=50_000.0, max_participation_percent=100.0)
C: position-size-aware only, no blanket floor (min_avg_daily_volume=1.0, max_participation_percent=1.0)
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analytics import funnel_counts  # noqa: E402
from backtesting import BacktestConfig  # noqa: E402
from backtesting.experiment import ExperimentSpec, run_experiments  # noqa: E402
from config import load_settings  # noqa: E402
from predictive_ranking import build_dataset  # noqa: E402
from scanner import AlpacaScannerDataSource  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predictive_research_experiment import (  # noqa: E402
    CachingScannerDataSource,
    SuffixStrippingScannerDataSource,
    prefetch_metrics_and_catalysts,
    real_symbol,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "research_checkpoints" / "predictive_experiment"

with open(OUT_DIR / "phase3_checkpoint.pkl", "rb") as fh:
    checkpoint = pickle.load(fh)
candidates = checkpoint["candidates"]
bars_by_symbol = checkpoint["bars_by_symbol"]

settings = load_settings()
scanner_ds = CachingScannerDataSource(SuffixStrippingScannerDataSource(AlpacaScannerDataSource(settings)))
base_config = BacktestConfig()

specs = [
    ExperimentSpec(name="A_no_gate", risk_overrides={"min_avg_daily_volume": 1.0, "max_participation_percent": 100.0}),
    ExperimentSpec(name="B_hard_floor_only", risk_overrides={"min_avg_daily_volume": 50_000.0, "max_participation_percent": 100.0}),
    ExperimentSpec(name="C_participation_aware_only", risk_overrides={"min_avg_daily_volume": 1.0, "max_participation_percent": 1.0}),
]

print(f"prefetching metrics/catalyst cache for {len(candidates)} cached candidates "
      f"(bars are cached, this cache is not -- one-time cost, shared across all 3 variants)...", flush=True)
prefetch_metrics_and_catalysts(candidates, scanner_ds, base_config.lookback_days, base_config.catalyst_lookback_hours)

print(f"running A/B/C on {len(candidates)} cached candidates (no re-fetch)...", flush=True)
results = run_experiments(candidates, scanner_ds, bars_by_symbol, base_config, specs)

summary = {}
for name, (records, portfolio) in results.items():
    df = build_dataset(records, bars_by_symbol)
    theo = df["theoretical_success"].dropna().astype(bool)
    exe = df["executable_success"].dropna().astype(bool)
    funnel = funnel_counts(records)
    summary[name] = {
        "funnel": funnel,
        "theoretical_success_n": int(len(theo)),
        "theoretical_success_rate": float(theo.mean()) if len(theo) else None,
        "executable_success_n": int(len(exe)),
        "executable_success_rate": float(exe.mean()) if len(exe) else None,
        "closed_trades": len(portfolio.closed_trades),
        "net_pl": sum(t.net_pl for t in portfolio.closed_trades),
    }
    print(f"{name}: {json.dumps(summary[name], indent=2)}", flush=True)

(OUT_DIR / "liquidity_abc_experiment.json").write_text(json.dumps(summary, indent=2))
print("saved", OUT_DIR / "liquidity_abc_experiment.json", flush=True)
