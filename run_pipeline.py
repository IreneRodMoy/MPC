"""
run_pipeline.py
================
One command to run the full Bronze → Silver → Gold pipeline.

Usage:
    python run_pipeline.py

What it does:
    1. Bronze — ingest raw CSV, cast types, tag nulls
    2. Silver — feature engineering (velocity, amount ratio, device, burst, border)
    3. Gold   — scoring, flag reasons, aggregations, output CSV

Output files (all in data/gold/):
    - transactions_gold.parquet
    - transactions_flagged.csv      ← frontend + submission deliverable
    - summary_by_card.csv
    - summary_by_merchant.csv
"""

import importlib.util
import time
import logging
from pathlib import Path


# ── Load pipeline modules from their actual location ──────────────────────────

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

PIPELINE     = Path(__file__).resolve().parent / "src" / "backend" / "pipeline"
bronze_layer = _load("bronze", PIPELINE / "bronze.py")
silver_layer = _load("silver", PIPELINE / "silver.py")
gold_layer   = _load("gold",   PIPELINE / "gold.py")


# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [PIPELINE] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Runner ────────────────────────────────────────────────────────────────────

def run():
    log.info("=" * 55)
    log.info("Fraud Hunter Pipeline — starting")
    log.info("=" * 55)

    total_start = time.time()

    # ── Bronze ────────────────────────────────────────────
    log.info("Step 1/3 — Bronze")
    t = time.time()
    bronze_layer.run()
    log.info(f"Bronze done in {time.time() - t:.1f}s")

    # ── Silver ────────────────────────────────────────────
    log.info("Step 2/3 — Silver")
    t = time.time()
    silver_layer.run()
    log.info(f"Silver done in {time.time() - t:.1f}s")

    # ── Gold ──────────────────────────────────────────────
    log.info("Step 3/3 — Gold")
    t = time.time()
    gold_layer.run()
    log.info(f"Gold done in {time.time() - t:.1f}s")

    # ── Summary ───────────────────────────────────────────
    elapsed = time.time() - total_start
    log.info("=" * 55)
    log.info(f"Pipeline complete in {elapsed:.1f}s")
    log.info("Output → data/gold/transactions_flagged.csv")
    log.info("=" * 55)


if __name__ == "__main__":
    run()