"""
BRONZE LAYER — bronze.py
============================
Responsibility: ingest the raw CSV and produce a clean, typed parquet file.
No business logic. No feature engineering. No fraud detection.

What we do here:
  1. Load the raw CSV
  2. Cast every column to its correct type
  3. Add one null tag Silver actually needs
  4. Save to data/bronze/transactions_bronze.parquet
"""

import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [BRONZE] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# src/backend/pipeline/bronze.py → go up 3 levels to reach project root
ROOT       = Path(__file__).resolve().parent.parent.parent.parent
RAW_CSV    = ROOT / "data" / "raw" / "transactions.csv"
UPLOAD_CSV = ROOT / "data" / "bronze" / "raw_upload.csv"
BRONZE_OUT = ROOT / "data" / "bronze" / "transactions_bronze.parquet"


def get_input_csv() -> Path:
    if UPLOAD_CSV.exists():
        log.info(f"Found uploaded CSV at {UPLOAD_CSV} — using it for Bronze ingest")
        return UPLOAD_CSV
    return RAW_CSV


def load_raw(path: Path) -> pd.DataFrame:
    log.info(f"Reading raw CSV from {path}")
    df = pd.read_csv(path, dtype=str)
    log.info(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
    return df


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Casting column types...")
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
    df["amount"]    = df["amount"].str.replace(r"[^\d.]", "", regex=True).astype(float)

    for col in ["channel", "merchant_category", "cardholder_country", "merchant_country"]:
        df[col] = df[col].str.strip().str.lower()

    for col in ["transaction_id", "card_id", "merchant_name", "device_id", "ip_address"]:
        df[col] = df[col].str.strip()

    return df


def tag_expected_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    device_id and ip_address are null for in_person and atm rows — that's by design.
    Silver needs this flag so it doesn't treat these nulls as suspicious.
    """
    df["device_ip_null_expected"] = df["channel"].isin(["in_person", "atm"])
    log.info(f"Tagged {df['device_ip_null_expected'].sum()} rows where null device/IP is expected")
    return df


def run() -> pd.DataFrame:
    log.info("Starting Bronze layer")
    df = load_raw(get_input_csv())
    df = cast_types(df)
    df = tag_expected_nulls(df)
    BRONZE_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(BRONZE_OUT, index=False)
    log.info(f"Bronze saved → {BRONZE_OUT}  ({len(df):,} rows, {len(df.columns)} columns)")
    return df


if __name__ == "__main__":
    run()