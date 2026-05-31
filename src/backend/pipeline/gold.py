"""
GOLD LAYER — gold.py
=====================
Responsibility: scoring, flagging, human-readable reasons, aggregations.
Reads silver parquet. Adds 4 columns. Writes 4 output files.

New columns added:
  - fraud_score       float 0.0–1.0
  - is_flagged        bool  (score >= 0.35)
  - flag_reasons      str   pipe-separated human-readable reasons
  - review_status     str   "pending" — UI updates to approve/dismiss/escalate

Output files:
  - transactions_gold.parquet     full dataset, all columns
  - transactions_flagged.csv      same as parquet but CSV for frontend + submission
  - summary_by_card.csv           per-card aggregation for dashboard
  - summary_by_merchant.csv       per-merchant aggregation
"""

import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [GOLD] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT       = Path(__file__).resolve().parent.parent.parent.parent
SILVER_IN  = ROOT / "data" / "silver" / "transactions_silver.parquet"
GOLD_DIR   = ROOT / "data" / "gold"

# ── Scoring weights ────────────────────────────────────────────────────────────
# Each weight represents the contribution to fraud_score (0–1).
# Weights were chosen based on signal strength observed in EDA:
#   - velocity burst is the most reliable single signal (card testing pattern)
#   - amount ratio catches the gift card / electronics cash-out pattern
#   - odd hours and cross-border are weak alone, meaningful when stacked

WEIGHT_VELOCITY_BURST   = 0.35   # tx_count_last_30min >= 4
WEIGHT_AMOUNT_RATIO     = 0.25   # amount_vs_card_median >= 8
WEIGHT_RISKY_CATEGORY   = 0.20   # gift_card or electronics
WEIGHT_MERCHANT_BURST   = 0.10   # merchant_card_burst_24h >= 5
WEIGHT_ODD_HOURS        = 0.05   # is_odd_hours
WEIGHT_CROSS_BORDER     = 0.05   # merchant country is high-risk (CN, MX only)

FLAG_THRESHOLD          = 0.35   # minimum score to be flagged

RISKY_CATEGORIES             = {"gift_card", "electronics"}
HIGH_RISK_MERCHANT_COUNTRIES = {"CN", "MX"}  # US/GB/FR are normal cross-border, not suspicious


# ── Scoring ────────────────────────────────────────────────────────────────────

def compute_fraud_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute fraud_score as a weighted sum of binary signals.
    Each signal fires True/False; its weight is added when True.
    Score is capped at 1.0.

    Thresholds:
      - velocity:  >= 4 transactions in 30 min (catches card testing bursts)
      - amount:    >= 8× the card's own median (catches gift card cash-outs)
      - category:  gift_card or electronics (high-risk merchant categories)
      - burst:     >= 5 different cards hitting same merchant in 24h
      - odd hours: 2am–5am (weak signal, always stacked)
      - border:    cardholder country != merchant country (weak, always stacked)
    """
    log.info("Computing fraud scores...")

    score = pd.Series(0.0, index=df.index)

    velocity_fires = df["tx_count_last_30min"] >= 4
    amount_fires   = df["amount_vs_card_median"] >= 8
    category_fires = df["merchant_category"].isin(RISKY_CATEGORIES)
    burst_fires    = df["merchant_card_burst_24h"] >= 5
    odd_fires      = df["is_odd_hours"]
    border_fires   = df["merchant_country"].isin(HIGH_RISK_MERCHANT_COUNTRIES)

    score += velocity_fires * WEIGHT_VELOCITY_BURST
    score += amount_fires   * WEIGHT_AMOUNT_RATIO
    score += category_fires * WEIGHT_RISKY_CATEGORY
    score += burst_fires    * WEIGHT_MERCHANT_BURST
    score += odd_fires      * WEIGHT_ODD_HOURS
    score += border_fires   * WEIGHT_CROSS_BORDER

    df["fraud_score"] = score.clip(upper=1.0).round(2)

    log.info(f"  score > 0:    {(df['fraud_score'] > 0).sum()} transactions")
    log.info(f"  score >= 0.35: {(df['fraud_score'] >= FLAG_THRESHOLD).sum()} transactions")
    log.info(f"  score >= 0.60: {(df['fraud_score'] >= 0.60).sum()} transactions")
    log.info(f"  score = 1.00:  {(df['fraud_score'] == 1.00).sum()} transactions")
    return df


def apply_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    is_flagged = True when fraud_score >= FLAG_THRESHOLD.
    review_status starts as 'pending' for all flagged rows.
    Non-flagged rows get 'clear' so the frontend can filter easily.
    """
    df["is_flagged"]     = df["fraud_score"] >= FLAG_THRESHOLD
    df["review_status"]  = df["is_flagged"].map({True: "pending", False: "clear"})
    log.info(f"Flagged {df['is_flagged'].sum()} transactions as suspicious")
    return df


# ── Flag reasons ───────────────────────────────────────────────────────────────

def build_flag_reasons(row: pd.Series) -> str:
    """
    Build a human-readable reasons string for a single transaction.
    Returns empty string for non-flagged rows.

    Rules:
      - Odd hours and cross-border ONLY appear when at least one
        stronger signal also fired — they never stand alone as reasons.
      - The amount ratio is always shown when >= 3× (context for reviewer)
        even if it didn't cross the 8× scoring threshold.
      - Reasons are joined with ' · ' for clean UI display.
    """
    if not row["is_flagged"]:
        return ""

    reasons = []
    has_strong_signal = False

    # velocity burst
    if row["tx_count_last_30min"] >= 4:
        reasons.append(
            f"Velocity burst: {row['tx_count_last_30min']} transactions in 30 min"
        )
        has_strong_signal = True

    # amount anomaly
    if row["amount_vs_card_median"] >= 8:
        reasons.append(
            f"Amount {row['amount_vs_card_median']:.1f}× card median"
        )
        has_strong_signal = True
    elif row["amount_vs_card_median"] >= 3:
        # below scoring threshold but worth showing to reviewer for context
        reasons.append(
            f"Amount {row['amount_vs_card_median']:.1f}× card median (elevated)"
        )

    # risky category
    if row["merchant_category"] == "gift_card":
        reasons.append("Gift card purchase")
        has_strong_signal = True
    elif row["merchant_category"] == "electronics":
        reasons.append("Electronics spike")
        has_strong_signal = True

    # cross-card merchant burst
    if row["merchant_card_burst_24h"] >= 5:
        reasons.append(
            f"Merchant burst: {int(row['merchant_card_burst_24h'])} cards hit "
            f"{row['merchant_name']} in 24h"
        )
        has_strong_signal = True

    # weak signals — only append if a strong signal already fired
    if has_strong_signal:
        if row["is_odd_hours"]:
            reasons.append(f"Odd hours: {int(row['hour_of_day'])}am")
        if row["merchant_country"] in HIGH_RISK_MERCHANT_COUNTRIES:
            reasons.append(
                f"High-risk merchant country: {row['merchant_country'].upper()}"
            )

    return " · ".join(reasons) if reasons else "Anomaly score threshold exceeded"


def add_flag_reasons(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Building flag reasons...")
    df["flag_reasons"] = df.apply(build_flag_reasons, axis=1)
    flagged = df[df["is_flagged"]]
    log.info(f"  {len(flagged)} flagged rows have reasons")
    log.info(f"  sample reason: {flagged['flag_reasons'].iloc[0] if len(flagged) else 'n/a'}")
    return df


# ── Aggregations ───────────────────────────────────────────────────────────────

def build_card_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-card summary for the reviewer dashboard.
    Shows which cards are most at-risk at a glance.
    """
    log.info("Building per-card summary...")
    summary = df.groupby("card_id").agg(
        total_tx        =("transaction_id", "count"),
        flagged_tx      =("is_flagged", "sum"),
        max_fraud_score =("fraud_score", "max"),
        total_amount    =("amount", "sum"),
    ).reset_index()

    summary["flag_rate"] = (summary["flagged_tx"] / summary["total_tx"]).round(2)

    # top reason per card = the reason from the highest-scored flagged tx
    top_reasons = (
        df[df["is_flagged"]]
        .sort_values("fraud_score", ascending=False)
        .groupby("card_id")["flag_reasons"]
        .first()
        .reset_index()
        .rename(columns={"flag_reasons": "top_reason"})
    )
    summary = summary.merge(top_reasons, on="card_id", how="left")
    summary["top_reason"] = summary["top_reason"].fillna("")
    summary = summary.sort_values("max_fraud_score", ascending=False)

    log.info(f"  card summary: {len(summary)} cards, {summary['flagged_tx'].sum():.0f} total flagged tx")
    return summary


def build_merchant_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-merchant summary for the cross-card view.
    Helps the reviewer spot compromised merchants quickly.
    """
    log.info("Building per-merchant summary...")
    summary = df.groupby("merchant_name").agg(
        total_tx             =("transaction_id", "count"),
        flagged_tx           =("is_flagged", "sum"),
        unique_cards         =("card_id", "nunique"),
        unique_cards_flagged =("card_id", lambda x: df.loc[x.index[df.loc[x.index, "is_flagged"]], "card_id"].nunique()),
        avg_amount           =("amount", "mean"),
        max_fraud_score      =("fraud_score", "max"),
    ).reset_index()

    summary["flag_rate"] = (summary["flagged_tx"] / summary["total_tx"]).round(2)
    summary = summary.sort_values("flagged_tx", ascending=False)

    log.info(f"  merchant summary: {len(summary)} merchants")
    return summary


# ── Save outputs ───────────────────────────────────────────────────────────────

def save_outputs(df: pd.DataFrame, card_summary: pd.DataFrame, merchant_summary: pd.DataFrame) -> None:
    GOLD_DIR.mkdir(parents=True, exist_ok=True)

    # full dataset
    df.to_parquet(GOLD_DIR / "transactions_gold.parquet", index=False)
    log.info(f"Saved → transactions_gold.parquet  ({len(df):,} rows, {len(df.columns)} cols)")

    # CSV for frontend + submission (timestamp as string for JSON compatibility)
    df_csv = df.copy()
    df_csv["timestamp"] = df_csv["timestamp"].astype(str)
    df_csv.to_csv(GOLD_DIR / "transactions_flagged.csv", index=False)
    log.info(f"Saved → transactions_flagged.csv")

    # aggregations
    card_summary.to_csv(GOLD_DIR / "summary_by_card.csv", index=False)
    log.info(f"Saved → summary_by_card.csv  ({len(card_summary)} cards)")

    merchant_summary.to_csv(GOLD_DIR / "summary_by_merchant.csv", index=False)
    log.info(f"Saved → summary_by_merchant.csv  ({len(merchant_summary)} merchants)")


# ── Runner ─────────────────────────────────────────────────────────────────────

def run() -> pd.DataFrame:
    log.info("Starting Gold layer")
    df = pd.read_parquet(SILVER_IN)
    log.info(f"Loaded silver: {len(df):,} rows, {len(df.columns)} columns")

    df = compute_fraud_score(df)
    df = apply_flag(df)
    df = add_flag_reasons(df)

    card_summary     = build_card_summary(df)
    merchant_summary = build_merchant_summary(df)

    save_outputs(df, card_summary, merchant_summary)

    log.info("Gold complete.")
    return df


if __name__ == "__main__":
    run()