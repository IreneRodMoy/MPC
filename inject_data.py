"""
inject_data.py
==============
Connects the Gold pipeline output to the frontend (67.html).
Replaces ONLY the mock txData array — zero UI changes.

Usage:
    python inject_data.py

What it does:
    1. Reads data/gold/transactions_flagged.csv
    2. Maps each flagged row to the exact txData shape the frontend expects
    3. Surgically replaces the const txData = [...] block in 67.html
    4. Writes the updated file back — all styling, layout, charts untouched

Run this once after run_pipeline.py completes.
"""

import pandas as pd
import json
import re
from pathlib import Path

ROOT     = Path(__file__).resolve().parent
CSV_PATH = ROOT /"data" / "gold" / "transactions_flagged.csv"
HTML_PATH = ROOT / "src" / "frontend" / "67.html"


# ── Score → UI mapping ────────────────────────────────────────────────────────
# The frontend uses "pattern" as a human label and "badgeClass" for color.
# We map fraud_score into three tiers that match the existing badge classes.

def pattern_label(row: pd.Series) -> str:
    """
    Derive a human-readable pattern name from flag_reasons.
    Priority: velocity burst > gift card > electronics > amount spike > merchant burst.
    Falls back to a risk-tier label so the badge always has text.
    """
    reasons = str(row.get("flag_reasons", ""))
    if "Velocity burst" in reasons:
        return "Velocity Burst"
    if "Gift card" in reasons:
        return "Gift Card Fraud"
    if "Electronics spike" in reasons:
        return "Electronics Spike"
    if "Amount" in reasons and float(row.get("fraud_score", 0)) >= 0.40:
        return "Amount Anomaly"
    if "Merchant burst" in reasons:
        return "Merchant Burst"
    score = float(row.get("fraud_score", 0))
    if score >= 0.50:
        return "High Risk"
    if score >= 0.40:
        return "Medium Risk"
    return "Elevated Risk"


def badge_class(score: float) -> str:
    """Map fraud score to existing badge CSS classes — no new classes added."""
    if score >= 0.50:
        return "badge-red"
    if score >= 0.40:
        return "badge-amber"
    return "badge-blue"


def score_to_100(score: float, min_score: float, max_score: float) -> int:
    if max_score == min_score:
        return 50
    normalized = (score - min_score) / (max_score - min_score)
    return min(100, max(0, round(normalized * 100)))


def format_timestamp(ts_str: str) -> str:
    """Return HH:MM:SS from ISO timestamp string."""
    try:
        return str(ts_str)[11:19]
    except Exception:
        return str(ts_str)[:19]


def format_location(row: pd.Series) -> str:
    """Build location string from merchant_country and channel."""
    country = str(row.get("merchant_country", "??")).upper()
    channel = str(row.get("channel", "")).replace("_", " ").title()
    return f"{country} · {channel}"


def build_tx_record(row: pd.Series, min_score: float, max_score: float) -> dict:
    """
    Convert one gold-layer row into the exact txData object shape
    that the existing frontend JavaScript expects.

    Shape (must match 67.html exactly):
    {
      id, card, amount, amountNum, pattern, badgeClass,
      ts, score, merchant, loc, txid, ip, reason
    }
    """
    score      = float(row.get("fraud_score", 0))
    amount     = float(row.get("amount", 0))
    ip         = row.get("ip_address", "N/A")
    ip         = "N/A" if pd.isna(ip) else str(ip)
    is_flagged = bool(row.get("is_flagged", False))
    status     = str(row.get("review_status", "pending" if is_flagged else "clear"))

    return {
        "id":         f"#{row['transaction_id']}",
        "card":       str(row["card_id"]),
        "amount":     f"${amount:,.2f}",
        "amountNum":  round(amount, 2),
        "pattern":    pattern_label(row) if is_flagged else "",
        "badgeClass": badge_class(score) if is_flagged else "badge-accent",
        "ts":         format_timestamp(str(row.get("timestamp", ""))),
        "score":      score_to_100(score, min_score, max_score) if is_flagged else 0,
        "merchant":   str(row.get("merchant_name", "")),
        "loc":        format_location(row),
        "txid":       str(row["transaction_id"]),
        "ip":         ip,
        "reason":     str(row.get("flag_reasons", "Anomaly score threshold exceeded")) if is_flagged else "Normal transaction. No fraud pattern detected.",
        "isFlagged":  is_flagged,
        "reviewStatus": status,
    }


# ── Dashboard summary stats ───────────────────────────────────────────────────
# The frontend has two stat elements: stat-total-frauds and stat-money-loss.
# Pattern cards are rendered dynamically in 67.html from the injected txData.

def build_stats(flagged: pd.DataFrame, all_df: pd.DataFrame) -> dict:
    total_flagged  = len(flagged)
    total_money    = flagged["amount"].sum()
    velocity_count = flagged["flag_reasons"].str.contains("Velocity burst", na=False).sum()
    amount_count   = flagged["flag_reasons"].str.contains("Amount", na=False).sum()
    gift_count     = flagged["flag_reasons"].str.contains("Gift card", na=False).sum()
    burst_count    = flagged["flag_reasons"].str.contains("Merchant burst", na=False).sum()
    # Gauge: ratio of flagged to total, scaled to 0-100
    gauge_score    = min(100, round((total_flagged / len(all_df)) * 100 * 10))

    return {
        "total_flagged":  total_flagged,
        "total_money":    f"${total_money:,.2f}",
        "velocity_count": int(velocity_count),
        "amount_count":   int(amount_count),
        "gift_count":     int(gift_count),
        "burst_count":    int(burst_count),
        "gauge_score":    gauge_score,
    }


# ── Surgical HTML injection ───────────────────────────────────────────────────

def inject_into_html(records: list[dict], stats: dict, html_path: Path) -> None:
    html = html_path.read_text(encoding="utf-8")

    # 1. Replace txData array
    new_tx_js = "const txData = " + json.dumps(records, indent=2, ensure_ascii=False) + ";"
    html = re.sub(
        r"const txData\s*=\s*\[.*?\];",
        new_tx_js,
        html,
        flags=re.DOTALL
    )

    # 2. Replace stat-total-frauds static text
    # The frontend sets this via calcSummaryStats() — we update that function's output
    # by replacing the hardcoded fallback values it starts with in the SVG/HTML.
    # stat-total-frauds and stat-money-loss are set by calcSummaryStats() which reads
    # txData — so once txData is real, calcSummaryStats() will compute correctly.
    # No further changes needed for those two stats.

    # 3. Update gauge SVG score number (the "64" hardcoded in the SVG text element)
    html = re.sub(
        r'(<text[^>]*text-anchor="middle"[^>]*font-size="24"[^>]*>)\d+(<\/text>)',
        rf'\g<1>{stats["gauge_score"]}\2',
        html
    )

    # 4. Pattern names, counts, and bars are now computed in 67.html from txData.
    # Do not overwrite them here; this keeps the dashboard aligned with any
    # backend-detected pattern labels.

    html_path.write_text(html, encoding="utf-8")


# ── Runner ────────────────────────────────────────────────────────────────────

def run():
    print("inject_data.py — connecting pipeline to frontend")
    print(f"  Reading: {CSV_PATH}")

    if not CSV_PATH.exists():
        print(f"  ERROR: {CSV_PATH} not found. Run run_pipeline.py first.")
        return

    if not HTML_PATH.exists():
        print(f"  ERROR: {HTML_PATH} not found. Check the path.")
        return

    df = pd.read_csv(CSV_PATH)
    print(f"  Loaded: {len(df):,} total transactions")

    flagged = df[df["is_flagged"] == True].copy()
    print(f"  Flagged: {len(flagged)} transactions")

    if flagged.empty:
        min_score = 0
        max_score = 1
    else:
        min_score = flagged["fraud_score"].min()
        max_score = flagged["fraud_score"].max()

    records = [build_tx_record(row, min_score, max_score) for _, row in df.iterrows()]
    stats   = build_stats(flagged, df)

    inject_into_html(records, stats, HTML_PATH)

    print(f"  Injected {len(records)} records into {HTML_PATH.name}")
    print(f"  Stats: {stats['total_flagged']} flagged · {stats['total_money']} at risk")
    print("  Done — open 67.html in your browser.")


if __name__ == "__main__":
    run()
