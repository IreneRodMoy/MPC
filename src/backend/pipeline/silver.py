"""
SILVER LAYER — silver.py
=========================
Responsibility: feature engineering only. No scores, no fraud labels.
Reads bronze parquet, adds 7 new signal columns, saves silver parquet.
 
New columns added:
  - hour_of_day              (int)
  - is_odd_hours             (bool)
  - amount_vs_card_median    (float)
  - tx_count_last_30min      (int)
  - is_new_device_for_card   (bool)
  - merchant_card_burst_24h  (int)
  - is_cross_border          (bool)
"""
 
import pandas as pd
from pathlib import Path
import logging
 
logging.basicConfig(level=logging.INFO, format="%(asctime)s [SILVER] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
 
ROOT       = Path(__file__).resolve().parent.parent.parent.parent
BRONZE_IN  = ROOT / "data" / "bronze" / "transactions_bronze.parquet"
SILVER_OUT = ROOT / "data" / "silver" / "transactions_silver.parquet"
 
 
# ── Group 1: Time features ─────────────────────────────────────────────────────
 
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract hour_of_day and is_odd_hours from timestamp.
 
    is_odd_hours = True when the transaction happens between 2am and 5am.
    On its own this isn't fraud — but combined with other signals it matters,
    so we compute it here and let Gold decide how much weight to give it.
    """
    log.info("Adding time features...")
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["is_odd_hours"] = df["hour_of_day"].between(2, 5)
    log.info(f"  odd-hours transactions: {df['is_odd_hours'].sum()}")
    return df
 
 
# ── Group 2: Per-card features ─────────────────────────────────────────────────
 
def add_amount_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each transaction, compute how large the amount is relative to
    that card's median across the whole month.
 
    Example: card_018 median = $31.79. A $1,753 Gift Card Mall charge
    gives a ratio of 55× — extremely anomalous for that card's profile.
 
    We use median (not mean) because it's robust to the outliers we're
    trying to detect — mean would be inflated by the fraud itself.
    """
    log.info("Adding amount vs card median ratio...")
    card_median = df.groupby("card_id")["amount"].median().rename("card_median")
    df = df.join(card_median, on="card_id")
    df["amount_vs_card_median"] = (df["amount"] / df["card_median"]).round(2)
    log.info(f"  max ratio in dataset: {df['amount_vs_card_median'].max():.1f}×")
    df = df.drop(columns=["card_median"])
    return df
 
 
def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each transaction, count how many OTHER transactions the same card
    made in the 30 minutes before it.
 
    This catches the card-testing pattern: card_023 fires 10 transactions
    in 28 minutes — each of those rows gets a high tx_count_last_30min.
 
    We sort by card + timestamp, then use a rolling window per card.
    The window is time-based (30 min), closed on the left so we don't
    count the transaction itself.
    """
    log.info("Adding velocity features (tx count in last 30 min per card)...")
    df = df.sort_values(["card_id", "timestamp"]).reset_index(drop=True)
 
    counts = pd.Series(index=df.index, dtype=float)
    for card_id, group in df.groupby("card_id"):
        ts_indexed = group.set_index("timestamp").sort_index()
        rolling = ts_indexed["amount"].rolling("30min", closed="both").count().astype(int) - 1
        counts.loc[group.index] = rolling.values  # map back via original integer index
 
    df["tx_count_last_30min"] = counts.fillna(0).astype(int)
    log.info(f"  max tx_count_last_30min: {df['tx_count_last_30min'].max()}")
    return df
 
 
def add_new_device_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each online transaction, was this device_id seen before on this card?
 
    We only look at prior transactions (no lookahead) — for each row we check
    whether this device appeared in any earlier row for the same card.
    Non-online rows get False (device/IP not applicable).
 
    A new device mid-month on a card that always uses the same device
    is a meaningful signal, especially when combined with a large amount.
    """
    log.info("Adding new device flag...")
    df = df.sort_values(["card_id", "timestamp"]).reset_index(drop=True)
    df["is_new_device_for_card"] = False
 
    for card_id, group in df.groupby("card_id"):
        online_rows = group[group["channel"] == "online"].index
        seen_devices = set()
        for idx in online_rows:
            device = df.loc[idx, "device_id"]
            if pd.isna(device):
                continue
            if device not in seen_devices:
                # first time we see this device on this card
                df.loc[idx, "is_new_device_for_card"] = True
                seen_devices.add(device)
 
    log.info(f"  new device transactions: {df['is_new_device_for_card'].sum()}")
    return df
 
 
# ── Group 3: Cross-card features ───────────────────────────────────────────────
def add_merchant_burst_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each transaction, count how many DIFFERENT cards hit this same
    merchant in the 24 hours before it.

    O(n log n) approach:
      - Sort once by timestamp
      - For each merchant group, use searchsorted to find the 24h window start
      - Count distinct cards in that slice with a sliding pointer
    """
    log.info("Adding merchant card burst (last 24h, cross-card)...")
    df = df.sort_values("timestamp").reset_index(drop=True)

    burst_counts = pd.Series(0, index=df.index, dtype=int)
    timestamps_ns = df["timestamp"].astype("int64").values  # nanoseconds

    window_ns = pd.Timedelta(hours=24).value  # nanoseconds

    for merchant, group in df.groupby("merchant_name"):
        idx = group.index.values          # positions in df
        ts  = timestamps_ns[idx]          # timestamps for this merchant
        cards = df.loc[idx, "card_id"].values

        for i in range(len(idx)):
            # binary search: first position where ts >= ts[i] - 24h
            lo = ts.searchsorted(ts[i] - window_ns, side="left")
            # window is [lo, i) — exclude current row
            burst_counts.iloc[idx[i]] = len(set(cards[lo:i]))

    df["merchant_card_burst_24h"] = burst_counts
    log.info(f"  max merchant burst in 24h: {df['merchant_card_burst_24h'].max()} unique cards")
    return df
 
 
def add_cross_border_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    True when the cardholder's home country differs from the merchant's country.
    A CA card buying from a CN merchant (AliExpress) is cross-border.
 
    Standalone this is weak — lots of legit cross-border purchases exist.
    Gold will only count it when combined with other signals.
    """
    log.info("Adding cross-border flag...")
    df["is_cross_border"] = df["cardholder_country"] != df["merchant_country"]
    log.info(f"  cross-border transactions: {df['is_cross_border'].sum()}")
    return df
 
 
# ── Runner ─────────────────────────────────────────────────────────────────────
 
def run() -> pd.DataFrame:
    log.info("Starting Silver layer")
    df = pd.read_parquet(BRONZE_IN)
    log.info(f"Loaded bronze: {len(df):,} rows, {len(df.columns)} columns")
 
    df = add_time_features(df)
    df = add_amount_ratio(df)
    df = add_velocity_features(df)
    df = add_new_device_flag(df)
    df = add_merchant_burst_flag(df)
    df = add_cross_border_flag(df)
 
    SILVER_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SILVER_OUT, index=False)
    log.info(f"Silver saved → {SILVER_OUT}  ({len(df):,} rows, {len(df.columns)} columns)")
    return df
 
 
if __name__ == "__main__":
    run()