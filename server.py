from collections import Counter, defaultdict
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import shutil
from pathlib import Path

# Import your existing pipeline and injection logic
import run_pipeline 
from inject_data import build_tx_record 

app = FastAPI()

# Allow your frontend to talk to this API (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/analyze")
async def analyze_document(file: UploadFile = File(...)):
    # 1. Save the uploaded file to your Bronze ingest folder
    ingest_path = Path("data/bronze/raw_upload.csv")
    with ingest_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2. Run your existing pipeline!
    run_pipeline.run()

    # 3. Read the resulting Gold data for the frontend
    gold_csv = Path("data/gold/transactions_flagged.csv")
    df = pd.read_csv(gold_csv)

    if df.empty:
        return JSONResponse(content=[])

    # 4. Convert all processed rows into the frontend txData shape
    min_score = df["fraud_score"].min()
    max_score = df["fraud_score"].max()
    records = [build_tx_record(row, min_score, max_score) for _, row in df.iterrows()]

    # 5. Return the JSON payload to the frontend
    return JSONResponse(content=records)


def _safe_string(value):
    if pd.isna(value):
        return ''
    return str(value).strip()


def _format_node_id(prefix, raw_value):
    if not raw_value:
        return None
    clean = str(raw_value).strip().replace(' ', '_').replace('/', '-').replace(':', '_')
    return f"{prefix}:{clean}"


def _build_constellation_payload(csv_path: Path):
    if not csv_path.exists():
        return {"nodes": [], "edges": []}

    df = pd.read_csv(csv_path)
    if df.empty:
        return {"nodes": [], "edges": []}

    card_nodes = {}
    merchant_nodes = {}
    device_nodes = {}
    ip_nodes = {}
    tx_nodes = {}
    edges = []
    device_card = defaultdict(set)
    ip_card = defaultdict(set)
    device_scores = defaultdict(float)
    ip_scores = defaultdict(float)

    for _, row in df.iterrows():
        txid = _safe_string(row.get("transaction_id") or row.get("txid"))
        if not txid:
            continue

        card = _safe_string(row.get("card_id") or row.get("card"))
        merchant = _safe_string(row.get("merchant_name") or row.get("merchant")) or "Unknown Merchant"
        ip = _safe_string(row.get("ip_address") or row.get("ip"))
        device = _safe_string(row.get("device_id") or row.get("device"))
        score = float(row.get("fraud_score", 0) or 0)
        is_flagged = bool(row.get("is_flagged") in [True, "True", "true", 1, "1"])
        reason = _safe_string(row.get("flag_reasons") or row.get("reason"))

        tx_node_id = _format_node_id("tx", txid)
        card_node_id = _format_node_id("card", card)
        merchant_node_id = _format_node_id("merch", merchant)
        ip_node_id = _format_node_id("ip", ip)
        device_node_id = _format_node_id("dev", device)

        tx_nodes[tx_node_id] = {
            "id": tx_node_id,
            "label": txid,
            "group": "transaction",
            "title": reason,
            "risk_level": "Critical" if is_flagged else "Low",
            "score": score
        }

        if card_node_id:
            existing = card_nodes.get(card_node_id)
            if not existing:
                card_nodes[card_node_id] = {
                    "id": card_node_id,
                    "label": card,
                    "group": "card",
                    "risk_level": "Low",
                    "score": score,
                    "transaction_count": 0
                }
                existing = card_nodes[card_node_id]
            existing["score"] = max(existing["score"], score)
            existing["transaction_count"] += 1
            if is_flagged and existing["score"] >= 0.35:
                existing["risk_level"] = "High"
            edges.append({"id": f"e-{card_node_id}-{tx_node_id}", "from": card_node_id, "to": tx_node_id})

        if merchant_node_id:
            existing = merchant_nodes.get(merchant_node_id)
            if not existing:
                merchant_nodes[merchant_node_id] = {
                    "id": merchant_node_id,
                    "label": merchant,
                    "group": "merchant",
                    "risk_level": "Low",
                    "score": score,
                    "transaction_count": 0
                }
                existing = merchant_nodes[merchant_node_id]
            existing["score"] = max(existing["score"], score)
            existing["transaction_count"] += 1
            if is_flagged and existing["score"] >= 0.35:
                existing["risk_level"] = "Medium"
            edges.append({"id": f"e-{tx_node_id}-{merchant_node_id}", "from": tx_node_id, "to": merchant_node_id})

        if ip_node_id:
            existing = ip_nodes.get(ip_node_id)
            if not existing:
                ip_nodes[ip_node_id] = {
                    "id": ip_node_id,
                    "label": ip,
                    "group": "ip",
                    "risk_level": "Low",
                    "score": score,
                    "transaction_count": 0
                }
                existing = ip_nodes[ip_node_id]
            existing["score"] = max(existing["score"], score)
            existing["transaction_count"] += 1
            if is_flagged and existing["score"] >= 0.35:
                existing["risk_level"] = "Medium"
            ip_card[ip_node_id].add(card_node_id)
            device_scores[ip_node_id] = max(device_scores[ip_node_id], score)
            edges.append({"id": f"e-{tx_node_id}-{ip_node_id}", "from": tx_node_id, "to": ip_node_id})

        if device_node_id:
            existing = device_nodes.get(device_node_id)
            if not existing:
                device_nodes[device_node_id] = {
                    "id": device_node_id,
                    "label": device,
                    "group": "device",
                    "risk_level": "Low",
                    "score": score,
                    "transaction_count": 0
                }
                existing = device_nodes[device_node_id]
            existing["score"] = max(existing["score"], score)
            existing["transaction_count"] += 1
            if is_flagged and existing["score"] >= 0.35:
                existing["risk_level"] = "Medium"
            device_card[device_node_id].add(card_node_id)
            device_scores[device_node_id] = max(device_scores[device_node_id], score)
            edges.append({"id": f"e-{tx_node_id}-{device_node_id}", "from": tx_node_id, "to": device_node_id})

    nodes = [*card_nodes.values(), *merchant_nodes.values(), *ip_nodes.values(), *device_nodes.values(), *tx_nodes.values()]

    cluster_count = 0
    for device_id, cards in device_card.items():
        if len([c for c in cards if c]) >= 3:
            cluster_count += 1
            cluster_id = f"cluster:device:{cluster_count}"
            nodes.append({"id": cluster_id, "label": "Device reuse cluster", "group": "cluster", "risk_level": "Critical", "score": max(device_scores[device_id], 0.8)})
            edges.append({"id": f"e-{cluster_id}-{device_id}", "from": cluster_id, "to": device_id})
            for card_id in cards:
                if card_id:
                    edges.append({"id": f"e-{cluster_id}-{card_id}", "from": cluster_id, "to": card_id})

    for ip_id, cards in ip_card.items():
        if len([c for c in cards if c]) >= 3:
            cluster_count += 1
            cluster_id = f"cluster:ip:{cluster_count}"
            nodes.append({"id": cluster_id, "label": "IP reuse cluster", "group": "cluster", "risk_level": "High", "score": max(ip_scores[ip_id], 0.7)})
            edges.append({"id": f"e-{cluster_id}-{ip_id}", "from": cluster_id, "to": ip_id})
            for card_id in cards:
                if card_id:
                    edges.append({"id": f"e-{cluster_id}-{card_id}", "from": cluster_id, "to": card_id})

    return {"nodes": nodes, "edges": edges}


@app.get("/api/constellation")
async def get_constellation():
    gold_csv = Path("data/gold/transactions_flagged.csv")
    payload = _build_constellation_payload(gold_csv)
    return JSONResponse(content=payload)

# Run this server in your terminal using:
# uvicorn server:app --reload