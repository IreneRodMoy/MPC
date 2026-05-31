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
    # (You might need to slightly tweak run_pipeline.py so it reads the new raw_upload.csv)
    run_pipeline.run()

    # 3. Read the resulting Gold data
    gold_csv = Path("data/gold/transactions_flagged.csv")
    df = pd.read_csv(gold_csv)
    flagged = df[df["is_flagged"] == True].copy().sort_values("fraud_score", ascending=False)

    # 4. Format it using the exact same logic you already wrote in inject_data.py!
    min_score = flagged["fraud_score"].min()
    max_score = flagged["fraud_score"].max()
    
    records = [build_tx_record(row, min_score, max_score) for _, row in flagged.iterrows()]

    # 5. Return the JSON payload to the frontend
    return JSONResponse(content=records)

# Run this server in your terminal using:
# uvicorn server:app --reload