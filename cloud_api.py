"""ViSPAC Cloud Layer API.

This module implements the cloud tier of the ViSPAC (Vital Signs Prioritized 
Adaptive Compression) edge-fog-cloud architecture for healthcare IoT systems.

The cloud layer is responsible for:
    - Receiving processed vital signs data from the fog layer
    - Persisting events to a database (PostgreSQL or SQLite)
    - Storing data organized by patient risk level (HIGH, MODERATE, LOW, MINIMAL)

Architecture:
    Edge (sensors) -> Fog (NEWS2 scoring) -> Cloud (this module, storage)

Database Schema:
    The 'events' table stores patient vital signs with their calculated
    NEWS2 scores and risk classifications. Schema is auto-created at startup.

Environment Variables:
    CLOUD_DB_URL: PostgreSQL connection string (e.g., postgresql://user:pass@host:5432/db)
    CLOUD_DB_PATH: SQLite file path (default: cloud_data.sqlite3), used if CLOUD_DB_URL not set

Author: Mateus Roveda
Master's Dissertation - ViSPAC Project
"""

from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Body, Path
from contextlib import asynccontextmanager
from pydantic import BaseModel
import sqlite3, json, os, logging
import uvicorn

# Ensure logs directory exists
LOG_PATH = "logs/cloud_log.txt"
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

# Configure logging with unbuffered file handler
file_handler = logging.FileHandler(LOG_PATH, 'a')
file_handler.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[file_handler, stream_handler],
    force=True)
log = logging.getLogger("vispac-cloud")
log.setLevel(logging.INFO)
log.addHandler(file_handler)
log.addHandler(stream_handler)

DB_PATH = os.environ.get("CLOUD_DB_PATH", "cloud_data.sqlite3")
DB_URL = os.environ.get("CLOUD_DB_URL", "")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context to initialize DB schema without using on_event (deprecated)."""
    try:
        conn = get_conn()
        conn.close()
        log.info("Database schema ensured at startup.")
    except Exception as e:
        # Defensive logging; compose should wait for DB health, but just in case
        log.error(f"Database init on startup failed: {e}")
    yield
    # No shutdown actions required

app = FastAPI(title="ViSPAC Cloud Ingest API", lifespan=lifespan)

RISK_VALUES = {"high", "moderate", "low", "minimal"}

class IngestItem(BaseModel):
    """Pydantic model for incoming patient data from the fog layer.
    
    This model represents a single patient event with vital signs snapshot
    and their calculated NEWS2 score.
    
    Attributes:
        patient_id: Unique identifier for the patient.
        signal: Type of signal being recorded ('hr' for heart rate, 'spo2' for oxygen saturation).
        data: Compressed time-series data from the edge layer (SDT-compressed points).
        forced_vitals: Additional vital signs that were forced/simulated for testing.
        score: Calculated NEWS2 score (0-20 range, where higher = more critical).
        risk: Risk classification based on NEWS2 (HIGH/MODERATE/LOW/MINIMAL).
        hr: Heart rate in beats per minute (bpm).
        spo2: Oxygen saturation percentage (SpO2).
        rr: Respiratory rate in breaths per minute.
        temp: Body temperature in Celsius.
        sys_bp: Systolic blood pressure in mmHg.
        on_o2: Whether the patient is receiving supplemental oxygen.
        spo2_scale: SpO2 scoring scale (1 for standard, 2 for COPD patients).
        consciousness: AVPU consciousness level (A=Alert, V=Voice, P=Pain, U=Unresponsive).
    """
    
    patient_id: str
    signal: str | None = None
    data: Any | None = None
    forced_vitals: Dict[str, Any] | None = None
    score: int
    risk: str
    # vitals snapshot
    hr: float | None = None
    spo2: float | None = None
    rr: float | None = None
    temp: float | None = None
    sys_bp: float | None = None
    on_o2: bool | None = None
    spo2_scale: int | None = None
    consciousness: str | None = None


def get_conn():
    """Create and return a database connection.
    
    Automatically detects whether to use PostgreSQL or SQLite based on
    the CLOUD_DB_URL environment variable. Creates the 'events' table
    if it doesn't exist.
    
    Returns:
        Connection object (psycopg2 connection for PostgreSQL, 
        sqlite3 connection for SQLite).
    
    Raises:
        Exception: If database connection fails.
    
    Note:
        The returned connection should be closed by the caller after use.
    """
    if DB_URL.startswith("postgres"):
        import psycopg2
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id SERIAL PRIMARY KEY,
                patient_id TEXT,
                risk TEXT,
                score INTEGER,
                signal TEXT,
                data_json TEXT,
                hr REAL,
                spo2 REAL,
                rr REAL,
                temp REAL,
                sys_bp REAL,
                on_o2 BOOLEAN,
                spo2_scale INTEGER,
                consciousness TEXT,
                received_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
        return conn
    else:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT,
                risk TEXT,
                score INTEGER,
                signal TEXT,
                data_json TEXT,
                hr REAL,
                spo2 REAL,
                rr REAL,
                temp REAL,
                sys_bp REAL,
                on_o2 INTEGER,
                spo2_scale INTEGER,
                consciousness TEXT,
                received_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        return conn

@app.post("/cloud/ingest/{risk}", summary="Receive items by risk stream and store to DB")
async def ingest_by_risk(
    risk: str = Path(..., description="Risk stream: high|moderate|low|minimal"),
    items: List[IngestItem] = Body(...),
):
    r = risk.lower()
    if r not in RISK_VALUES:
        raise HTTPException(400, f"Invalid risk stream: {risk}")

    log.info(f"Received {len(items)} items for {risk.upper()} risk stream")
    # Optional validation: ensure each item has risk compatible with path
    for it in items:
    # Allow for case differences; normalize both
        it_risk_norm = it.risk.lower()
        if it_risk_norm != r:
            raise HTTPException(400, f"Item risk {it.risk} != path risk {risk}")

    try:
        conn = get_conn()
        if DB_URL.startswith("postgres"):
            cur = conn.cursor()
            cur.executemany(
                """
                INSERT INTO events(patient_id, risk, score, signal, data_json,
                                   hr, spo2, rr, temp, sys_bp, on_o2, spo2_scale, consciousness)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                [
                    (
                        it.patient_id,
                        it.risk,
                        it.score,
                        it.signal,
                        json.dumps(it.data) if it.data is not None else None,
                        it.hr,
                        it.spo2,
                        it.rr,
                        it.temp,
                        it.sys_bp,
                        it.on_o2,
                        it.spo2_scale,
                        it.consciousness,
                    )
                    for it in items
                ],
            )
            conn.commit(); conn.close()
        else:
            cur = conn.cursor()
            cur.executemany(
                """
                INSERT INTO events(patient_id, risk, score, signal, data_json,
                                   hr, spo2, rr, temp, sys_bp, on_o2, spo2_scale, consciousness)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                [
                    (
                        it.patient_id,
                        it.risk,
                        it.score,
                        it.signal,
                        json.dumps(it.data) if it.data is not None else None,
                        it.hr,
                        it.spo2,
                        it.rr,
                        it.temp,
                        it.sys_bp,
                        1 if it.on_o2 else 0 if it.on_o2 is not None else None,
                        it.spo2_scale,
                        it.consciousness,
                    )
                    for it in items
                ],
            )
            conn.commit(); conn.close()
    except Exception as e:
        log.error(f"DB error storing {len(items)} items: {e}")
        raise HTTPException(500, f"DB error: {e}")

    log.info(f"Successfully stored {len(items)} items to DB ({risk.upper()})")
    return {"accepted": len(items), "risk": r}


if __name__ == "__main__":
    # CLOUD layer API
    # Run the app object directly to ensure lifespan runs reliably in containers
    log.info("Starting CLOUD API on port 9000")
    uvicorn.run(app, host="0.0.0.0", port=9000, log_config=None)
