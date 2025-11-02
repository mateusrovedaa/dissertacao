from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Body, Path
from contextlib import asynccontextmanager
from pydantic import BaseModel
import sqlite3, json, os
import uvicorn

DB_PATH = os.environ.get("CLOUD_DB_PATH", "cloud_data.sqlite3")
DB_URL = os.environ.get("CLOUD_DB_URL", "")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Contexto de vida (lifespan) do FastAPI para inicializar o esquema do BD sem usar on_event (obsoleto)."""
    try:
        conn = get_conn()
        conn.close()
        print("[cloud_api] Database schema ensured at startup.")
    except Exception as e:
    # Log defensivo; o compose deve aguardar saúde do BD, mas por garantia
        print(f"[cloud_api] Database init on startup failed: {e}")
    yield
    # No shutdown actions required

app = FastAPI(title="ViSPAC Cloud Ingest API", lifespan=lifespan)

RISK_VALUES = {"alto", "moderado", "baixo", "minimo"}

class IngestItem(BaseModel):
    patient_id: str
    signal: str | None = None
    data: Any | None = None
    forced_vitals: Dict[str, Any] | None = None
    score: int
    risk: str
    # instantâneo de sinais vitais
    hr: float | None = None
    spo2: float | None = None
    rr: float | None = None
    temp: float | None = None
    sys_bp: float | None = None
    on_o2: bool | None = None
    spo2_scale: int | None = None
    consciousness: str | None = None


def get_conn():
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
    risk: str = Path(..., description="Risk stream: alto|moderado|baixo|minimo"),
    items: List[IngestItem] = Body(...),
):
    r = risk.lower()
    if r not in RISK_VALUES:
        raise HTTPException(400, f"Invalid risk stream: {risk}")

    # Verificação opcional: garantir que cada item tenha risco compatível com o path
    for it in items:
    # Permite diferença de acentuação; normaliza ambos
        it_risk_norm = it.risk.lower().replace("í", "i").replace("Í", "i")
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
        raise HTTPException(500, f"DB error: {e}")

    return {"accepted": len(items), "risk": r}


if __name__ == "__main__":
    # CLOUD layer API
    # Executa o objeto app diretamente para garantir que o lifespan rode de forma confiável em containers
    uvicorn.run(app, host="0.0.0.0", port=9000)
