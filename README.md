# ViSPAC edge/fog/cloud prototype

This repository prototypes an edge→fog→cloud flow for time-series vitals compression, scoring (NEWS2) and storage.

## Components

- edge: `vispac_edge_prototype.py` — simulator that collects samples, applies SDT and lossless, and sends batches to fog.
- fog: `news2_api.py` — FastAPI service (port 8000) that decompresses batches, computes NEWS2, returns scores to edge and forwards risk-based streams to cloud.
- cloud: `cloud_api.py` — FastAPI service (port 9000) that receives risk-specific streams and persists them to a DB (PostgreSQL via `CLOUD_DB_URL`, else SQLite via `CLOUD_DB_PATH`).
- mqtt: Eclipse Mosquitto broker used when running in MQTT mode.

## Datasets

This project supports two types of datasets for patient simulation:

- **Low Risk**: Healthy students data from Kaggle (~600 samples, 7 virtual patients)
- **High Risk**: ICU patients from PhysioNet BIDMC (~25,000 samples, 53 real patients)

**For detailed information about datasets, see [DATASETS.md](DATASETS.md)**

Quick start:
```bash
# Download high-risk dataset
python download_bidmc_data.py

# Run simulation with high-risk data
export DATASET_TYPE=high_risk
python vispac_edge_prototype.py

# Or use the interactive menu
python run_simulation.py
```

## Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run (HTTP mode)

```bash
# start cloud (port 9000)
python cloud_api.py &
# start fog (port 8000)
python news2_api.py &
# run edge simulator
python vispac_edge_prototype.py
```

## Run (MQTT mode)

```bash
# install a local MQTT broker (example: mosquitto)
# sudo apt install mosquitto
# mosquitto -v

# enable MQTT in the edge simulator
export EDGE_USE_MQTT=1
# optional: set broker and port
export MQTT_BROKER=127.0.0.1
export MQTT_PORT=1883

# start cloud and fog
python cloud_api.py &
python news2_api.py &

# run edge simulator (will publish batches to MQTT topic 'vispac/upload_batch')
python vispac_edge_prototype.py
```

## Configuration

- `CLOUD_BASE_URL` (env) — where fog forwards risk streams (default `http://127.0.0.1:9000`).
- `CLOUD_DB_URL` (env) — PostgreSQL connection string for cloud (e.g., `postgresql://user:pass@host:5432/dbname`).
- `CLOUD_DB_PATH` (env) — SQLite file used by cloud if `CLOUD_DB_URL` is not set (default `cloud_data.sqlite3`).
- `EDGE_USE_MQTT`, `MQTT_BROKER`, `MQTT_PORT` — configure MQTT behaviour.
- `DATASET_TYPE` (env) — choose dataset: `low_risk` or `high_risk` (see [DATASETS.md](DATASETS.md)).

## Run with Docker Compose (PostgreSQL + MQTT)

```bash
# build images and start services (db, cloud, fog, edge)
docker compose up --build

# logs
docker compose logs -f

# stop
docker compose down -v
```

The Compose stack runs PostgreSQL and Mosquitto and wires services:

- db: Postgres 17 (port 5432)
- mqtt: Mosquitto broker (port 1883)
- cloud: FastAPI (port 9000), uses `CLOUD_DB_URL=postgresql://vispac:vispac@db:5432/vispac` and ensures the schema at startup
- fog: FastAPI (port 8000), forwards to `http://cloud:9000` and listens to MQTT
  (forwards to cloud in risk priority order: HIGH → MODERATE → LOW → MINIMAL)
- edge: simulator, posts to `http://fog:8000/vispac/upload_batch` and uses MQTT when `EDGE_USE_MQTT=1`

Verify the database schema in Compose:

```bash
# look for startup message
docker compose logs -f cloud | grep -i schema

# describe the table in Postgres
docker compose exec db psql -U vispac -d vispac -c "\\d+ events"
```

Inspect stored events:

```bash
docker compose exec db psql -U vispac -d vispac -c "SELECT id, patient_id, risk, score, signal, received_at FROM events ORDER BY id DESC LIMIT 10;"
```

If you use SQLite locally (no `CLOUD_DB_URL`):

```bash
sqlite3 cloud_data.sqlite3 ".schema events"
sqlite3 cloud_data.sqlite3 "SELECT id, patient_id, risk, score, signal, received_at FROM events ORDER BY id DESC LIMIT 10;"
```

## Database model (events)

Columns persisted by the cloud layer:

- id: integer (SERIAL/AUTOINCREMENT)
- patient_id: text
- risk: text (normalized stream key: high|moderate|low|minimal)
- score: integer (NEWS2)
- signal: text (source signal name when relevant)
- data_json: text (raw JSON payload when present)
- hr: real
- spo2: real
- rr: real
- temp: real
- sys_bp: real
- on_o2: boolean (Postgres) / integer 0/1 (SQLite)
- spo2_scale: integer
- consciousness: text
- received_at: timestamp default now

### DDL (PostgreSQL)

```sql
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
);
```

### DDL (SQLite)

```sql
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
);
```

## Notes

- The edge supports HTTP POST and optional MQTT (enable via `EDGE_USE_MQTT=1`).
- Fog accepts both the legacy `hushman` header and `huffman` as an alias.
- Cloud ensures the `events` schema at startup. When `CLOUD_DB_URL` is set, Postgres is used; otherwise it falls back to SQLite.
- Each patient in the simulation uses specific data from the chosen dataset (see [DATASETS.md](DATASETS.md)).

## Additional Documentation

- **[DATASETS.md](DATASETS.md)** - Complete guide on datasets (low/high risk), download instructions, and usage
- **[DATASETS_GUIDE.md](DATASETS_GUIDE.md)** - Detailed technical documentation with statistics and examples
