"""ViSPAC Fog Layer API - NEWS2 Scoring and Data Processing.

This module implements the fog tier of the ViSPAC (Vital Signs Prioritized
Adaptive Compression) edge-fog-cloud architecture for healthcare IoT systems.

The fog layer is responsible for:
    - Receiving compressed batches from edge devices
    - Decompressing data (Huffman, LZW, or none)
    - Calculating NEWS2 (National Early Warning Score 2) for each patient
    - Forwarding data to the cloud layer organized by risk priority
    - Providing real-time feedback to edge devices for adaptive sampling

NEWS2 Algorithm:
    The National Early Warning Score 2 (NEWS2) is a standardized tool developed
    by the Royal College of Physicians (UK) for detecting patient deterioration.
    It assigns scores to six physiological parameters:
        - Respiratory rate (0-3 points)
        - Oxygen saturation (0-3 points, with special scale for COPD)
        - Supplemental oxygen (0-2 points)
        - Temperature (0-3 points)
        - Systolic blood pressure (0-3 points)
        - Heart rate (0-3 points)
        - Level of consciousness (0-3 points)
    
    Total score ranges from 0 to 20. Risk classification:
        - 0: MINIMAL risk
        - 1-4: LOW risk
        - 5-6: MODERATE risk (or any single parameter = 3)
        - 7+: HIGH risk (urgent clinical response required)

Author: Mateus Roveda
Master's Dissertation - ViSPAC Project
"""

from typing import Dict, Any, List, DefaultDict
from collections import defaultdict
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, Body, BackgroundTasks
from pydantic import BaseModel
import json, uvicorn, logging, time
import os, requests, threading, queue, itertools
try:
    import paho.mqtt.client as mqtt
except Exception:
    mqtt = None
from compressors import LZW, Huffman

# Ensure logs directory exists
LOG_PATH = "logs/fog_log.txt"
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
log = logging.getLogger("vispac-fog")
log.setLevel(logging.INFO)

# MQTT client storage
mqtt_client = None

# Async cloud forwarding (enabled via ASYNC_FORWARD=true for ViSPAC scenario)
ASYNC_FORWARD = os.environ.get("ASYNC_FORWARD", "false").lower() == "true"
RISK_PRIORITY = {"HIGH": 0, "MODERATE": 1, "LOW": 2, "MINIMAL": 3}
_cloud_queue = queue.PriorityQueue()
_queue_counter = itertools.count()


def _cloud_forward_worker():
    """Background worker that drains the priority queue and forwards to cloud.
    Items with higher clinical risk (lower priority number) are sent first."""
    base = os.environ.get("CLOUD_BASE_URL", "http://127.0.0.1:9000")
    while True:
        priority, _seq, risk, items = _cloud_queue.get()
        if items is None:  # shutdown sentinel
            break
        path = risk.lower()
        url = f"{base}/cloud/ingest/{path}"
        log.info(f"[QUEUE] Forwarding {len(items)} items to cloud ({risk})")
        try:
            requests.post(url, json=items, timeout=10)
            log.info(f"[QUEUE] Successfully forwarded {len(items)} items ({risk})")
        except Exception as e:
            log.error(f"[QUEUE] Failed to forward to cloud {url}: {e}")
        _cloud_queue.task_done()


def _enqueue_for_cloud(grouped: DefaultDict[str, List[Dict[str, Any]]]):
    """Enqueue items into the priority queue, ordered by clinical risk."""
    for risk, items in grouped.items():
        if items:
            _cloud_queue.put((RISK_PRIORITY.get(risk, 3), next(_queue_counter), risk, items))

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context for MQTT listener initialization"""
    global mqtt_client, _cloud_worker
    _cloud_worker = None
    # Startup: Initialize cloud forwarding worker if async mode
    if ASYNC_FORWARD:
        _cloud_worker = threading.Thread(target=_cloud_forward_worker, daemon=True)
        _cloud_worker.start()
        log.info("[QUEUE] Cloud priority queue worker started (async mode)")
    # Initialize MQTT listener
    if mqtt is not None:
        broker = os.environ.get('MQTT_BROKER','127.0.0.1')
        port = int(os.environ.get('MQTT_PORT','1883'))
        mqtt_client = mqtt.Client()
        
        # Configure TLS if CA certificate is provided
        ca_cert = os.environ.get('MQTT_CA_CERT')
        if ca_cert and os.path.exists(ca_cert):
            import ssl
            mqtt_client.tls_set(ca_certs=ca_cert, tls_version=ssl.PROTOCOL_TLS)
            mqtt_client.tls_insecure_set(True)  # Skip hostname check (dynamic IP)
            log.info(f"MQTT TLS enabled with CA: {ca_cert}")
        
        def on_message(client, userdata, message):
            try:
                data = json.loads(message.payload.decode())
                reply_topic = data.get('reply_topic')
                ctype = data.get('X-Compression-Type','none').lower()
                raw_payload = data.get('payload','').encode()
                log.info(f"[MQTT] Received message: {len(raw_payload)} bytes, compression={ctype}")
                start_time = time.time()
                for handler in log.handlers:
                    handler.flush()
                try:
                    process_start = time.time()
                    scores, to_cloud = _process_batch_payload(ctype, raw_payload)
                    process_duration = (time.time() - process_start) * 1000
                    
                    log.info(f"[MQTT] Processed batch: {len(scores)} patients")
                    for handler in log.handlers:
                        handler.flush()
                except HTTPException as e:
                    log.error(f"[MQTT] Processing error: {e}")
                    for handler in log.handlers:
                        handler.flush()
                    if reply_topic:
                        client.publish(reply_topic, json.dumps({'error': str(e)}))
                    return
                
                if ASYNC_FORWARD:
                    # ViSPAC: respond to edge immediately, then enqueue for cloud
                    if reply_topic:
                        client.publish(reply_topic, json.dumps({'scores': scores}))
                        log.info(f"[MQTT] Sent response to {reply_topic}: {len(scores)} scores")
                    _enqueue_for_cloud(to_cloud)
                    total_duration = (time.time() - start_time) * 1000
                    log.info(f"[FOG_METRICS] type=mqtt batch_size={len(scores)} process_ms={process_duration:.2f} forward_ms=0.00 total_ms={total_duration:.2f}")
                else:
                    # Baseline: forward to cloud synchronously, then respond
                    fwd_start = time.time()
                    _forward_to_cloud(to_cloud)
                    fwd_duration = (time.time() - fwd_start) * 1000
                    total_duration = (time.time() - start_time) * 1000
                    log.info(f"[FOG_METRICS] type=mqtt batch_size={len(scores)} process_ms={process_duration:.2f} forward_ms={fwd_duration:.2f} total_ms={total_duration:.2f}")
                    if reply_topic:
                        client.publish(reply_topic, json.dumps({'scores': scores}))
                        log.info(f"[MQTT] Sent response to {reply_topic}: {len(scores)} scores")
                for handler in log.handlers:
                    handler.flush()
            except Exception as e:
                log.error(f"MQTT handler error: {e}")
                for handler in log.handlers:
                    handler.flush()
        
        mqtt_client.on_message = on_message
        try:
            mqtt_client.connect(broker, port, 60)
            mqtt_client.subscribe('vispac/upload_batch')
            mqtt_client.loop_start()
            log.info(f"MQTT listener started on {broker}:{port} topic vispac/upload_batch")
        except Exception as e:
            log.error(f"MQTT connect failed: {e}")
    else:
        log.warning("paho-mqtt not installed; MQTT listener disabled.")
    
    yield
    
    # Shutdown: Cleanup MQTT client and cloud worker
    if mqtt_client:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
        log.info("MQTT listener stopped")
    if _cloud_worker and _cloud_worker.is_alive():
        _cloud_queue.put((99, 0, "", None))  # shutdown sentinel
        _cloud_worker.join(timeout=5)
        log.info("[QUEUE] Cloud priority queue worker stopped")

app = FastAPI(title="ViSPAC Scoring & Decompression API v2", lifespan=lifespan)

class Vitals(BaseModel):
    """Pydantic model representing a patient's vital signs.
    
    This model holds all physiological parameters required for NEWS2 calculation.
    Default values represent normal/healthy readings.
    
    Attributes:
        rr: Respiratory rate in breaths per minute. Normal: 12-20.
        spo2: Oxygen saturation percentage. Normal: 96-100%.
        spo2_scale: SpO2 scoring scale. 1=standard, 2=COPD patients (Scale 2).
        on_o2: Whether patient is receiving supplemental oxygen.
        temp: Body temperature in Celsius. Normal: 36.1-38.0°C.
        sys_bp: Systolic blood pressure in mmHg. Normal: 111-219.
        hr: Heart rate in beats per minute. Normal: 51-90.
        consciousness: AVPU scale level.
            A=Alert (normal), V=responds to Voice, P=responds to Pain, U=Unresponsive.
    """
    
    rr: float = 18
    spo2: float = 98
    spo2_scale: int = 1
    on_o2: bool = False
    temp: float = 36.5
    sys_bp: float = 120
    hr: float = 80
    consciousness: str = "A"

# --- NEWS2 Scoring Functions ------------------------------------------------
# Each function implements the scoring criteria from the Royal College of
# Physicians NEWS2 specification (2017). Scores range from 0 (normal) to 3 (critical).
# ------------------------------------------------------------------------------

def score_rr(rr: float) -> int:
    """Calculate NEWS2 score for respiratory rate.
    
    Scoring thresholds (breaths per minute):
        - ≤8:   Score 3 (Critical - bradypnea)
        - 9-11: Score 1 (Mild concern)
        - 12-20: Score 0 (Normal range)
        - 21-24: Score 2 (Moderate concern - tachypnea)
        - ≥25:  Score 3 (Critical - severe tachypnea)
    
    Args:
        rr: Respiratory rate in breaths per minute.
    
    Returns:
        int: NEWS2 score component (0-3).
    """
    return 3 if rr <= 8 else (1 if rr <= 11 else (0 if rr <= 20 else (2 if rr <= 24 else 3)))

def score_spo2(spo2: float, scale: int, on_o2: bool) -> int:
    """Calculate NEWS2 score for oxygen saturation (SpO2).
    
    NEWS2 provides two SpO2 scoring scales:
        - Scale 1: Standard scale for most patients
        - Scale 2: For patients with hypercapnic respiratory failure (e.g., COPD)
          who have target SpO2 of 88-92%
    
    Scale 1 Thresholds:
        - ≤91%: Score 3 (Critical hypoxemia)
        - 92-93%: Score 2
        - 94-95%: Score 1 (Mild hypoxemia)
        - ≥96%: Score 0 (Normal)
    
    Scale 2 Thresholds (for COPD patients):
        - ≤83%: Score 3 (Severe hypoxemia)
        - 84-85%: Score 2
        - 86-87%: Score 1
        - 88-92%: Score 0 (Target range for COPD)
        - >92% on O2: Score 1-3 (indicates possible over-oxygenation)
        - >92% off O2: Score 0
    
    Args:
        spo2: Oxygen saturation percentage (0-100).
        scale: Scoring scale (1=standard, 2=COPD).
        on_o2: Whether patient is receiving supplemental oxygen.
    
    Returns:
        int: NEWS2 score component (0-3).
    """
    if scale == 2:
        if spo2 <= 83: return 3
        if spo2 <= 85: return 2
        if spo2 <= 87: return 1
        if spo2 <= 92: return 0
        # spo2 > 92 on scale 2
        if on_o2:
            if spo2 <= 94: return 1
            if spo2 <= 96: return 2
            return 3
        else:
            return 0  # SpO2 > 92 without O2 on scale 2 is normal
    else:
        if spo2 <= 91: return 3
        if spo2 <= 93: return 2
        if spo2 <= 95: return 1
        return 0

def score_o2(on_o2: bool) -> int:
    """Calculate NEWS2 score for supplemental oxygen use.
    
    Any patient receiving supplemental oxygen scores 2 points,
    reflecting the clinical concern of oxygen dependency.
    
    Args:
        on_o2: Whether patient is receiving supplemental oxygen.
    
    Returns:
        int: NEWS2 score component (0 or 2).
    """
    return 2 if on_o2 else 0

def score_temp(temp: float) -> int:
    """Calculate NEWS2 score for body temperature.
    
    Scoring thresholds (Celsius):
        - ≤35.0°C: Score 3 (Critical - hypothermia)
        - 35.1-36.0°C: Score 1 (Mild hypothermia)
        - 36.1-38.0°C: Score 0 (Normal range)
        - 38.1-39.0°C: Score 1 (Mild fever)
        - ≥39.1°C: Score 2 (High fever)
    
    Args:
        temp: Body temperature in degrees Celsius.
    
    Returns:
        int: NEWS2 score component (0-3).
    """
    return 3 if temp <= 35 else (1 if temp <= 36 else (0 if temp <= 38 else (1 if temp <= 39 else 2)))

def score_bp(sys_bp: float) -> int:
    """Calculate NEWS2 score for systolic blood pressure.
    
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                Scoring thresholds (mmHg):
        - ≤90: Score 3 (Critical - severe hypotension)
        - 91-100: Score 2 (Moderate hypotension)
        - 101-110: Score 1 (Mild hypotension)
        - 111-219: Score 0 (Normal range)
        - ≥220: Score 3 (Critical - severe hypertension)
    
    Args:
        sys_bp: Systolic blood pressure in mmHg.
    
    Returns:
        int: NEWS2 score component (0-3).
    """
    return 3 if sys_bp <= 90 else (2 if sys_bp <= 100 else (1 if sys_bp <= 110 else (0 if sys_bp <= 219 else 3)))

def score_hr(hr: float) -> int:
    """Calculate NEWS2 score for heart rate.
    
    Scoring thresholds (beats per minute):
        - ≤40: Score 3 (Critical - severe bradycardia)
        - 41-50: Score 1 (Mild bradycardia)
        - 51-90: Score 0 (Normal range)
        - 91-110: Score 1 (Mild tachycardia)
        - 111-130: Score 2 (Moderate tachycardia)
        - ≥131: Score 3 (Critical - severe tachycardia)
    
    Args:
        hr: Heart rate in beats per minute.
    
    Returns:
        int: NEWS2 score component (0-3).
    """
    return 3 if hr <= 40 else (1 if hr <= 50 else (0 if hr <= 90 else (1 if hr <= 110 else (2 if hr <= 130 else 3))))

def score_consc(consciousness: str) -> int:
    """Calculate NEWS2 score for level of consciousness.
    
    Uses the AVPU scale:
        - A (Alert): Score 0 - Patient is fully alert and oriented
        - V (Voice): Score 3 - Patient responds to voice commands
        - P (Pain): Score 3 - Patient responds only to painful stimuli
        - U (Unresponsive): Score 3 - Patient is unresponsive
    
    Any deviation from 'Alert' scores 3 points, triggering immediate concern.
    
    Args:
        consciousness: AVPU level as single character ('A', 'V', 'P', or 'U').
    
    Returns:
        int: NEWS2 score component (0 or 3).
    """
    return 0 if consciousness.upper() == "A" else 3

# ------------------------------------------------------------------------------

def calculate(v: Vitals) -> int:
    """Calculate the total NEWS2 score from all vital parameters.
    
    Sums scores from all seven physiological parameters. Special handling
    for Scale 2 patients where the total might be just the SpO2 component.
    
    Risk Classification based on total score:
        - 0: MINIMAL risk - routine monitoring
        - 1-4: LOW risk - ward-based care
        - 5-6: MODERATE risk - urgent response, increase monitoring frequency
        - 7+: HIGH risk - emergency response, consider ICU
        - Any single parameter = 3: MODERATE risk minimum (triggers clinical review)
    
    Args:
        v: Vitals object containing all physiological parameters.
    
    Returns:
        int: Total NEWS2 score (0-20 range).
    """
    s = (score_rr(v.rr) + score_spo2(v.spo2, v.spo2_scale, v.on_o2) + 
         score_o2(v.on_o2) + score_temp(v.temp) + score_bp(v.sys_bp) + 
         score_hr(v.hr) + score_consc(v.consciousness))
    return 0 if v.spo2_scale == 2 and s == score_spo2(v.spo2, v.spo2_scale, v.on_o2) else s

@app.post("/news2", summary="Calculate NEWS2 (partial or complete JSON)")
def news2_route(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """User-friendly endpoint for Postman. Send only the fields you want to change."""
    vitals = Vitals(**{**Vitals().model_dump(), **payload})
    
    scores = {
        "rr": score_rr(vitals.rr),
        "spo2": score_spo2(vitals.spo2, vitals.spo2_scale, vitals.on_o2),
        "o2": score_o2(vitals.on_o2),
        "temp": score_temp(vitals.temp),
        "sys_bp": score_bp(vitals.sys_bp),
        "hr": score_hr(vitals.hr),
        "consciousness": score_consc(vitals.consciousness)
    }
    
    total = calculate(vitals)
    return {"component_scores": scores, "total_score": total}

def _risk_from_score(score: int) -> str:
    """Convert NEWS2 score to risk classification string.
    
    Args:
        score: Total NEWS2 score.
    
    Returns:
        str: Risk level ('HIGH', 'MODERATE', 'LOW', or 'MINIMAL').
    """
    return 'HIGH' if score >= 7 else ('MODERATE' if score >= 5 else ('LOW' if score >= 1 else 'MINIMAL'))

def _forward_to_cloud(grouped: DefaultDict[str, List[Dict[str, Any]]]):
    base = os.environ.get("CLOUD_BASE_URL", "http://127.0.0.1:9000")
    # Priority order for sending: HIGH > MODERATE > LOW > MINIMAL
    priority_order = ["HIGH", "MODERATE", "LOW", "MINIMAL"]
    for risk in priority_order:
        items = grouped.get(risk, [])
        if not items:
            continue
        path = risk.lower()  # "HIGH" → "high", "MODERATE" → "moderate", etc.
        url = f"{base}/cloud/ingest/{path}"
        log.info(f"Forwarding {len(items)} items to cloud ({risk})")
        try:
            requests.post(url, json=items, timeout=10)
            log.info(f"Successfully forwarded {len(items)} items to cloud ({risk})")
        except Exception as e:
            # Best-effort forwarding; don't fail the response to edge
            log.error(f"Failed to forward to cloud {url}: {e}")

@app.post("/vispac/upload_batch")
async def upload_batch(req: Request, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    start_time = time.time()
    ctype = req.headers.get("X-Compression-Type","none").lower()
    raw = await req.body()
    log.info(f"Received batch: {len(raw)} bytes, compression={ctype}")
    
    process_start = time.time()
    scores, to_cloud = _process_batch_payload(ctype, raw)
    process_duration = (time.time() - process_start) * 1000
    
    log.info(f"Processed batch: {len(scores)} patients, forwarding to cloud")
    background_tasks.add_task(_forward_to_cloud, to_cloud)
    
    # HTTP requests return immediately after processing, backgrounding the forwarding
    # We can't measure forwarding time here, but we can measure processing time
    total_duration = (time.time() - start_time) * 1000
    log.info(f"[FOG_METRICS] type=http batch_size={len(scores)} process_ms={process_duration:.2f} forward_ms=0.00 total_ms={total_duration:.2f}")
    
    return {"batch_processed": len(scores), "scores": scores}


def _process_batch_payload(ctype: str, raw_bytes: bytes):
    """Decodes the payload (same logic as HTTP endpoint) and calculates scores.
    Returns (scores_dict, grouped_items_for_cloud)."""
    try:
        if ctype=="lzw":
            payload = LZW().decompress(raw_bytes.decode())
        elif ctype in ("huffman", "hushman"):
            data = json.loads(raw_bytes)
            payload = Huffman().decompress(data["payload"], data["codes"], data.get("padding", 0))
        else:
            payload = raw_bytes.decode()
        batch = json.loads(payload)
    except Exception as e:
        raise HTTPException(400, f"Failed to decode batch: {e}")

    scores = {}
    to_cloud: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for pkg in batch:
        # Start with defaults
        vidict = Vitals().model_dump()
        
        # Use vitals sent by edge if available
        if "vitals" in pkg:
            # Update with provided vitals (keeping defaults for missing fields)
            for key, value in pkg["vitals"].items():
                if value is not None:
                    vidict[key] = value
        else:
            # Fallback for legacy packets without vitals
            signal = pkg.get("signal")
            if pkg.get("data"):
                vidict[signal] = pkg["data"][-1][1]
            if "forced_vitals" in pkg:
                for key, value in pkg["forced_vitals"].items():
                    if value is not None:
                        vidict[key] = value
        
        # Debug: Check for None values before calculation
        none_fields = [k for k, v in vidict.items() if v is None]
        if none_fields:
            log.warning(f"Patient {pkg.get('patient_id')}: None values in fields: {none_fields}, vidict={vidict}")
        
        try:
            total = calculate(Vitals(**vidict))
        except Exception as e:
            log.error(f"Calculate error for patient {pkg.get('patient_id')}: {e}, vidict={vidict}")
            raise
        scores[pkg["patient_id"]] = total
        risco = _risk_from_score(total)
    # Forward a snapshot of vitals used in calculation; omit raw_size/post_sdt_size
        to_store = {
            "patient_id": pkg.get("patient_id"),
            "signal": pkg.get("signal"),
            "data": pkg.get("data"),
            "forced_vitals": pkg.get("forced_vitals"),
            "score": total,
            "risk": risco,
            # vitals snapshot
            "hr": vidict.get("hr"),
            "spo2": vidict.get("spo2"),
            "rr": vidict.get("rr"),
            "temp": vidict.get("temp"),
            "sys_bp": vidict.get("sys_bp"),
            "on_o2": vidict.get("on_o2"),
            "spo2_scale": vidict.get("spo2_scale"),
            "consciousness": vidict.get("consciousness"),
        }
        to_cloud[risco].append(to_store)
    return scores, to_cloud


if __name__ == "__main__":
    # FOG layer API
    log.info("Starting FOG API on port 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_config=None)
