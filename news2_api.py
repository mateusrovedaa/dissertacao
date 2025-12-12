from typing import Dict, Any, List, DefaultDict
from collections import defaultdict
from fastapi import FastAPI, Request, HTTPException, Body, BackgroundTasks
from pydantic import BaseModel
import json, uvicorn
import os, requests
try:
    import paho.mqtt.client as mqtt
except Exception:
    mqtt = None
from compressors import LZW, Huffman

app = FastAPI(title="ViSPAC Scoring & Decompression API v2")

class Vitals(BaseModel):
    rr: float = 18;   spo2: float = 98; spo2_scale: int = 1
    on_o2: bool = False; temp: float = 36.5; sys_bp: float = 120
    hr: float = 80; consciousness: str = "A"

# --- scoring functions ------------------------------------------

def score_rr(rr): return 3 if rr<=8 else (1 if rr<=11 else (0 if rr<=20 else (2 if rr<=24 else 3)))

def score_spo2(spo2, scale, on_o2):
    if scale==2:
        if spo2<=83: return 3
        if spo2<=85: return 2
        if spo2<=87: return 1
        if not on_o2 and spo2<=92: return 0
        if on_o2:
            if spo2<=92: return 0
            if spo2<=94: return 1
            if spo2<=96: return 2
            return 3
    else:
        if spo2<=91: return 3
        if spo2<=93: return 2
        if spo2<=95: return 1
        return 0

def score_o2(o): return 2 if o else 0

def score_temp(t): return 3 if t<=35 else (1 if t<=36 else (0 if t<=38 else (1 if t<=39 else 2)))

def score_bp(bp): return 3 if bp<=90 else (2 if bp<=100 else (1 if bp<=110 else (0 if bp<=219 else 3)))

def score_hr(hr): return 3 if hr<=40 else (1 if hr<=50 else (0 if hr<=90 else (1 if hr<=110 else (2 if hr<=130 else 3))))

def score_consc(c): return 0 if c.upper()=="A" else 3

# ---------------------------------------------------------------------------

def calculate(v: Vitals) -> int:
    s = score_rr(v.rr)+score_spo2(v.spo2,v.spo2_scale,v.on_o2)+score_o2(v.on_o2)+score_temp(v.temp)+score_bp(v.sys_bp)+score_hr(v.hr)+score_consc(v.consciousness)
    return 0 if v.spo2_scale==2 and s==score_spo2(v.spo2,v.spo2_scale,v.on_o2) else s

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
        try:
            requests.post(url, json=items, timeout=10)
        except Exception as e:
            # Best-effort forwarding; don't fail the response to edge
            print(f"[FOG] Failed to forward to cloud {url}: {e}")

@app.post("/vispac/upload_batch")
async def upload_batch(req: Request, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    ctype = req.headers.get("X-Compression-Type","none").lower()
    raw = await req.body()
    scores, to_cloud = _process_batch_payload(ctype, raw)
    background_tasks.add_task(_forward_to_cloud, to_cloud)
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
        vidict = Vitals().model_dump()
        if "forced_vitals" in pkg:
            vidict.update(pkg["forced_vitals"])
        else:
            signal = pkg.get("signal")
            if pkg.get("data"):
                vidict[signal] = pkg["data"][-1][1]
        total = calculate(Vitals(**vidict))
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


@app.on_event("startup")
def _start_mqtt_listener():
    """If paho-mqtt is available, starts an MQTT client in background
    to receive messages from edge and respond with scores."""
    if mqtt is None:
        print("[FOG] paho-mqtt not installed; MQTT listener disabled.")
        return

    broker = os.environ.get('MQTT_BROKER','127.0.0.1')
    port = int(os.environ.get('MQTT_PORT','1883'))
    client = mqtt.Client()

    def on_message(client, userdata, message):
        try:
            data = json.loads(message.payload.decode())
            reply_topic = data.get('reply_topic')
            ctype = data.get('X-Compression-Type','none').lower()
            raw_payload = data.get('payload','').encode()
            try:
                scores, to_cloud = _process_batch_payload(ctype, raw_payload)
            except HTTPException as e:
                # publish error
                if reply_topic:
                    client.publish(reply_topic, json.dumps({'error': str(e)}))
                return
            # forward to cloud
            _forward_to_cloud(to_cloud)
            # respond
            if reply_topic:
                client.publish(reply_topic, json.dumps({'scores': scores}))
        except Exception as e:
            print(f"[FOG MQTT] handler error: {e}")

    client.on_message = on_message
    try:
        client.connect(broker, port, 60)
        client.subscribe('vispac/upload_batch')
        client.loop_start()
        print(f"[FOG] MQTT listener started on {broker}:{port} topic vispac/upload_batch")
    except Exception as e:
        print(f"[FOG] MQTT connect failed: {e}")

if __name__ == "__main__":
    # FOG layer API
    uvicorn.run("__main__:app", host="0.0.0.0", port=8000, reload=True)
