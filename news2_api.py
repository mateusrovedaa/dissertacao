# news2_api.py

from typing import Dict, Any
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import uvicorn
import json
from compressors import LZW, Huffman

app = FastAPI(title="ViSPAC Scoring & Decompression API")

class Vitals(BaseModel):
    rr: float
    spo2: float
    spo2_scale: int = 1
    on_o2: bool = False
    temp: float
    sys_bp: float
    hr: float
    consciousness: str

def score_rr(rr: float) -> int:
    if rr <= 8: return 3
    if 9 <= rr <= 11: return 1
    if 12 <= rr <= 20: return 0
    if 21 <= rr <= 24: return 2
    return 3
def score_spo2(spo2: float, scale: int, on_o2: bool) -> int:
    if scale == 2:
        if spo2 <= 83: return 3
        if 84 <= spo2 <= 85: return 2
        if 86 <= spo2 <= 87: return 1
        if not on_o2 and 88 <= spo2 <= 92: return 0
        if on_o2:
            if 88 <= spo2 <= 92: return 0
            if 93 <= spo2 <= 94: return 1
            if 95 <= spo2 <= 96: return 2
            if spo2 >= 97: return 3
    else:
        if spo2 <= 91: return 3
        if 92 <= spo2 <= 93: return 2
        if 94 <= spo2 <= 95: return 1
        return 0
    return 0
def score_o2_supplemental(on_o2: bool) -> int: return 2 if on_o2 else 0
def score_temp(temp: float) -> int:
    if temp <= 35.0: return 3
    if 35.1 <= temp <= 36.0: return 1
    if 36.1 <= temp <= 38.0: return 0
    if 38.1 <= temp <= 39.0: return 1
    return 2
def score_sys_bp(bp: float) -> int:
    if bp <= 90: return 3
    if 91 <= bp <= 100: return 2
    if 101 <= bp <= 110: return 1
    if 111 <= bp <= 219: return 0
    return 3
def score_hr(hr: float) -> int:
    if hr <= 40: return 3
    if 41 <= hr <= 50: return 1
    if 51 <= hr <= 90: return 0
    if 91 <= hr <= 110: return 1
    if 111 <= hr <= 130: return 2
    return 3
def score_consciousness(level: str) -> int:
    lvl = level.upper()
    if lvl == 'A': return 0
    if lvl in ('C', 'V', 'P', 'U'): return 3
    raise ValueError(f"Invalid consciousness level: {level}")

@app.post("/news2")
def calculate_news2(v: Vitals) -> Dict[str, Any]:
    scores = {
        "respiratory_rate": score_rr(v.rr), "oxygen_saturation": score_spo2(v.spo2, v.spo2_scale, v.on_o2),
        "supplemental_o2": score_o2_supplemental(v.on_o2), "temperature": score_temp(v.temp),
        "systolic_bp": score_sys_bp(v.sys_bp), "heart_rate": score_hr(v.hr),
        "consciousness_level": score_consciousness(v.consciousness),
    }
    scores["total"] = sum(scores.values())
    return {"component_scores": scores, "total_score": scores["total"]}

@app.post("/vispac/upload")
async def handle_compressed_data(request: Request) -> Dict[str, Any]:
    compression_type = request.headers.get("X-Compression-Type", "none").lower()
    body = await request.body()
    try:
        if compression_type == "lzw":
            decompressed_str = LZW().decompress(json.loads(body))
        elif compression_type == "huffman":
            data = json.loads(body)
            decompressed_str = Huffman().decompress(data['payload'], data['codes'])
        else:
            decompressed_str = body.decode('utf-8')
        package_data = json.loads(decompressed_str)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro no processamento do pacote: {e}")

    latest_vitals = package_data.get('data', [{}])[-1]
    
    # Adiciona valores default para todos os campos para garantir que o modelo Vitals não falhe
    required_fields = Vitals.model_fields.keys()
    for field in required_fields:
        if field not in latest_vitals:
            latest_vitals[field] = Vitals.model_fields[field].default
    
    vitals_model = Vitals(**latest_vitals)
    score_result = calculate_news2(vitals_model)

    return {"patient_id": package_data.get('patient_id'), "total_score": score_result['total_score']}

if __name__ == "__main__":
    uvicorn.run("news2_api:app", host="0.0.0.0", port=8000, reload=True)