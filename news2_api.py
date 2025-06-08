from typing import Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="NEWS2 Scoring API")

class Vitals(BaseModel):
    rr: float
    spo2: float
    spo2_scale: int = 1
    on_o2: bool = False
    temp: float
    sys_bp: float
    hr: float
    consciousness: str  # 'A', 'C', 'V', 'P', or 'U'

def score_rr(rr: float) -> int:
    if rr <= 8: return 3
    if 9 <= rr <= 11: return 1
    if 12 <= rr <= 20: return 0
    if 21 <= rr <= 24: return 2
    return 3  # rr >=25

def score_spo2(spo2: float, scale: int, on_o2: bool) -> int:
    if scale == 2:
        if spo2 <= 83: return 3
        if 84 <= spo2 <= 85: return 2
        if 86 <= spo2 <= 87: return 1
        # spo2 >= 88 and <= 92 and in air
        if not on_o2:
            return 0
        # on supplemental O2, use the extended bands
        if 88 <= spo2 <= 92: return 0
        if 93 <= spo2 <= 94: return 1
        if 95 <= spo2 <= 96: return 2
        if spo2 >= 97: return 3
    else:
        if spo2 <= 91: return 3
        if 92 <= spo2 <= 93: return 2
        if 94 <= spo2 <= 95: return 1
        return 0

def score_o2_supplemental(on_o2: bool) -> int:
    return 2 if on_o2 else 0

def score_temp(temp: float) -> int:
    if temp <= 35.0: return 3
    if 35.1 <= temp <= 36.0: return 1
    if 36.1 <= temp <= 38.0: return 0
    if 38.1 <= temp <= 39.0: return 1
    return 2  # temp >=39.1

def score_sys_bp(bp: float) -> int:
    if bp <= 90: return 3
    if 91 <= bp <= 100: return 2
    if 101 <= bp <= 110: return 1
    if 111 <= bp <= 219: return 0
    return 3  # bp >=220

def score_hr(hr: float) -> int:
    if hr <= 40: return 3
    if 41 <= hr <= 50: return 1
    if 51 <= hr <= 90: return 0
    if 91 <= hr <= 110: return 1
    if 111 <= hr <= 130: return 2
    return 3  # hr >=131

def score_consciousness(level: str) -> int:
    lvl = level.upper()
    if lvl == 'A': return 0
    if lvl in ('C', 'V', 'P', 'U'): return 3
    raise ValueError(f"Invalid consciousness level: {level}")

@app.post("/news2")
def calculate_news2(v: Vitals) -> Dict[str, Any]:
    scores = {
        "respiratory_rate":    score_rr(v.rr),
        "oxygen_saturation":   score_spo2(v.spo2, v.spo2_scale, v.on_o2),
        "supplemental_o2":     score_o2_supplemental(v.on_o2),
        "temperature":         score_temp(v.temp),
        "systolic_bp":         score_sys_bp(v.sys_bp),
        "heart_rate":          score_hr(v.hr),
        "consciousness_level": score_consciousness(v.consciousness),
    }
    scores["total"] = sum(scores.values())
    return {"component_scores": scores, "total_score": scores["total"]}

if __name__ == "__main__":
    uvicorn.run("news2_api:app", host="0.0.0.0", port=8000, reload=True)