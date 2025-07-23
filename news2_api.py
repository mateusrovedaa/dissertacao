from typing import Dict, Any
from fastapi import FastAPI, Request, HTTPException, Body
from pydantic import BaseModel
import json, uvicorn
from compressors import LZW, Huffman

app = FastAPI(title="ViSPAC Scoring & Decompression API v2")

class Vitals(BaseModel):
    rr: float = 18;   spo2: float = 98; spo2_scale: int = 1
    on_o2: bool = False; temp: float = 36.5; sys_bp: float = 120
    hr: float = 80; consciousness: str = "A"

# --- scorers (unchanged) ----------------------------------------------------

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

@app.post("/news2", summary="Calcula NEWS2 (JSON parcial ou completo)")
def news2_route(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """Endpoint amigável para Postman. Envie apenas os campos que deseja alterar."""
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

@app.post("/vispac/upload_batch")
async def upload_batch(req: Request) -> Dict[str, Any]:
    ctype = req.headers.get("X-Compression-Type","none")
    raw = await req.body()
    try:
        if ctype=="lzw":
            payload = LZW().decompress(raw.decode())
        elif ctype=="huffman":
            data = json.loads(raw)
            payload = Huffman().decompress(data["payload"], data["codes"])
        else:
            payload = raw.decode()
        batch = json.loads(payload)
    except Exception as e:
        raise HTTPException(400, f"Falha ao decodificar lote: {e}")

    scores = {}
    for pkg in batch:
        vidict = Vitals().model_dump()
        if "forced_vitals" in pkg:
            vidict.update(pkg["forced_vitals"])
        else:
            signal = pkg.get("signal")
            if pkg.get("data"):
                vidict[signal] = pkg["data"][-1][1]
        scores[pkg["patient_id"]] = calculate(Vitals(**vidict))
    return {"batch_processed": len(scores), "scores": scores}

if __name__ == "__main__":
    uvicorn.run("__main__:app", host="0.0.0.0", port=8000, reload=True)