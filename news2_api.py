# news2_api.py

from typing import Dict, Any, Optional
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import uvicorn
import json
from compressors import LZW, Huffman

app = FastAPI(title="ViSPAC Scoring & Decompression API")

# Modelo Pydantic atualizado para ser mais robusto, com valores padrão
class Vitals(BaseModel):
    rr: float = 18
    spo2: float = 98
    spo2_scale: int = 1
    on_o2: bool = False
    temp: float = 36.5
    sys_bp: float = 120
    hr: float = 80
    consciousness: str = "A"

# --- Funções de Cálculo de Escore (sem alterações) ---
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
    """Calcula o escore NEWS2 para um conjunto de sinais vitais."""
    scores = {
        "respiratory_rate": score_rr(v.rr),
        "oxygen_saturation": score_spo2(v.spo2, v.spo2_scale, v.on_o2),
        "supplemental_o2": score_o2_supplemental(v.on_o2),
        "temperature": score_temp(v.temp),
        "systolic_bp": score_sys_bp(v.sys_bp),
        "heart_rate": score_hr(v.hr),
        "consciousness_level": score_consciousness(v.consciousness),
    }
    total_score = sum(scores.values())
    # A pontuação da escala 2 de SpO2 não pode ser a única a contribuir com o total se for > 0
    if v.spo2_scale == 2 and scores["oxygen_saturation"] > 0 and total_score == scores["oxygen_saturation"]:
        total_score = 0

    return {"component_scores": scores, "total_score": total_score}

@app.post("/vispac/upload_batch")
async def handle_compressed_batch(request: Request) -> Dict[str, Any]:
    """
    NOVO: Endpoint para receber, descomprimir e processar um lote de dados de pacientes.
    """
    compression_type = request.headers.get("X-Compression-Type", "none").lower()
    body = await request.body()
    
    try:
        decompressed_str = ""
        if compression_type == "lzw":
            # O corpo é uma lista de inteiros em formato JSON string
            compressed_list = json.loads(body)
            decompressed_str = LZW().decompress(compressed_list)
        elif compression_type == "huffman":
            # O corpo é um dicionário JSON com 'payload' e 'codes'
            data = json.loads(body)
            decompressed_str = Huffman().decompress(data['payload'], data['codes'])
        else: # "none"
            decompressed_str = body.decode('utf-8')
        
        batch_data = json.loads(decompressed_str)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro no processamento do lote: {e}")

    scores = {}
    print("\n--- [API] Lote Recebido para Processamento ---")
    # Processa cada pacote de paciente no lote
    for package in batch_data:
        patient_id = package.get('patient_id')
        if not patient_id:
            continue
        
        # Inicia com um modelo de sinais vitais padrão
        vitals_dict = Vitals().model_dump()

        # Verifica se é uma simulação de risco alto
        if 'forced_vitals' in package:
            print(f"  -> [API] Paciente {patient_id}: Detectado pacote de risco alto simulado.")
            vitals_dict.update(package['forced_vitals'])
        else:
            # Pega o último valor do sinal para o cálculo do score
            signal_type = package.get('signal')
            if package.get('data'):
                last_value = package['data'][-1][1]
                vitals_dict[signal_type] = last_value
        
        # Cria o modelo Pydantic a partir do dicionário final
        vitals_model = Vitals(**vitals_dict)
        
        # SAÍDA SOLICITADA: Mostra os valores que serão usados para o cálculo
        print(f"  -> [API] Calculando score para {patient_id} com os valores: {vitals_model.model_dump()}")
        
        score_result = calculate_news2(vitals_model)
        scores[patient_id] = score_result['total_score']

    print("--- [API] Processamento do Lote Concluído ---\n")
    return {"batch_processed": len(scores), "scores": scores}


if __name__ == "__main__":
    uvicorn.run("news2_api:app", host="0.0.0.0", port=8000, reload=True)
