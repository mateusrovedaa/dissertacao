# vispac_edge_prototype.py

import time
import random
import json
import requests
import pandas as pd
from compressors import SwingingDoorCompressor, LZW, Huffman

# --- Constantes e Configurações ---
API_URL = "http://127.0.0.1:8000/vispac/upload"
DATASET_PATH = "dataset.csv"
K_STABLE_READINGS = 3 
HUFFMAN_SIZE_THRESHOLD = 33 * 1024
IC_VERIF_SECS = 10 * 60 # Algoritmo 3: verificação a cada 10 minutos

# --- Algoritmo 1: Configurador por Sinal Vital ---
def get_vispac_config(news2_score):
    risco = "MÍNIMO"
    if news2_score >= 7: risco = "ALTO"
    elif 5 <= news2_score <= 6: risco = "MODERADO"
    elif 1 <= news2_score <= 4: risco = "BAIXO"

    params_map = {
        "ALTO": {"ic_fc": 30, "eps_fc": 2, "dc_fc": 2, "ic_spo2": 30, "eps_spo2": 1, "dc_spo2": 1, "t_sdt": 15, "ic_max_fc": 30, "ic_max_spo2": 30},
        "MODERADO": {"ic_fc": 2*60, "eps_fc": 5, "dc_fc": 5, "ic_spo2": 3*60, "eps_spo2": 1, "dc_spo2": 1, "t_sdt": 1*60, "ic_max_fc": 30*60, "ic_max_spo2": 30*60},
        "BAIXO": {"ic_fc": 5*60, "eps_fc": 5, "dc_fc": 5, "ic_spo2": 10*60, "eps_spo2": 2, "dc_spo2": 2, "t_sdt": 3*60, "ic_max_fc": 2*3600, "ic_max_spo2": 2*3600},
        "MÍNIMO": {"ic_fc": 10*60, "eps_fc": 10, "dc_fc": 10, "ic_spo2": 15*60, "eps_spo2": 3, "dc_spo2": 3, "t_sdt": 5*60, "ic_max_fc": 6*3600, "ic_max_spo2": 6*3600}
    }
    config = params_map[risco]
    config['risco'] = risco
    return config

class Patient:
    def __init__(self, patient_id, dataset_path):
        self.patient_id = patient_id
        self.dataset = None
        self.dataset_index = 0
        self.load_dataset(dataset_path)
        
        self.current_news2 = 0
        self.params = get_vispac_config(self.current_news2)
        
        self.stable_fc_count = 0
        self.stable_spo2_count = 0
        
        initial_vitals = self.get_next_vital(peek=True)
        self.last_sent_fc = initial_vitals['hr']
        self.last_sent_spo2 = initial_vitals['spo2']
        
        self.current_ic_fc = self.params['ic_fc']
        self.current_ic_spo2 = self.params['ic_spo2']

    def load_dataset(self, path):
        try:
            self.dataset = pd.read_csv(path)
            print(f"[PACIENTE] Dataset '{path}' carregado com {len(self.dataset)} registros.")
        except FileNotFoundError:
            print(f"[ERRO] Dataset '{path}' não encontrado. Encerrando.")
            exit()

    def get_next_vital(self, peek=False):
        if self.dataset is None: return None
        row = self.dataset.iloc[self.dataset_index]
        vitals = {"timestamp": time.time(), "hr": row['hr'], "spo2": row['spo2']}
        if not peek:
            self.dataset_index = (self.dataset_index + 1) % len(self.dataset)
        return vitals

    def update_risk_and_params(self, new_news2_score):
        print(f"\n[PACIENTE {self.patient_id}] Recebido novo NEWS2: {new_news2_score}. Risco anterior: {self.params['risco']}")
        self.current_news2 = new_news2_score
        self.params = get_vispac_config(new_news2_score)
        self.stable_fc_count = 0
        self.stable_spo2_count = 0
        self.current_ic_fc = self.params['ic_fc']
        self.current_ic_spo2 = self.params['ic_spo2']
        print(f"-> Novo Risco: {self.params['risco']}. Parâmetros de coleta e compressão atualizados.")

    def apply_backoff(self, signal_type, latest_value):
        if self.params['risco'] == "ALTO": return
        if signal_type == 'fc':
            if abs(latest_value - self.last_sent_fc) <= self.params['eps_fc']:
                self.stable_fc_count += 1
            else:
                self.stable_fc_count = 0
                self.current_ic_fc = self.params['ic_fc']
            if self.stable_fc_count >= K_STABLE_READINGS:
                new_ic = min(self.params['ic_max_fc'], self.current_ic_fc * 2)
                if new_ic > self.current_ic_fc:
                    print(f"[ALGORITMO 2] FC estável. Aplicando back-off. IC FC: {self.current_ic_fc}s -> {new_ic}s")
                    self.current_ic_fc = new_ic
                self.stable_fc_count = 0
            self.last_sent_fc = latest_value
        elif signal_type == 'spo2':
            if abs(latest_value - self.last_sent_spo2) <= self.params['eps_spo2']:
                self.stable_spo2_count += 1
            else:
                self.stable_spo2_count = 0
                self.current_ic_spo2 = self.params['ic_spo2']
            if self.stable_spo2_count >= K_STABLE_READINGS:
                new_ic = min(self.params['ic_max_spo2'], self.current_ic_spo2 * 2)
                if new_ic > self.current_ic_spo2:
                    print(f"[ALGORITMO 2] SpO2 estável. Aplicando back-off. IC SpO2: {self.current_ic_spo2}s -> {new_ic}s")
                    self.current_ic_spo2 = new_ic
                self.stable_spo2_count = 0
            self.last_sent_spo2 = latest_value

def process_and_send_data(patient, compressed_data, signal_type, original_size):
    t_amostra = time.time()
    last_vital_value = compressed_data[-1][1] if compressed_data else 0
    package_data = {
        "patient_id": patient.patient_id,
        "data": [{"timestamp": d[0], signal_type: d[1]} for d in compressed_data],
        "rr": 18, "spo2_scale": 1, "on_o2": False, "temp": 36.5, "sys_bp": 120, "consciousness": 'A'
    }
    package_data[signal_type] = last_vital_value
    
    uncompressed_payload_str = json.dumps(package_data)
    payload_to_send = None
    compression_header = "none"

    if patient.params['risco'] != "ALTO":
        uncompressed_size = len(uncompressed_payload_str.encode('utf-8'))
        if uncompressed_size <= HUFFMAN_SIZE_THRESHOLD:
            print("-> [Compressão de Saída] Pacote pequeno, aplicando Huffman.")
            compressed_package = Huffman().compress(uncompressed_payload_str)
            payload_to_send = json.dumps(compressed_package).encode('utf-8')
            compression_header = "huffman"
        else:
            print("-> [Compressão de Saída] Pacote grande, aplicando LZW.")
            compressed_package = LZW().compress(uncompressed_payload_str)
            payload_to_send = json.dumps(compressed_package).encode('utf-8')
            compression_header = "lzw"
    else:
        print("-> [Compressão de Saída] Risco ALTO. Pulando compressão de saída para baixa latência.")
        payload_to_send = uncompressed_payload_str.encode('utf-8')

    headers = {"X-Compression-Type": compression_header, "Content-Type": "application/json"}
    try:
        print(f"-> [Enviador] Enviando pacote de {len(payload_to_send)} bytes para a API...")
        response = requests.post(API_URL, data=payload_to_send, headers=headers, timeout=10)
        response.raise_for_status()
        
        t_ajuste = time.time()
        api_response = response.json()
        new_score = api_response.get("total_score")
        
        tc = (len(payload_to_send) / original_size) * 100 if original_size > 0 else 0
        t_loop = t_ajuste - t_amostra
        
        print(f"   -> Resposta da API recebida. Novo NEWS2: {new_score}")
        print(f"   -> MÉTRICAS: Taxa de Compressão Final (T_C): {tc:.2f}% | Latência do Ciclo (T_loop): {t_loop:.4f}s")
        
        # Atualiza o risco do paciente e aplica o backoff com o valor que foi enviado
        patient.update_risk_and_params(new_score)
        patient.apply_backoff(signal_type, last_vital_value)

    except requests.exceptions.RequestException as e:
        print(f"[ERRO DE REDE] Não foi possível conectar à API: {e}")

def main_simulation_loop():
    patient = Patient(patient_id="PID-001", dataset_path=DATASET_PATH)
    sdt_compressor = SwingingDoorCompressor()
    
    fc_buffer, spo2_buffer = [], []
    last_fc_collection_time = time.time()
    last_spo2_collection_time = time.time()
    last_keep_alive_check_time = time.time()
    
    print("="*50)
    print("Iniciando Simulação do Protótipo ViSPAC - Dispositivo de Borda")
    print(f"Paciente: {patient.patient_id}, Risco Inicial: {patient.params['risco']}")
    print("="*50)

    while True:
        current_time = time.time()
        vitals = patient.get_next_vital()
        if vitals is None: break
        
        fc_buffer.append((vitals['timestamp'], vitals['hr']))
        spo2_buffer.append((vitals['timestamp'], vitals['spo2']))
        
        if int(current_time) % 10 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] Monitorando... FC: {vitals['hr']:.1f}, SpO2: {vitals['spo2']:.1f} | "
                  f"IC FC: {patient.current_ic_fc:.0f}s, IC SpO2: {patient.current_ic_spo2:.0f}s")

        # --- Algoritmo 3: Malha de Segurança (Keep-Alive) ---
        if (current_time - last_keep_alive_check_time) >= IC_VERIF_SECS:
            print("\n[ALGORITMO 3] Realizando verificação da malha de segurança (Keep-Alive)...")
            current_fc = vitals['hr']
            current_spo2 = vitals['spo2']
            
            if abs(current_fc - patient.last_sent_fc) > patient.params['eps_fc']:
                print("!!! [ALGORITMO 3] Variação anômala em FC detectada! Disparando coleta imediata.")
                last_fc_collection_time = 0 # Força a coleta no próximo ciclo
            
            if abs(current_spo2 - patient.last_sent_spo2) > patient.params['eps_spo2']:
                print("!!! [ALGORITMO 3] Variação anômala em SpO2 detectada! Disparando coleta imediata.")
                last_spo2_collection_time = 0 # Força a coleta no próximo ciclo
            
            last_keep_alive_check_time = current_time

        # --- Gatilho de Coleta Principal (baseado no IC) ---
        if (current_time - last_fc_collection_time) >= patient.current_ic_fc:
            print(f"\n--- [GATILHO DE COLETA FC] --- Tempo de IC ({patient.current_ic_fc:.0f}s) atingido.")
            original_size = len(json.dumps(fc_buffer).encode('utf-8'))
            compressed_fc = sdt_compressor.compress(fc_buffer, patient.params['dc_fc'], patient.params['t_sdt'])
            if compressed_fc:
                process_and_send_data(patient, compressed_fc, "hr", original_size)
            fc_buffer = []
            last_fc_collection_time = current_time

        if (current_time - last_spo2_collection_time) >= patient.current_ic_spo2:
            print(f"\n--- [GATILHO DE COLETA SpO2] --- Tempo de IC ({patient.current_ic_spo2:.0f}s) atingido.")
            original_size = len(json.dumps(spo2_buffer).encode('utf-8'))
            compressed_spo2 = sdt_compressor.compress(spo2_buffer, patient.params['dc_spo2'], patient.params['t_sdt'])
            if compressed_spo2:
                process_and_send_data(patient, compressed_spo2, "spo2", original_size)
            spo2_buffer = []
            last_spo2_collection_time = current_time

        time.sleep(1)

if __name__ == "__main__":
    main_simulation_loop()
