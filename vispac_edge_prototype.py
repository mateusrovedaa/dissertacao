# vispac_edge_prototype.py

import time
import random
import json
import requests
import pandas as pd
from compressors import SwingingDoorCompressor, LZW, Huffman

# --- Constantes e Configurações ---
API_URL = "http://127.0.0.1:8000/vispac/upload_batch"
DATASET_PATH = "dataset.csv"
LOG_FILE_PATH = "simulation_log.txt"
K_STABLE_READINGS = 3
NUM_PATIENTS = 10
IC_VERIF_SECS = 45 # Algoritmo 3: verificação a cada 45 segundos

# --- Configurações do Empacotador por Prioridade (Timeout = IC_min / 2) ---
ASSEMBLER_CONFIG = {
    "ALTO":     {"timeout": 15, "size_limit": 2 * 1024},    # IC_min = 30s
    "MODERADO": {"timeout": 30, "size_limit": 20 * 1024},   # IC_min = 60s
    "BAIXO":    {"timeout": 60, "size_limit": 50 * 1024},   # IC_min = 120s
    "MÍNIMO":   {"timeout": 90, "size_limit": 50 * 1024},   # IC_min = 180s
}

# --- Configurações da Compressão de Saída ---
# Pacotes menores que 1MB não usarão compressão lossless para evitar sobrecarga.
LOSSLESS_MIN_SIZE_BYTES = 1024 * 1024 # 1 MB

# --- Algoritmo 1: Configurador por Sinal Vital ---
def get_vispac_config(news2_score):
    risco = "MÍNIMO"
    if news2_score >= 7: risco = "ALTO"
    elif 5 <= news2_score <= 6: risco = "MODERADO"
    elif 1 <= news2_score <= 4: risco = "BAIXO"
    params_map = {
        "ALTO": {"ic_fc": 30, "eps_fc": 2, "dc_fc": 2, "ic_spo2": 30, "eps_spo2": 1, "dc_spo2": 1, "t_sdt": 15},
        "MODERADO": {"ic_fc": 60, "eps_fc": 5, "dc_fc": 5, "ic_spo2": 90, "eps_spo2": 1, "dc_spo2": 1, "t_sdt": 1*60},
        "BAIXO": {"ic_fc": 120, "eps_fc": 5, "dc_fc": 5, "ic_spo2": 150, "eps_spo2": 2, "dc_spo2": 2, "t_sdt": 3*60},
        "MÍNIMO": {"ic_fc": 180, "eps_fc": 10, "dc_fc": 10, "ic_spo2": 240, "eps_spo2": 3, "dc_spo2": 3, "t_sdt": 5*60}
    }
    config = params_map[risco]
    config['risco'] = risco
    return config

def log_message(message, file_handle):
    print(message)
    file_handle.write(message + "\n")

class Patient:
    def __init__(self, patient_id, dataset):
        self.patient_id = patient_id
        self.dataset = dataset
        self.dataset_index = random.randint(0, len(dataset) - 1)
        self.current_news2 = 0
        self.params = get_vispac_config(self.current_news2)
        self.fc_buffer, self.spo2_buffer = [], []
        self.last_fc_collection_time = time.time()
        self.last_spo2_collection_time = time.time()
        initial_vitals = self.get_next_vital(peek=True)
        self.last_sent_fc = initial_vitals['hr']
        self.last_sent_spo2 = initial_vitals['spo2']
        self.current_ic_fc = self.params['ic_fc']
        self.current_ic_spo2 = self.params['ic_spo2']

    def get_next_vital(self, peek=False):
        row = self.dataset.iloc[self.dataset_index]
        vitals = {"timestamp": time.time(), "hr": float(row['hr']), "spo2": float(row['spo2'])}
        if not peek:
            self.dataset_index = (self.dataset_index + 1) % len(self.dataset)
        return vitals

    def update_risk_and_params(self, new_news2_score, log_file):
        log_message(f"\n[PACIENTE {self.patient_id}] Recebido novo NEWS2: {new_news2_score}. Risco anterior: {self.params['risco']}", log_file)
        self.current_news2 = new_news2_score
        self.params = get_vispac_config(new_news2_score)
        self.current_ic_fc = self.params['ic_fc']
        self.current_ic_spo2 = self.params['ic_spo2']
        log_message(f"-> Novo Risco: {self.params['risco']}. Parâmetros de coleta e compressão atualizados.", log_file)

class Assembler:
    """Módulo Empacotador com filas de prioridade baseadas no risco."""
    def __init__(self, config):
        self.queues = {
            risk: {
                "buffer": [], 
                "current_size": 0, 
                "first_data_timestamp": None, 
                "config": conf
            }
            for risk, conf in config.items()
        }

    def add_data(self, package, risk, log_file):
        queue = self.queues[risk]
        if not queue["buffer"]:
            queue["first_data_timestamp"] = time.time()
        queue["buffer"].append(package)
        queue["current_size"] += len(json.dumps(package).encode('utf-8'))
        log_message(f"  [Empacotador] Pacote adicionado à fila '{risk}'. Tamanho da fila: {queue['current_size']} / {queue['config']['size_limit']} bytes.", log_file)

    def get_ready_batches(self):
        """Verifica todas as filas e retorna os lotes prontos para envio."""
        ready_batches = []
        for risk, queue in self.queues.items():
            if not queue["buffer"]:
                continue
            
            size_reached = queue["current_size"] >= queue["config"]["size_limit"]
            timeout_reached = (time.time() - queue["first_data_timestamp"]) >= queue["config"]["timeout"]
            
            if size_reached or timeout_reached:
                reason = "timeout" if timeout_reached else "tamanho"
                batch = queue["buffer"]
                ready_batches.append({"batch": batch, "risk": risk, "reason": reason})
                # Reseta a fila
                queue["buffer"] = []
                queue["current_size"] = 0
                queue["first_data_timestamp"] = None
        return ready_batches

def create_high_risk_package(batch, log_file):
    log_message("\n*** SIMULANDO EVENTO DE RISCO ALTO ***", log_file)
    if not batch: return batch
    high_risk_package = batch[0].copy()
    high_risk_package['forced_vitals'] = {'hr': 140, 'spo2': 85, 'rr': 30, 'consciousness': 'V', 'temp': 43}
    batch[0] = high_risk_package
    return batch

def process_and_send_batch(batch_info, log_file, force_high_risk=False):
    batch = batch_info["batch"]
    risk = batch_info["risk"]
    log_message("\n" + "="*20 + f" [ENVIO DE LOTE - RISCO {risk}] " + "="*20, log_file)
    t_amostra = time.time()
    
    if force_high_risk:
        batch = create_high_risk_package(batch, log_file)
    
    has_high_risk_patient = any('forced_vitals' in p for p in batch) or any(p.get('risco') == 'ALTO' for p in batch)
    uncompressed_payload_str = json.dumps(batch)
    true_original_size = len(uncompressed_payload_str.encode('utf-8'))
    
    payload_to_send, compression_header = None, "none"
    
    if has_high_risk_patient:
        log_message("-> [Compressão de Saída] Lote contém paciente de Risco ALTO. Pulando compressão para baixa latência.", log_file)
        payload_to_send = uncompressed_payload_str.encode('utf-8')
    elif true_original_size < LOSSLESS_MIN_SIZE_BYTES:
        log_message(f"-> [Compressão de Saída] Lote ({true_original_size}b) menor que o limiar ({LOSSLESS_MIN_SIZE_BYTES}b). Pulando compressão lossless.", log_file)
        payload_to_send = uncompressed_payload_str.encode('utf-8')
    else:
        log_message("-> [Compressão de Saída] Lote grande, aplicando LZW.", log_file)
        compressed_package = LZW().compress(uncompressed_payload_str)
        payload_to_send = json.dumps(compressed_package).encode('utf-8')
        compression_header = "lzw"

    final_size = len(payload_to_send)
    headers = {"X-Compression-Type": compression_header, "Content-Type": "application/json"}

    try:
        log_message(f"-> [Enviador] Enviando lote. Tamanho: {true_original_size}b -> {final_size}b.", log_file)
        response = requests.post(API_URL, data=payload_to_send, headers=headers, timeout=15)
        response.raise_for_status()
        
        t_ajuste = time.time()
        api_response = response.json()
        
        tc = (final_size / true_original_size) * 100 if true_original_size > 0 else 0
        ed = 100 - tc
        t_loop = t_ajuste - t_amostra
        
        log_message(f"   -> Resposta da API recebida: {api_response}", log_file)
        log_message(f"   -> MÉTRICAS DO LOTE: T_C: {tc:.2f}% | Economia (ED): {ed:.2f}% | T_loop: {t_loop:.4f}s", log_file)
        
        return api_response.get("scores", {})

    except requests.exceptions.RequestException as e:
        log_message(f"\n[ERRO DE REDE] Não foi possível conectar à API: {e}", log_file)
        return None

def main_simulation_loop(log_file):
    sdt_compressor = SwingingDoorCompressor()
    try:
        dataset = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        log_message(f"[ERRO] Dataset '{DATASET_PATH}' não encontrado. Encerrando.", log_file)
        return

    patients = [Patient(f"PID-{i:03d}", dataset) for i in range(1, NUM_PATIENTS + 1)]
    assembler = Assembler(config=ASSEMBLER_CONFIG)
    
    log_message("="*50, log_file)
    log_message(f"Iniciando Simulação ViSPAC com {NUM_PATIENTS} Pacientes e Filas de Prioridade", log_file)
    log_message("="*50, log_file)
    
    batch_send_count = 0
    last_keep_alive_check_time = time.time()

    while True:
        current_time = time.time()
        
        if (current_time - last_keep_alive_check_time) >= IC_VERIF_SECS:
            log_message("\n[ALGORITMO 3] Realizando verificação da malha de segurança (Keep-Alive)...", log_file)
            for p in patients:
                vitals = p.get_next_vital(peek=True)
                if abs(vitals['hr'] - p.last_sent_fc) > p.params['eps_fc']:
                    log_message(f"!!! [ALGORITMO 3] Variação anômala em FC para {p.patient_id}! Disparando coleta imediata.", log_file)
                    p.last_fc_collection_time = 0
                if abs(vitals['spo2'] - p.last_sent_spo2) > p.params['eps_spo2']:
                    log_message(f"!!! [ALGORITMO 3] Variação anômala em SpO2 para {p.patient_id}! Disparando coleta imediata.", log_file)
                    p.last_spo2_collection_time = 0
            last_keep_alive_check_time = current_time

        for patient in patients:
            vitals = patient.get_next_vital()
            patient.fc_buffer.append((vitals['timestamp'], vitals['hr']))
            patient.spo2_buffer.append((vitals['timestamp'], vitals['spo2']))
            
            if (current_time - patient.last_fc_collection_time) >= patient.current_ic_fc:
                log_message(f"\n--- [COLETA FC] Paciente: {patient.patient_id} (Risco: {patient.params['risco']}) ---", log_file)
                
                # CORREÇÃO: Cria o pacote hipotético com dados brutos para medir o tamanho original real
                original_package = {"patient_id": patient.patient_id, "risco": patient.params['risco'], "signal": "hr", "data": patient.fc_buffer}
                true_original_size = len(json.dumps(original_package).encode('utf-8'))

                compressed_fc = sdt_compressor.compress(patient.fc_buffer, patient.params['dc_fc'], patient.params['t_sdt'])
                
                if compressed_fc:
                    # Cria o pacote real com os dados comprimidos pelo SDT
                    final_package = {"patient_id": patient.patient_id, "risco": patient.params['risco'], "signal": "hr", "data": compressed_fc}
                    post_sdt_size = len(json.dumps(final_package).encode('utf-8'))
                    sdt_reduction = (1 - (post_sdt_size / true_original_size)) * 100 if true_original_size > 0 else 0
                    log_message(f"-> [Compressão SDT] Tamanho do Pacote: {true_original_size}b -> {post_sdt_size}b (Economia: {sdt_reduction:.2f}%)", log_file)
                    assembler.add_data(final_package, patient.params['risco'], log_file)

                patient.fc_buffer = []
                patient.last_fc_collection_time = current_time
            
            if (current_time - patient.last_spo2_collection_time) >= patient.current_ic_spo2:
                log_message(f"\n--- [COLETA SpO2] Paciente: {patient.patient_id} (Risco: {patient.params['risco']}) ---", log_file)
                
                original_package = {"patient_id": patient.patient_id, "risco": patient.params['risco'], "signal": "spo2", "data": patient.spo2_buffer}
                true_original_size = len(json.dumps(original_package).encode('utf-8'))

                compressed_spo2 = sdt_compressor.compress(patient.spo2_buffer, patient.params['dc_spo2'], patient.params['t_sdt'])
                
                if compressed_spo2:
                    final_package = {"patient_id": patient.patient_id, "risco": patient.params['risco'], "signal": "spo2", "data": compressed_spo2}
                    post_sdt_size = len(json.dumps(final_package).encode('utf-8'))
                    sdt_reduction = (1 - (post_sdt_size / true_original_size)) * 100 if true_original_size > 0 else 0
                    log_message(f"-> [Compressão SDT] Tamanho do Pacote: {true_original_size}b -> {post_sdt_size}b (Economia: {sdt_reduction:.2f}%)", log_file)
                    assembler.add_data(final_package, patient.params['risco'], log_file)

                patient.spo2_buffer = []
                patient.last_spo2_collection_time = current_time

        ready_batches = assembler.get_ready_batches()
        for batch_info in ready_batches:
            log_message(f"\n[Empacotador] Gatilho de envio para a fila '{batch_info['risk']}' atingido por {batch_info['reason']}!", log_file)
            batch_send_count += 1
            
            force_high_risk_event = (batch_send_count == 2)
            
            feedback_scores = process_and_send_batch(batch_info, log_file, force_high_risk=force_high_risk_event)
            
            if feedback_scores:
                log_message("\n[Feedback Loop] Aplicando scores recebidos aos pacientes...", log_file)
                for patient_id, new_score in feedback_scores.items():
                    for p in patients:
                        if p.patient_id == patient_id:
                            p.update_risk_and_params(new_score, log_file)
                            break
        
        queue_sizes = " | ".join([f"{risk[:3]}: {q['current_size']}b" for risk, q in assembler.queues.items()])
        print(f"\r[{time.strftime('%H:%M:%S')}] Monitorando... Filas: [ {queue_sizes} ]", end="")
        time.sleep(1)

if __name__ == "__main__":
    with open(LOG_FILE_PATH, "w") as log_file:
        main_simulation_loop(log_file)
