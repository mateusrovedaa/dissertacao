# vispac_edge.py

import json
import time
import requests
from compressors import LZW, Huffman, SwingingDoor
from typing import Dict

class Configurator:
    """Ajusta os parâmetros de coleta com base no risco clínico (NEWS2)."""
    def get_config(self, news2_score: int):
        if news2_score >= 7: risk = "ALTO"; config = {'ic_hr': 30, 'ic_spo2': 30, 'deviation_hr': 2, 'deviation_spo2': 1}
        elif 5 <= news2_score <= 6: risk = "MODERADO"; config = {'ic_hr': 120, 'ic_spo2': 180, 'deviation_hr': 5, 'deviation_spo2': 1}
        elif 1 <= news2_score <= 4: risk = "BAIXO"; config = {'ic_hr': 300, 'ic_spo2': 600, 'deviation_hr': 5, 'deviation_spo2': 2}
        else: risk = "MÍNIMO"; config = {'ic_hr': 1200, 'ic_spo2': 1800, 'deviation_hr': 10, 'deviation_spo2': 3}
        return risk, config

class ViSPACEdge:
    """Implementação final do ViSPAC Edge com lógica de gatilho unificada."""
    def __init__(self, patient_id: str, api_url: str):
        self.patient_id, self.api_url = patient_id, api_url
        self.configurator, self.lzw = Configurator(), LZW()
        self.risk, self.config = self.configurator.get_config(0)
        self.sdt_hr = SwingingDoor(self.config['deviation_hr'])
        self.sdt_spo2 = SwingingDoor(self.config['deviation_spo2'])
        self.data_buffer, self.last_transmission_time = [], 0

    def process_and_send(self, vital_signs: dict, current_sim_time: float):
        hr_point, spo2_point = (current_sim_time, vital_signs['hr']), (current_sim_time, vital_signs['spo2'])
        
        is_significant_change = bool(self.sdt_hr.compress(hr_point) or self.sdt_spo2.compress(spo2_point))

        if is_significant_change:
            self.data_buffer.append({"time": current_sim_time, **vital_signs})
            print(f"[{self.patient_id}] Ponto de dado significativo adicionado ao buffer.")

        # --- LÓGICA DE GATILHO UNIFICADA E FINAL ---
        # Se houver QUALQUER mudança significativa em paciente de baixo/moderado risco, UMA transmissão é enviada.
        if is_significant_change and self.risk in ["MÍNIMO", "BAIXO", "MODERADO"]:
            print(f"[{self.patient_id}] GATILHO POR DADOS (Malha de Segurança): Mudança detectada. Enviando para reavaliação.")
            self.transmit_buffer(vital_signs, current_sim_time)
        
        # O gatilho por tempo só é verificado se o gatilho por dados não foi acionado
        elif (current_sim_time - self.last_transmission_time) >= self.config['ic_hr'] or \
             (current_sim_time - self.last_transmission_time) >= self.config['ic_spo2']:
            print(f"[{self.patient_id}] GATILHO POR TEMPO: Intervalo de checagem atingido.")
            self.transmit_buffer(vital_signs, current_sim_time)

    def transmit_buffer(self, vitals: Dict, sim_time: float):
        """Transmite todo o buffer acumulado."""
        if not self.data_buffer:
            # Se o gatilho for por tempo mas não houver dados novos, não faz nada.
            print(f"[{self.patient_id}] Gatilho por tempo, mas sem dados novos no buffer. Nenhuma transmissão.")
            return

        package = {"patient_id": self.patient_id, "data": self.data_buffer}
        self.transmit(package, vitals, sim_time)
        self.data_buffer = [] # Limpa o buffer após o envio
        self.last_transmission_time = sim_time # Atualiza o temporizador único

    def transmit(self, package: Dict, current_vitals: Dict, sim_time: float):
        """Método central que comprime e envia qualquer pacote de dados."""
        original_str = json.dumps(package)
        payload = json.dumps(self.lzw.compress(original_str))
        print(f"[{self.patient_id}] Transmissão disparada com LZW...")
        self.send_to_api(payload, "lzw", current_vitals, len(original_str.encode('utf-8')))

    def send_to_api(self, payload: str, compressor: str, vitals: Dict, size: int=0):
        # (código interno deste método permanece o mesmo)
        headers = {"X-Compression-Type": compressor, "Content-Type": "application/json"}
        try:
            response = requests.post(self.api_url, data=payload, headers=headers)
            if response.status_code == 200:
                data = response.json()
                score = data['total_score']
                print(f"[{self.patient_id}] Notificação da API recebida: NEWS2 Score = {score}")
                self.risk, self.config = self.configurator.get_config(score)
                self.sdt_hr.deviation, self.sdt_spo2.deviation = self.config['deviation_hr'], self.config['deviation_spo2']
                print(f"[{self.patient_id}] AJUSTE: Risco={self.risk}, Novos Intervalos(HR/SpO2)={self.config['ic_hr']}s/{self.config['ic_spo2']}s")
            else:
                print(f"[{self.patient_id}] Erro na API: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"[{self.patient_id}] Falha na conexão com a API: {e}")

    def flush_buffer(self, final_vitals: Dict, sim_time: float):
        if self.data_buffer:
            print(f"[{self.patient_id}] Forçando transmissão final do buffer...")
            self.transmit_buffer(final_vitals, sim_time)