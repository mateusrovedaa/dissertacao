# vispac_edge.py

import json
import time
import requests
from compressors import LZW, Huffman, SwingingDoor
from typing import Dict

class Configurator:
    def get_config(self, news2_score: int):
        if news2_score >= 7: risk = "ALTO"; config = {'ic': 30, 'deviation_hr': 2, 'deviation_spo2': 1}
        elif 5 <= news2_score <= 6: risk = "MODERADO"; config = {'ic': 120, 'deviation_hr': 5, 'deviation_spo2': 1}
        elif 1 <= news2_score <= 4: risk = "BAIXO"; config = {'ic': 300, 'deviation_hr': 5, 'deviation_spo2': 2}
        else: risk = "MÍNIMO"; config = {'ic': 600, 'deviation_hr': 10, 'deviation_spo2': 3}
        return risk, config

class ViSPACEdge:
    def __init__(self, patient_id: str, api_url: str):
        self.patient_id, self.api_url = patient_id, api_url
        self.configurator, self.lzw, self.huffman = Configurator(), LZW(), Huffman()
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

        if is_significant_change and self.risk in ["MÍNIMO", "BAIXO"]:
            print(f"[{self.patient_id}] GATILHO POR DADOS (Malha de Segurança): Mudança detectada. Enviando para reavaliação.")
            self.transmit_data(vital_signs, current_sim_time)
        elif (current_sim_time - self.last_transmission_time) >= self.config['ic']:
            print(f"[{self.patient_id}] GATILHO POR TEMPO: Intervalo de {self.config['ic']}s atingido.")
            self.transmit_data(vital_signs, current_sim_time)

    def transmit_data(self, current_vitals: Dict, current_sim_time: float):
        if not self.data_buffer:
            self.data_buffer.append({"time": current_sim_time, **current_vitals})
        original_package_str = json.dumps({"patient_id": self.patient_id, "data": self.data_buffer})
        payload = json.dumps(self.lzw.compress(original_package_str))
        print(f"[{self.patient_id}] Transmissão disparada com LZW...")
        self.send_to_api(payload, "lzw", current_vitals, len(original_package_str.encode('utf-8')))
        self.data_buffer, self.last_transmission_time = [], current_sim_time

    def send_to_api(self, payload: str, compressor_name: str, vitals: Dict, original_size: int = 0):
        headers = {"X-Compression-Type": compressor_name, "Content-Type": "application/json"}
        try:
            response = requests.post(self.api_url, data=payload, headers=headers)
            if response.status_code == 200:
                data = response.json()
                news2_score = data['total_score']
                print(f"[{self.patient_id}] Notificação da API recebida: NEWS2 Score = {news2_score}")
                self.risk, self.config = self.configurator.get_config(news2_score)
                self.sdt_hr.deviation, self.sdt_spo2.deviation = self.config['deviation_hr'], self.config['deviation_spo2']
                print(f"[{self.patient_id}] AJUSTE: Risco={self.risk}, Novo Intervalo={self.config['ic']}s, Nova DeviationHR={self.config['deviation_hr']}")
            else:
                print(f"[{self.patient_id}] Erro na API: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"[{self.patient_id}] Falha na conexão com a API: {e}")

    def flush_buffer(self, final_vitals: Dict, current_sim_time: float):
        if self.data_buffer: self.transmit_data(final_vitals, current_sim_time)