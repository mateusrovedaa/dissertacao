# test_no_compression.py

import time
import json
import requests
from typing import Dict, Any

# Copiamos os dados e o Tester, mas usaremos uma classe Edge diferente
DADOS_FIXOS_MOCK = {
    # (Use os mesmos dados do mock_test_with_timing.py)
}

class EdgeNoCompression:
    """Uma versão do Edge que NÃO realiza NENHUMA compressão."""
    def __init__(self, patient_id: str, api_url: str):
        self.patient_id = patient_id
        self.api_url = api_url
        self.data_buffer = []
        # Não precisa de Configurator, pois o envio é sempre no final
    
    def process_and_send(self, vital_signs: dict, current_sim_time: float):
        # Apenas adiciona todos os pontos ao buffer, sem compressão lossy
        self.data_buffer.append({"time": current_sim_time, **vital_signs})
        print(f"[{self.patient_id}] Ponto de dado adicionado ao buffer (sem SDT).")

    def flush_buffer(self, final_vitals: Dict):
        print(f"[{self.patient_id}] Forçando transmissão final (sem compressão)...")
        if not self.data_buffer:
            print(f"[{self.patient_id}] Buffer final vazio.")
            return

        payload = json.dumps({"patient_id": self.patient_id, "data": self.data_buffer})
        
        # O tamanho original e comprimido são os mesmos
        original_size = len(payload.encode('utf-8'))
        
        self.send_to_api(payload, original_size, final_vitals)
        self.data_buffer = []

    def send_to_api(self, payload: str, original_size: int, vitals: Dict):
        # O header indica que não há compressão
        headers = {"X-Compression-Type": "none", "Content-Type": "application/json"}
        # A lógica de medição e relatório será feita na classe Tester
        # Esta classe base apenas envia.
        try:
            requests.post(self.api_url, data=payload, headers=headers)
        except requests.exceptions.RequestException as e:
            print(f"[{self.patient_id}] Falha na conexão com a API: {e}")

class NoCompressionTester(EdgeNoCompression):
    """Classe de teste para o cenário sem compressão."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = []

    def send_to_api(self, payload: str, original_size: int, vitals: Dict):
        t_loop_start = time.perf_counter_ns()
        super().send_to_api(payload, original_size, vitals)
        t_loop_end = time.perf_counter_ns()
        
        self.metrics.append({
            "loop_time_ms": (t_loop_end - t_loop_start) / 1_000_000,
            "data_size": original_size,
        })
    
    def print_summary_report(self):
        if not self.metrics:
            print(f"\n--- Relatório (Sem Compressão) para {self.patient_id} ---")
            print("  Nenhuma transmissão foi realizada.")
            return

        print(f"\n--- Relatório (Sem Compressão) para {self.patient_id} ---")
        avg_loop_time = sum(m['loop_time_ms'] for m in self.metrics) / len(self.metrics)
        total_data = sum(m['data_size'] for m in self.metrics)
        
        print(f"  - Tempo Médio de Loop (T_loop): {avg_loop_time:.2f} ms")
        print(f"  - Tamanho Total dos Dados Enviados: {total_data} bytes")

def run_no_compression_test():
    API_UPLOAD_URL = "http://127.0.0.1:8000/vispac/upload"
    testers = {pid: NoCompressionTester(patient_id=pid, api_url=API_UPLOAD_URL) for pid in DADOS_FIXOS_MOCK}
    
    print("\n--- Iniciando Teste de Base (SEM COMPRESSÃO) ---")
    # A simulação é idêntica, mas usa a classe NoCompressionTester
    # ... (código do loop principal de run_fixed_mock_test) ...
    
    for tester in testers.values():
        tester.print_summary_report()

if __name__ == "__main__":
    # Cole os dados mock e o loop principal aqui para tornar o script executável
    run_no_compression_test()