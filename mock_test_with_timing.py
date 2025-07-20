# mock_test_with_timing.py

import time
import json
import requests
from vispac_edge import ViSPACEdge 
from typing import Dict, Any

DADOS_FIXOS_MOCK = {
    "paciente_A_estavel": [
        # Dados completamente planos para garantir que o SwingingDoor não dispare.
        {'hr': 75, 'spo2': 98, 'rr': 15, 'temp': 36.8, 'sys_bp': 120, 'consciousness': 'A', 'on_o2': False, 'spo2_scale': 1}
    ] * 8,
    "paciente_B_critico": [
        {'hr': 80, 'spo2': 97, 'rr': 18, 'temp': 37.1, 'sys_bp': 125, 'consciousness': 'A', 'on_o2': False, 'spo2_scale': 1},
        {'hr': 95, 'spo2': 95, 'rr': 22, 'temp': 37.5, 'sys_bp': 110, 'consciousness': 'A', 'on_o2': False, 'spo2_scale': 1},
        {'hr': 112, 'spo2': 93, 'rr': 24, 'temp': 38.2, 'sys_bp': 105, 'consciousness': 'A', 'on_o2': True, 'spo2_scale': 1},
        {'hr': 115, 'spo2': 92, 'rr': 25, 'temp': 38.3, 'sys_bp': 100, 'consciousness': 'A', 'on_o2': True, 'spo2_scale': 1},
        {'hr': 135, 'spo2': 90, 'rr': 28, 'temp': 38.5, 'sys_bp': 88, 'consciousness': 'V', 'on_o2': True, 'spo2_scale': 1},
        {'hr': 138, 'spo2': 89, 'rr': 30, 'temp': 38.6, 'sys_bp': 85, 'consciousness': 'V', 'on_o2': True, 'spo2_scale': 1},
        {'hr': 140, 'spo2': 89, 'rr': 31, 'temp': 38.7, 'sys_bp': 84, 'consciousness': 'V', 'on_o2': True, 'spo2_scale': 1},
        {'hr': 135, 'spo2': 90, 'rr': 28, 'temp': 38.5, 'sys_bp': 88, 'consciousness': 'V', 'on_o2': True, 'spo2_scale': 1},
    ]
}

class ViSPACEdgeTester(ViSPACEdge):
    """Classe de teste que herda da ViSPACEdge para adicionar medição de tempo."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = []

    def send_to_api(self, payload: str, compressor_name: str, vitals: Dict, original_size: int):
        """
        Sobrescreve o método pai para medir o tempo do loop de comunicação
        e depois chama o método pai para executar a ação.
        """
        t_loop_start = time.perf_counter_ns()
        
        # Chama o método da classe pai para fazer a chamada de rede real
        super().send_to_api(payload, compressor_name, vitals, original_size)
        
        t_loop_end = time.perf_counter_ns()
        
        compressed_size = len(payload.encode('utf-8'))
        self.metrics.append({
            "loop_time_ms": (t_loop_end - t_loop_start) / 1_000_000,
            "original_size": original_size,
            "compressed_size": compressed_size,
        })

    def print_summary_report(self):
        """Imprime um relatório de desempenho consolidado para o paciente."""
        if not self.metrics:
            print(f"\n--- Relatório Final para {self.patient_id} ---")
            print("  Nenhuma transmissão foi realizada (além da inicial, se houve).")
            print("--------------------------------------------------")
            return

        print(f"\n--- Relatório Final para {self.patient_id} ---")
        avg_loop_time = sum(m['loop_time_ms'] for m in self.metrics) / len(self.metrics)
        total_original = sum(m['original_size'] for m in self.metrics)
        total_compressed = sum(m['compressed_size'] for m in self.metrics)
        reduction = (1 - total_compressed / total_original) * 100 if total_original > 0 else 0
        
        print(f"  - Transmissões Realizadas: {len(self.metrics)}")
        print(f"  - Tempo Médio de Loop (T_loop): {avg_loop_time:.2f} ms")
        print(f"  - Taxa de Redução de Dados: {reduction:.2f}%")
        print("--------------------------------------------------")

def run_fixed_mock_test():
    """Função principal que orquestra a simulação com múltiplos pacientes."""
    API_UPLOAD_URL = "http://127.0.0.1:8000/vispac/upload"
    SIMULATED_TIME_PER_STEP_SECONDS = 15
    
    testers = {pid: ViSPACEdgeTester(patient_id=pid, api_url=API_UPLOAD_URL) for pid in DADOS_FIXOS_MOCK}
    
    print("--- Iniciando Teste com Lógica Final ---")
    num_steps = max(len(data) for data in DADOS_FIXOS_MOCK.values())
    simulation_clock = 0

    for i in range(num_steps):
        simulation_clock += SIMULATED_TIME_PER_STEP_SECONDS
        print(f"\n--- [Passo de Tempo: {i+1} | Relógio Simulado: {simulation_clock}s] ---")
        
        for patient_id, tester in testers.items():
            if i < len(DADOS_FIXOS_MOCK[patient_id]):
                vitals = DADOS_FIXOS_MOCK[patient_id][i]
                print(f"Lendo Sinais para [{patient_id}]: HR={vitals['hr']}, SpO2={vitals['spo2']}")
                tester.process_and_send(vitals, simulation_clock)
        
        time.sleep(0.1)

    print("\n--- Fim da Coleta. Forçando transmissão dos buffers restantes. ---")
    for patient_id, tester in testers.items():
        final_vitals = DADOS_FIXOS_MOCK[patient_id][-1]
        tester.flush_buffer(final_vitals, simulation_clock)

    for tester in testers.values():
        tester.print_summary_report()

if __name__ == "__main__":
    run_fixed_mock_test()