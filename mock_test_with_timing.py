# mock_test_with_timing.py

import time
import json
import requests
from vispac_edge import ViSPACEdge
from typing import Dict

DADOS_FIXOS_MOCK = {
    "paciente_A_estavel": [
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = []

    def send_to_api(self, payload: str, compressor: str, vitals: Dict, size: int=0):
        t_start = time.perf_counter_ns()
        super().send_to_api(payload, compressor, vitals, size)
        t_end = time.perf_counter_ns()
        self.metrics.append({
            "loop_time_ms": (t_end - t_start) / 1_000_000,
            "original_size": size, "compressed_size": len(payload.encode('utf-8')),
        })

    def print_summary_report(self):
        if not self.metrics:
            print(f"\n--- Relatório Final para {self.patient_id} ---\n  Nenhuma transmissão foi realizada.")
            return
        print(f"\n--- Relatório Final para {self.patient_id} ---")
        avg_loop = sum(m['loop_time_ms'] for m in self.metrics) / len(self.metrics)
        total_orig = sum(m['original_size'] for m in self.metrics)
        total_comp = sum(m['compressed_size'] for m in self.metrics)
        reduction = (1 - total_comp / total_orig) * 100 if total_orig > 0 else 0
        print(f"  - Transmissões Realizadas: {len(self.metrics)}")
        print(f"  - Tempo Médio de Loop (T_loop): {avg_loop:.2f} ms")
        print(f"  - Taxa de Redução de Dados: {reduction:.2f}%")

def run_fixed_mock_test():
    API_URL = "http://127.0.0.1:8000/vispac/upload"
    TIME_STEP = 15
    testers = {pid: ViSPACEdgeTester(patient_id=pid, api_url=API_URL) for pid in DADOS_FIXOS_MOCK}
    print("--- Iniciando Teste com Lógica Final ---")
    num_steps = max(len(data) for data in DADOS_FIXOS_MOCK.values())
    sim_clock = 0
    for i in range(num_steps):
        sim_clock += TIME_STEP
        print(f"\n--- [Passo de Tempo: {i+1} | Relógio Simulado: {sim_clock}s] ---")
        for pid, tester in testers.items():
            if i < len(DADOS_FIXOS_MOCK[pid]):
                vitals = DADOS_FIXOS_MOCK[pid][i]
                print(f"Lendo Sinais para [{pid}]: HR={vitals['hr']}, SpO2={vitals['spo2']}")
                tester.process_and_send(vitals, sim_clock)
        time.sleep(0.1)
    print("\n--- Fim da Coleta. Forçando transmissão dos buffers restantes. ---")
    for pid, tester in testers.items():
        tester.flush_buffer(DADOS_FIXOS_MOCK[pid][-1], sim_clock)
    for tester in testers.values():
        tester.print_summary_report()

if __name__ == "__main__":
    run_fixed_mock_test()