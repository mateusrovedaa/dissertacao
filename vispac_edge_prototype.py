import time, json, random, requests, logging, os, uuid
import pandas as pd
import numpy as np
from compressors import SwingingDoorCompressor, Huffman, LZW

# ---------------- Constantes ----------------
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000/vispac/upload_batch")
DATASET_PATH = "dataset.csv"
LOG_PATH = "logs/edge_log.txt"
K_STABLE = 3
KEEP_ALIVE = 600
HUFF_MIN, LZW_MIN = 1 * 1024, 32 * 1024

ASSEMBLER_CONFIG = {
    "ALTO":     {"timeout": 15,  "size_limit": 5 * 1024},
    "MODERADO": {"timeout": 60,  "size_limit": 20 * 1024},
    "BAIXO":    {"timeout": 150, "size_limit": 50 * 1024},
    "MÍNIMO":   {"timeout": 300, "size_limit": 50 * 1024},
}

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, 'w'), logging.StreamHandler()])
log = logging.getLogger("vispac-edge")

def reconstruct_signal(original_signal_buf, compressed_signal):
    """
    Reconstrói um sinal de mesmo tamanho que o original usando interpolação
    linear a partir dos pontos do sinal comprimido.
    """
    original_timestamps = [p[0] for p in original_signal_buf]
    compressed_timestamps = [p[0] for p in compressed_signal]
    compressed_values = [p[1] for p in compressed_signal]
    
    # Usa a função de interpolação do NumPy
    reconstructed_values = np.interp(original_timestamps, compressed_timestamps, compressed_values)
    return reconstructed_values

def calculate_prd(original_signal, reconstructed_signal):
    """
    Calcula a Taxa de Distorção (PRD) usando a fórmula padrão.
    """
    original = np.array(original_signal)
    reconstructed = np.array(reconstructed_signal)

    # Evita divisão por zero se o sinal original for nulo
    sum_sq_original = np.sum(original**2)
    if sum_sq_original == 0:
        return 0.0

    sum_sq_diff = np.sum((original - reconstructed)**2)
    prd = np.sqrt(sum_sq_diff / sum_sq_original) * 100
    return prd
# -----------------------------------------------------------------


# --------------- Dataset Simulado --------------------
SAMPLE_DATA=[(80,98),(81,98),(80,99),(82,98),(83,98),(82,97),(85,98),(84,98),(86,99),
    (88,98),(90,97),(92,96),(95,95),(98,94),(105,93),(110,92),(112,91),(115,90),
    (118,91),(120,92),(115,93),(110,94),(105,95),(100,96),(95,97),(92,98),(90,98),
    (88,99),(86,98),(85,98)]

def ensure_dataset():
    if not os.path.exists(DATASET_PATH):
        pd.DataFrame(SAMPLE_DATA, columns=['hr','spo2']).to_csv(DATASET_PATH,index=False)
    return pd.read_csv(DATASET_PATH)

# ------------- Parâmetros Alg. 1 ------------
PARAMS={
 "ALTO":    dict(ic_fc=30,  eps_fc=2,  dc_fc=2,  ic_spo2=30,  eps_spo2=1, dc_spo2=1, t_sdt=15,  ic_max=30*60),
 "MODERADO":dict(ic_fc=120, eps_fc=5,  dc_fc=5,  ic_spo2=180, eps_spo2=1, dc_spo2=1, t_sdt=60,  ic_max=30*60),
 "BAIXO":   dict(ic_fc=300, eps_fc=5,  dc_fc=5,  ic_spo2=600, eps_spo2=2, dc_spo2=2, t_sdt=180, ic_max=2*3600),
 "MÍNIMO":  dict(ic_fc=600, eps_fc=10, dc_fc=10, ic_spo2=900, eps_spo2=3, dc_spo2=3, t_sdt=300, ic_max=6*3600)
}

# --------------- Classes --------------------
class Patient:
    def __init__(self, pid, df, news2=0, persistent_high_risk=False):
        self.id=pid; self.df=df; self.idx=random.randrange(len(df))
        self.persistent_high_risk=persistent_high_risk
        self.scenario_step=0
        self.update_risk(news2)
        self.last_sent_fc=self._cur('hr'); self.last_sent_spo2=self._cur('spo2')
        self.fc_buf,self.spo2_buf=[],[]
        self.last_fc_col=self.last_spo2_col=time.time()
        self.stable_fc=self.stable_spo2=0
    def _cur(self,c):return float(self.df.iloc[self.idx][c])
    def next(self):
        row=self.df.iloc[self.idx]; self.idx=(self.idx+1)%len(self.df)
        return {"timestamp":time.time(),"hr":float(row['hr']),"spo2":float(row['spo2'])}

    def update_risk(self,score):
        if self.persistent_high_risk and score<7: score=7
        old=getattr(self,'risk','N/A')
        self.news2=score
        self.risk=('ALTO' if score>=7 else 'MODERADO' if score>=5 else 'BAIXO' if score>=1 else 'MÍNIMO')
        if self.risk!=old:
            log.info(f"[RISCO] {self.id}: {old} → {self.risk} (NEWS2={score})")
        p=PARAMS[self.risk]
        self.ic_fc,self.ic_spo2=p['ic_fc'],p['ic_spo2']
        self.eps_fc,self.eps_spo2=p['eps_fc'],p['eps_spo2']
        self.dc_fc,self.dc_spo2=p['dc_fc'],p['dc_spo2']
        self.t_sdt,self.ic_max=p['t_sdt'],p['ic_max']

    def backoff(self,sig,latest):
        if self.risk=='ALTO':return
        eps=self.eps_fc if sig=='hr' else self.eps_spo2
        st_attr='stable_fc' if sig=='hr' else 'stable_spo2'
        ic_attr='ic_fc' if sig=='hr' else 'ic_spo2'
        last_attr='last_sent_fc' if sig=='hr' else 'last_sent_spo2'
        st=getattr(self,st_attr); last=getattr(self,last_attr)
        if abs(latest-last)<=eps: st+=1
        else: st=0; setattr(self,ic_attr,PARAMS[self.risk][ic_attr])
        if st>=K_STABLE:
            cur=getattr(self,ic_attr); new=min(cur*2,self.ic_max)
            if new>cur: log.info(f"[BACKOFF] {self.id} {sig.upper()} {cur}s→{new}s")
            setattr(self,ic_attr,new); st=0
        setattr(self,st_attr,st); setattr(self,last_attr,latest)

    def forced_vitals(self):
        if self.id=="PID-004":
            cycle=self.scenario_step%3; self.scenario_step+=1
            if cycle==1: return {"hr":100,"temp":38.5}
            elif cycle==2: return {"hr":120,"sys_bp":95,"temp":39.5}
        elif self.id=="PID-005":
            if self.scenario_step==0: self.scenario_step+=1; return {"hr":120,"sys_bp":95,"temp":39.5}
            elif self.scenario_step==1: self.scenario_step+=1; return {"hr":100,"temp":38.5}
        elif self.persistent_high_risk:
            return {"hr":130,"spo2":86,"rr":30,"temp":39.5,"sys_bp":85,"consciousness":"V"}
        return {}

class Assembler:
    def __init__(self,cfg):
        self.queues={r:{"buffer":[],"size":0,"timestamp":None,"config":conf} for r,conf in cfg.items()}
    def add(self,pkg,risk):
        q=self.queues[risk]
        if not q['buffer']: q['timestamp']=time.time()
        q['buffer'].append(pkg); q['size']+=pkg['post_sdt_size']
        log.info(f"  [Empac] +{pkg['post_sdt_size']}b → fila {risk}={q['size']}b")
    def get_ready_batches(self):
        ready=[]
        for r,q in self.queues.items():
            if not q['buffer']: continue
            if q['size']>=q['config']['size_limit'] or (q['timestamp'] and time.time()-q['timestamp']>=q['config']['timeout']):
                reason="tamanho" if q['size']>=q['config']['size_limit'] else "timeout"
                ready.append({"batch":q['buffer'],"risk":r,"reason":reason})
                q['buffer'],q['size'],q['timestamp']=[],0,None
        return ready

# ------------- Compressão lossless -------------
def lossless(txt: str, risk: str):
    raw = txt.encode(); size=len(raw)
    if risk == 'ALTO' or size < HUFF_MIN: return raw, 'none'
    if size < LZW_MIN: return json.dumps(Huffman().compress(txt)).encode(), 'hushman'
    return json.dumps(LZW().compress(txt)).encode(), 'lzw'

# ----------- Envio do lote ----------------
def send_batch(batch_info):
    batch=batch_info['batch']; risk=batch_info['risk']
    total_raw_size = sum(p.get('raw_size', 0) for p in batch)
    payload_str=json.dumps(batch)
    raw_payload_size = len(payload_str.encode())
    payload, hdr = lossless(payload_str, risk)

    if hdr == 'none':
        if risk == 'ALTO':
            log.info(f"  [ETAPA 2] Compressão Lossless pulada (Risco ALTO).")
        elif raw_payload_size < HUFF_MIN:
            log.info(f"  [ETAPA 2] Compressão Lossless pulada (Lote {raw_payload_size}b < limiar {HUFF_MIN}b).")
    else:
        log.info(f"  [ETAPA 2] Compressão Lossless aplicada ({hdr}).")

    final_size=len(payload)
    start=time.time()
    # Opção MQTT: defina EDGE_USE_MQTT=1 e MQTT_BROKER / MQTT_PORT se necessário
    use_mqtt = os.environ.get('EDGE_USE_MQTT','0') in ('1','true','True')
    if use_mqtt:
        try:
            import paho.mqtt.client as mqtt
            resp_topic = f"vispac/resp/{uuid.uuid4()}"
            msg = json.dumps({'reply_topic': resp_topic, 'X-Compression-Type': hdr, 'payload': payload.decode()})

            q = []
            received = {'data': None}
            def _on_message(client, userdata, message):
                try:
                    received['data'] = json.loads(message.payload.decode())
                except Exception:
                    received['data'] = None

            client = mqtt.Client()
            client.on_message = _on_message
            broker = os.environ.get('MQTT_BROKER','127.0.0.1')
            port = int(os.environ.get('MQTT_PORT','1883'))
            client.connect(broker, port, 60)
            client.loop_start()
            client.subscribe(resp_topic)
            client.publish('vispac/upload_batch', msg)

            # wait for response (max 10s)
            timeout = 10
            waited = 0
            while waited < timeout and received['data'] is None:
                time.sleep(0.1); waited += 0.1
            client.loop_stop(); client.disconnect()
            resp = received['data'] or {}
            elapsed=time.time()-start
            ratio=100*final_size/total_raw_size if total_raw_size>0 else 0
            log.info(f"[ENVIO MQTT] Lote '{risk}' | {len(batch)} pkts | {total_raw_size}b→{final_size}b ({ratio:.1f}%) | {hdr} | {elapsed:.3f}s | scores={resp.get('scores')}")
            return resp.get('scores',{})
        except Exception as e:
            log.error(f"Falha ao enviar lote via MQTT: {e}")
            return {}

    try:
        r=requests.post(API_URL,data=payload,headers={'X-Compression-Type':hdr,'Content-Type':'application/json'},timeout=10)
        r.raise_for_status(); resp=r.json()
        elapsed=time.time()-start
        ratio=100*final_size/total_raw_size if total_raw_size>0 else 0
        log.info(f"[ENVIO] Lote '{risk}' | {len(batch)} pkts | {total_raw_size}b→{final_size}b ({ratio:.1f}%) | {hdr} | {elapsed:.3f}s | scores={resp.get('scores')}")
        return resp.get('scores',{})
    except Exception as e:
        log.error(f"Falha ao enviar lote: {e}")
        return {}

# --------------- MAIN LOOP ----------------
def main():
    df=ensure_dataset()
    patients=[
        Patient("PID-001",df,news2=11,persistent_high_risk=True),
        Patient("PID-002",df), Patient("PID-003",df), Patient("PID-004",df),
        Patient("PID-005",df,news2=5), Patient("PID-006",df), Patient("PID-007",df)
    ]
    assembler=Assembler(ASSEMBLER_CONFIG)
    sdt=SwingingDoorCompressor()

    prd_accumulator = {'hr': [], 'spo2': []}

    log.info(f"Simulação iniciada com {len(patients)} pacientes (PID-001 ALTO, PID-005 MODERADO)")
    last_keep=time.time()
    
    try:
        while True:
            now=time.time()
            if now-last_keep>=KEEP_ALIVE:
                log.info("[ALG3] Keep‑alive verificação …")
                for p in patients:
                    v=p.next()
                    if abs(v['hr']-p.last_sent_fc)>p.eps_fc: p.last_fc_col=0
                    if abs(v['spo2']-p.last_sent_spo2)>p.eps_spo2: p.last_spo2_col=0
                last_keep=now
            for p in patients:
                v=p.next(); p.fc_buf.append((v['timestamp'],v['hr'])); p.spo2_buf.append((v['timestamp'],v['spo2']))
                
                if now-p.last_fc_col>=p.ic_fc:
                    raw_size=len(json.dumps(p.fc_buf).encode())
                    comp=sdt.compress(p.fc_buf,p.dc_fc,p.t_sdt)
                    if comp:
                        post_sdt_size=len(json.dumps(comp).encode())
                        reduction=100*(1-post_sdt_size/raw_size) if raw_size>0 else 0
                        log.info(f"[COLETA FC] {p.id} {p.risk} | {len(p.fc_buf)}→{len(comp)} pts | {raw_size}b→{post_sdt_size}b ({reduction:.1f}%)")
                        
                        if p.risk == 'ALTO':
                            original_signal_values = [point[1] for point in p.fc_buf]
                            reconstructed_signal = reconstruct_signal(p.fc_buf, comp)
                            prd_value = calculate_prd(original_signal_values, reconstructed_signal)
                            prd_accumulator['hr'].append(prd_value)
                            log.info(f"  [DISTORÇÃO FC] {p.id} PRD={prd_value:.4f}%")

                        entry={'patient_id':p.id,'signal':'hr','risco':p.risk,'data':comp, 'raw_size':raw_size, 'post_sdt_size':post_sdt_size}
                        forced=p.forced_vitals(); 
                        if forced: entry['forced_vitals']=forced
                        assembler.add(entry,p.risk)
                    p.fc_buf=[]; p.last_fc_col=now; p.backoff('hr',v['hr'])
                
                if now-p.last_spo2_col>=p.ic_spo2:
                    raw_size=len(json.dumps(p.spo2_buf).encode())
                    comp=sdt.compress(p.spo2_buf,p.dc_spo2,p.t_sdt)
                    if comp:
                        post_sdt_size=len(json.dumps(comp).encode())
                        reduction=100*(1-post_sdt_size/raw_size) if raw_size>0 else 0
                        log.info(f"[COLETA SpO2] {p.id} {p.risk} | {len(p.spo2_buf)}→{len(comp)} pts | {raw_size}b→{post_sdt_size}b ({reduction:.1f}%)")
                        
                        if p.risk == 'ALTO':
                            original_signal_values = [point[1] for point in p.spo2_buf]
                            reconstructed_signal = reconstruct_signal(p.spo2_buf, comp)
                            prd_value = calculate_prd(original_signal_values, reconstructed_signal)
                            prd_accumulator['spo2'].append(prd_value)
                            log.info(f"  [DISTORÇÃO SpO2] {p.id} PRD={prd_value:.4f}%")

                        entry={'patient_id':p.id,'signal':'spo2','risco':p.risk,'data':comp, 'raw_size':raw_size, 'post_sdt_size':post_sdt_size}
                        forced=p.forced_vitals(); 
                        if forced: entry['forced_vitals']=forced
                        assembler.add(entry,p.risk)
                    p.spo2_buf=[]; p.last_spo2_col=now; p.backoff('spo2',v['spo2'])
            
            for b in assembler.get_ready_batches():
                log.info(f"[Empac] lote fila '{b['risk']}' pronto por {b['reason']}")
                feedback=send_batch(b)
                if feedback:
                    for pid,score in feedback.items():
                        for p in patients:
                            if p.id==pid: p.update_risk(score)
            
            queue_sizes = " | ".join([f"{r[:3]}:{q['size']}b" for r,q in assembler.queues.items()])
            print(f"\r[{time.strftime('%H:%M:%S')}] Monitorando... Filas: [ {queue_sizes} ]", end="")
            time.sleep(0.5)

    finally:
        log.info("="*50)
        log.info("Simulação encerrada. Estatísticas de PRD para Risco ALTO (PID-001):")
        if prd_accumulator['hr']:
            avg_hr = np.mean(prd_accumulator['hr'])
            min_hr = np.min(prd_accumulator['hr'])
            max_hr = np.max(prd_accumulator['hr'])
            log.info(f"  FC (hr): Média={avg_hr:.4f}% | Mín={min_hr:.4f}% | Máx={max_hr:.4f}%")
        if prd_accumulator['spo2']:
            avg_spo2 = np.mean(prd_accumulator['spo2'])
            min_spo2 = np.min(prd_accumulator['spo2'])
            max_spo2 = np.max(prd_accumulator['spo2'])
            log.info(f"  SpO2:    Média={avg_spo2:.4f}% | Mín={min_spo2:.4f}% | Máx={max_spo2:.4f}%")
        log.info("="*50)


if __name__=='__main__':
    main()