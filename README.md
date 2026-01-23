# ViSPAC - Vital Sign Prioritization and Adaptive Compression

Repositório do protótipo Edge-Fog-Cloud para compressão adaptativa e priorização inteligente no monitoramento de sinais vitais de pacientes.

## Visão Geral

O ViSPAC é um modelo que combina compressão de dados (SDT + Huffman/LZW) com priorização clínica baseada no escore NEWS2 em uma arquitetura distribuída de três camadas:

- **Edge**: Coleta sinais vitais, aplica compressão adaptativa e transmite para a Fog
- **Fog**: Calcula o escore NEWS2, envia callbacks de risco para a Edge e encaminha dados para a Cloud
- **Cloud**: Armazena dados e disponibiliza dashboards de visualização

## Componentes

| Componente | Arquivo | Descrição |
|------------|---------|-----------|
| Edge | `vispac_edge_prototype.py` | Simulador que coleta amostras, aplica SDT e compressão lossless, implementa back-off exponencial e vigilância contínua |
| Fog | `news2_api.py` | API FastAPI (porta 8000) que descomprime batches, calcula NEWS2, retorna scores à Edge e encaminha streams para Cloud |
| Cloud | `cloud_api.py` | API FastAPI (porta 9000) que recebe streams por nível de risco e persiste em PostgreSQL/SQLite |
| MQTT | Eclipse Mosquitto | Broker MQTT para comunicação Edge-Fog |
| Análise | `analyze_logs.py` | Script para análise de logs e geração de dashboard HTML interativo |

## Estrutura do Projeto

```
dissertacao/
├── vispac_edge_prototype.py  # Simulador Edge com 3 cenários
├── news2_api.py              # API Fog (NEWS2 + callbacks)
├── cloud_api.py              # API Cloud (armazenamento)
├── compressors.py            # Algoritmos SDT, Huffman, LZW
├── analyze_logs.py           # Análise de logs + Dashboard HTML
├── terraform/                # Infraestrutura AWS (20 Edges + Fog + Cloud)
├── results/                  # Dashboards e CSVs gerados
├── logs/                     # Logs dos experimentos
└── datasets/                 # Dados de pacientes (BIDMC, Student Health)
```

## Datasets

O projeto suporta dois tipos de datasets para simulação:

- **Low Risk**: Dados de estudantes saudáveis do Kaggle (~600 amostras, 7 pacientes virtuais)
- **High Risk**: Pacientes de UTI do PhysioNet BIDMC (~25.000 amostras, 53 pacientes reais)

Para detalhes completos, veja [DATASETS.md](DATASETS.md).

## Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Execução Local

### Modo HTTP

```bash
# Iniciar Cloud (porta 9000)
python cloud_api.py &

# Iniciar Fog (porta 8000)
python news2_api.py &

# Executar simulador Edge
python vispac_edge_prototype.py
```

### Modo MQTT

```bash
# Instalar broker MQTT (ex: mosquitto)
# sudo apt install mosquitto && mosquitto -v

# Configurar variáveis de ambiente
export EDGE_USE_MQTT=1
export MQTT_BROKER=127.0.0.1
export MQTT_PORT=1883

# Iniciar serviços
python cloud_api.py &
python news2_api.py &
python vispac_edge_prototype.py
```

### Docker Compose

```bash
# Build e start (PostgreSQL + MQTT + serviços)
docker compose up --build

# Logs
docker compose logs -f

# Stop
docker compose down -v
```

## Cenários de Experimento

O simulador suporta três cenários configuráveis via variável `SCENARIO`:

| Cenário | Nome | Descrição |
|---------|------|-----------|
| 1 | Baseline | Transmissão contínua a 1 Hz, sem compressão |
| 2 | Static | Compressão SDT + Huffman/LZW com parâmetros fixos |
| 3 | ViSPAC | Compressão adaptativa + priorização NEWS2 + back-off + callbacks |

```bash
# Executar cenário específico
export SCENARIO=3
python vispac_edge_prototype.py
```

## Infraestrutura AWS (Terraform)

O diretório `terraform/` contém a infraestrutura para deploy em AWS:

- **20 instâncias Edge** (t2.small, us-east-1) com limites de CPU/memória (permitindo criar mais instâncias)
- **1 instância Fog** (t3.small, us-east-1)
- **1 instância Cloud** (t3.small, us-west-1) com PostgreSQL
- **VPC Peering** entre regiões para simular latência realista

### Deploy

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars
# Editar terraform.tfvars com suas configurações

terraform init
terraform plan
terraform apply
```

Para detalhes completos, veja [terraform/README.md](terraform/README.md).

## Análise de Resultados

O script `analyze_logs.py` processa os logs dos experimentos e gera:

- **CSVs** com métricas por Edge
- **Dashboard HTML** interativo com gráficos comparativos
- **Relatório Markdown** com resumo dos resultados

### Uso

```bash
# Analisar um cenário
python analyze_logs.py logs/scenario3_vispac --output results/

# Comparar todos os cenários
python analyze_logs.py --compare logs/ --output results/
```

### Dashboard Interativo

Após a análise, o dashboard é gerado em `results/dashboard.html` e pode ser visualizado em:
**https://roveda.dev/dissertacao/results/dashboard.html**

O dashboard inclui:
- Tabela comparativa dos três cenários
- Gráficos de compressão, latência e PRD
- Evolução do intervalo de coleta por paciente (back-off/reset)
- Métricas Fog e Cloud

## Métricas Coletadas

| Métrica | Descrição |
|---------|-----------|
| Transmissões | Número de pacotes MQTT enviados |
| Taxa de Compressão (%) | `(1 - comprimido/bruto) × 100` |
| PRD (%) | Distorção do sinal reconstruído vs original |
| Latência (ms) | Tempo Edge → Fog (coleta a confirmação) |
| PRD por Risco | Distorção segmentada por nível NEWS2 |

## Configuração

### Variáveis de Ambiente

| Variável | Descrição | Default |
|----------|-----------|---------|
| `SCENARIO` | Cenário de execução (1, 2 ou 3) | 3 |
| `CLOUD_BASE_URL` | URL da API Cloud | `http://127.0.0.1:9000` |
| `CLOUD_DB_URL` | String de conexão PostgreSQL | - |
| `CLOUD_DB_PATH` | Caminho SQLite (fallback) | `cloud_data.sqlite3` |
| `EDGE_USE_MQTT` | Habilitar MQTT | 0 |
| `MQTT_BROKER` | Endereço do broker | `127.0.0.1` |
| `MQTT_PORT` | Porta do broker | `1883` |
| `DATASET_TYPE` | Tipo de dataset | `high_risk` |

## Modelo de Dados (PostgreSQL)

```sql
CREATE TABLE IF NOT EXISTS events (
  id SERIAL PRIMARY KEY,
  patient_id TEXT,
  risk TEXT,
  score INTEGER,
  signal TEXT,
  data_json TEXT,
  hr REAL,
  spo2 REAL,
  rr REAL,
  temp REAL,
  sys_bp REAL,
  on_o2 BOOLEAN,
  spo2_scale INTEGER,
  consciousness TEXT,
  received_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Documentação Adicional

- **[DATASETS.md](DATASETS.md)** - Guia completo de datasets
- **[terraform/README.md](terraform/README.md)** - Documentação da infraestrutura AWS

## Resultados Principais

Os experimentos de 12 horas com 20 dispositivos Edge demonstraram:

| Métrica | Baseline | Static | ViSPAC |
|---------|----------|--------|--------|
| Transmissões | 6.726.598 | 567.108 | **120.919** |
| Compressão (%) | 0,0 | 74,0 | **80,7** |
| PRD (%) | 0,0 | 3,89 | **1,30** |
| Latência (ms) | 1.380 | 1.271 | **1.199** |

O ViSPAC alcançou:
- **98,2%** de redução nas transmissões vs Baseline
- **80,7%** de taxa de compressão média
- **1,30%** de distorção PRD (menor que Static apesar de maior compressão)
- **98,2%** de redução no armazenamento Cloud

## Licença

Este projeto é parte da dissertação de mestrado de Mateus Roveda, sob orientação do Prof. Dr. Rodrigo da Rosa Righi, no Programa de Pós-Graduação em Computação Aplicada da UNISINOS.
