# VISPAC AWS Terraform

Este diretório contém a infraestrutura como código (IaC) para deploy do VISPAC na AWS.

## Arquitetura

```
┌────────────────────────────────────────────────────────────────┐
│                         VPC (10.0.0.0/16)                      │
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │          Subnet Privada (10.0.1.0/24) - Edge + Fog      │   │
│  │                                                         │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐        │   │
│  │  │ Edge-01 │ │ Edge-02 │ │ Edge-03 │ │ Edge-04 │        │   │
│  │  │t3.micro │ │t3.micro │ │t3.micro │ │t3.micro │        │   │
│  │  │ 1-13    │ │ 14-27   │ │ 28-40   │ │ 41-53   │        │   │
│  │  │ 256MB   │ │ 256MB   │ │ 256MB   │ │ 256MB   │        │   │
│  │  │ 50%CPU  │ │ 50%CPU  │ │ 50%CPU  │ │ 50%CPU  │        │   │
│  │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘        │   │
│  │       └───────────┴─────┬─────┴───────────┘             │   │
│  │                         │ MQTT (1883)                   │   │
│  │                   ┌─────▼─────┐                         │   │
│  │                   │    FOG    │                         │   │
│  │                   │ t3.small  │                         │   │
│  │                   │ Mosquitto │                         │   │
│  │                   │ NEWS2 API │                         │   │
│  │                   └─────┬─────┘                         │   │
│  └─────────────────────────│───────────────────────────────┘   │
│                            │                                   │
│  ┌─────────────────────────│───────────────────────────────┐   │
│  │        Subnet Pública (10.0.2.0/24) - Cloud             │   │
│  │                   ┌─────▼─────┐                         │   │
│  │                   │   CLOUD   │                         │   │
│  │                   │ t3.small  │                         │   │
│  │                   │Cloud API  │                         │   │
│  │                   │PostgreSQL │                         │   │
│  │                   └───────────┘                         │   │
│  └─────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────┘
```

## Pré-requisitos

1. **AWS CLI** configurado com credenciais
2. **Terraform** >= 1.0 instalado
3. **SSH Key** para acesso às instâncias

## Quick Start

```bash
# 1. Gerar chave SSH
ssh-keygen -t rsa -b 4096 -f ~/.ssh/vispac -N ""

# 2. Copiar e editar variáveis
cp terraform.tfvars.example terraform.tfvars
# Edite terraform.tfvars com seus valores

# 3. Inicializar Terraform
terraform init

# 4. Ver plano de execução
terraform plan

# 5. Aplicar infraestrutura
terraform apply

# 6. Ver outputs
terraform output
```

## Variáveis Importantes

| Variável | Descrição | Default |
|----------|-----------|---------|
| `edge_count` | Número de instâncias Edge | 4 |
| `edge_memory_limit_mb` | Limite de memória por Edge | 256MB |
| `edge_cpu_limit_percent` | Limite de CPU por Edge | 50% |
| `total_patients` | Total de pacientes no dataset | 53 |

## Distribuição de Pacientes

Com 4 edges e 53 pacientes:
- **Edge-01**: Pacientes 1-13
- **Edge-02**: Pacientes 14-27
- **Edge-03**: Pacientes 28-40
- **Edge-04**: Pacientes 41-53

## Monitoramento

```bash
# SSH para uma instância
ssh -i ~/.ssh/vispac ubuntu@<IP>

# Ver logs do serviço
sudo journalctl -u vispac-edge -f
sudo journalctl -u vispac-fog -f
sudo journalctl -u vispac-cloud -f

# Ver logs da aplicação
tail -f /var/log/vispac-edge.log
tail -f /home/vispac/app/logs/edge_log.txt
```

## Custos Estimados (us-east-1)

| Recurso | Tipo | Qtd | Custo/hora | Custo/mês |
|---------|------|-----|------------|-----------|
| Edge | t3.micro | 4 | $0.0104 x 4 | ~$30 |
| Fog | t3.small | 1 | $0.0208 | ~$15 |
| Cloud | t3.small | 1 | $0.0208 | ~$15 |
| **Total** | | | | **~$60/mês** |

## Destruir Infraestrutura

```bash
terraform destroy
```

## Troubleshooting

### Edge não conecta ao Fog
```bash
# No Edge, verificar conectividade MQTT
nc -z <FOG_IP> 1883

# Ver logs
sudo journalctl -u vispac-edge -f
```

### Fog não conecta ao Cloud
```bash
# No Fog, verificar conectividade HTTP
curl http://<CLOUD_IP>:9000/health

# Ver logs
sudo journalctl -u vispac-fog -f
```
