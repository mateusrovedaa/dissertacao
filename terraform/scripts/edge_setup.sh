#!/bin/bash
# =============================================================================
# Edge Instance Setup Script
# Simulates constrained IoT device with resource limits
# =============================================================================

set -e

# Variables from Terraform
EDGE_ID="${edge_id}"
HIGH_PATIENTS="${high_patients}"
LOW_PATIENTS="${low_patients}"
SPECIFIC_PATIENTS="${specific_patients}"
LOW_SPECIFIC_PATIENTS="${low_specific_patients}"
MQTT_BROKER="${mqtt_broker}"
GIT_REPO="${git_repo}"
GIT_BRANCH="${git_branch}"
MEMORY_LIMIT_MB="${memory_limit}"
CPU_LIMIT_PERCENT="${cpu_limit}"
SCENARIO="${scenario}"
EXPERIMENT_DURATION="${experiment_duration}"

LOG_FILE="/var/log/vispac-edge-setup.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "======================================"
echo "VISPAC Edge Setup - $EDGE_ID"
echo "High Risk Patients: $HIGH_PATIENTS"
echo "Low Risk Patients: $LOW_PATIENTS"
echo "======================================"

# Update system
apt-get update
apt-get install -y git python3 python3-pip python3-venv cgroup-tools netcat-openbsd

# Create app user
useradd -m -s /bin/bash vispac || true

# Clone repository
cd /home/vispac
sudo -u vispac git clone -b "$GIT_BRANCH" "$GIT_REPO" app || {
    cd /home/vispac/app
    sudo -u vispac git pull origin "$GIT_BRANCH"
}
cd /home/vispac/app

# Create virtual environment
sudo -u vispac python3 -m venv venv
sudo -u vispac ./venv/bin/pip install --upgrade pip
sudo -u vispac ./venv/bin/pip install -r requirements.txt

# Download datasets based on patient configuration
if [ "$HIGH_PATIENTS" -gt 0 ]; then
    echo "Downloading high_risk (BIDMC) dataset..."
    sudo -u vispac ./venv/bin/python download_bidmc_data.py || echo "BIDMC dataset download failed, may already exist"
fi

# Note: low_risk dataset needs to be pre-loaded or downloaded separately
# The low_risk dataset uses processed Kaggle data

# =============================================================================
# Resource Limits Setup (using cgroups v2)
# =============================================================================

echo "Setting up resource limits..."

# Create cgroup for edge process
CGROUP_PATH="/sys/fs/cgroup/vispac-edge"

# Check if cgroups v2 is available
if [ -f /sys/fs/cgroup/cgroup.controllers ]; then
    echo "Using cgroups v2"
    
    # Enable controllers
    echo "+memory +cpu" > /sys/fs/cgroup/cgroup.subtree_control 2>/dev/null || true
    
    mkdir -p "$CGROUP_PATH"
    
    # Memory limit
    MEMORY_LIMIT_BYTES=$((MEMORY_LIMIT_MB * 1024 * 1024))
    echo "$MEMORY_LIMIT_BYTES" > "$CGROUP_PATH/memory.max"
    echo "$MEMORY_LIMIT_BYTES" > "$CGROUP_PATH/memory.high"
    
    # CPU limit (cpu.max format: quota period)
    # For 50% of 1 CPU: quota=50000, period=100000
    CPU_QUOTA=$((CPU_LIMIT_PERCENT * 1000))
    echo "$CPU_QUOTA 100000" > "$CGROUP_PATH/cpu.max"
    
    echo "Resource limits set: Memory=$${MEMORY_LIMIT_MB}MB, CPU=$${CPU_LIMIT_PERCENT}%"
else
    echo "Using cgroups v1"
    
    # Memory limit
    mkdir -p /sys/fs/cgroup/memory/vispac-edge
    MEMORY_LIMIT_BYTES=$((MEMORY_LIMIT_MB * 1024 * 1024))
    echo "$MEMORY_LIMIT_BYTES" > /sys/fs/cgroup/memory/vispac-edge/memory.limit_in_bytes
    
    # CPU limit
    mkdir -p /sys/fs/cgroup/cpu/vispac-edge
    CPU_QUOTA=$((CPU_LIMIT_PERCENT * 1000))
    echo "$CPU_QUOTA" > /sys/fs/cgroup/cpu/vispac-edge/cpu.cfs_quota_us
    echo "100000" > /sys/fs/cgroup/cpu/vispac-edge/cpu.cfs_period_us
fi

# =============================================================================
# Create systemd service with resource limits
# =============================================================================

cat > /etc/systemd/system/vispac-edge.service << EOF
[Unit]
Description=VISPAC Edge Service - $EDGE_ID
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=vispac
WorkingDirectory=/home/vispac/app
Environment="EDGE_ID=$EDGE_ID"
Environment="HIGH_PATIENTS=$HIGH_PATIENTS"
Environment="LOW_PATIENTS=$LOW_PATIENTS"
Environment="SPECIFIC_PATIENTS=$SPECIFIC_PATIENTS"
Environment="LOW_SPECIFIC_PATIENTS=$LOW_SPECIFIC_PATIENTS"
Environment="MQTT_BROKER=$MQTT_BROKER"
Environment="MQTT_PORT=1883"
Environment="EDGE_USE_MQTT=1"
Environment="SCENARIO=$SCENARIO"
ExecStart=/home/vispac/app/venv/bin/python vispac_edge_prototype.py

# Resource Limits (alternative to cgroups, systemd native)
MemoryMax=$${MEMORY_LIMIT_MB}M
MemoryHigh=$((MEMORY_LIMIT_MB * 90 / 100))M
CPUQuota=$${CPU_LIMIT_PERCENT}%

# Restart policy
Restart=always
RestartSec=10

# Logging
StandardOutput=append:/var/log/vispac-edge.log
StandardError=append:/var/log/vispac-edge-error.log

[Install]
WantedBy=multi-user.target
EOF

# Create log files
touch /var/log/vispac-edge.log /var/log/vispac-edge-error.log
chown vispac:vispac /var/log/vispac-edge.log /var/log/vispac-edge-error.log

# Enable and start service
systemctl daemon-reload
systemctl enable vispac-edge

# Wait for fog to be ready before starting
echo "Waiting for Fog service to be ready..."
for i in {1..60}; do
    if nc -z "$MQTT_BROKER" 1883 2>/dev/null; then
        echo "Fog MQTT broker is ready!"
        break
    fi
    echo "Waiting for MQTT broker... ($i/60)"
    sleep 5
done

systemctl start vispac-edge

echo "======================================"
echo "Edge setup complete!"
echo "Edge ID: $EDGE_ID"
echo "High Patients: $HIGH_PATIENTS"
echo "Low Patients: $LOW_PATIENTS"
echo "MQTT Broker: $MQTT_BROKER:1883"
echo "Memory Limit: $${MEMORY_LIMIT_MB}MB"
echo "CPU Limit: $${CPU_LIMIT_PERCENT}%"
echo "Experiment Duration: $${EXPERIMENT_DURATION}h"
echo "======================================"

# =============================================================================
# Schedule Experiment End and Log Collection
# =============================================================================

apt-get install -y at
systemctl enable atd
systemctl start atd

cat > /home/ubuntu/collect_logs.sh << 'COLLECT_SCRIPT'
#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
HOSTNAME=$(hostname)
LOG_ARCHIVE="/home/ubuntu/vispac_logs_$${HOSTNAME}_$${TIMESTAMP}.tar.gz"

echo "Stopping VISPAC services..."
systemctl stop vispac-edge || true

echo "Collecting logs..."
tar -czf "$LOG_ARCHIVE" \
    /var/log/vispac-*.log \
    /home/vispac/app/logs/ \
    2>/dev/null || true

echo "Logs archived to: $LOG_ARCHIVE"
chown ubuntu:ubuntu "$LOG_ARCHIVE"

echo "Experiment completed at $(date)" > /home/ubuntu/experiment_complete.txt
COLLECT_SCRIPT

chmod +x /home/ubuntu/collect_logs.sh
chown ubuntu:ubuntu /home/ubuntu/collect_logs.sh

# Schedule the log collection
echo "/home/ubuntu/collect_logs.sh" | at now + $${EXPERIMENT_DURATION} hours

echo "Log collection scheduled in $${EXPERIMENT_DURATION} hours"
