#!/bin/bash
# =============================================================================
# Fog Instance Setup Script
# MQTT Broker + NEWS2 API
# =============================================================================

set -e

# Variables from Terraform
CLOUD_URL="${cloud_url}"
GIT_REPO="${git_repo}"
GIT_BRANCH="${git_branch}"
EXPERIMENT_DURATION="${experiment_duration}"

LOG_FILE="/var/log/vispac-fog-setup.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "======================================"
echo "VISPAC Fog Setup"
echo "Cloud URL: $CLOUD_URL"
echo "======================================"

# Update system
apt-get update
apt-get install -y git python3 python3-pip python3-venv mosquitto mosquitto-clients netcat-openbsd

# Configure Mosquitto MQTT Broker
cat > /etc/mosquitto/conf.d/vispac.conf << 'EOF'
listener 1883 0.0.0.0
allow_anonymous true
max_queued_messages 10000
max_packet_size 1048576
EOF

systemctl enable mosquitto
systemctl restart mosquitto

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

# Create logs directory
sudo -u vispac mkdir -p /home/vispac/app/logs

# =============================================================================
# Create systemd service for Fog API
# =============================================================================

cat > /etc/systemd/system/vispac-fog.service << EOF
[Unit]
Description=VISPAC Fog Service - NEWS2 API
After=network.target mosquitto.service
Requires=mosquitto.service

[Service]
Type=simple
User=vispac
WorkingDirectory=/home/vispac/app
Environment="MQTT_BROKER=127.0.0.1"
Environment="MQTT_PORT=1883"
Environment="CLOUD_BASE_URL=$CLOUD_URL"
ExecStart=/home/vispac/app/venv/bin/python news2_api.py

# Restart policy
Restart=always
RestartSec=5

# Logging
StandardOutput=append:/var/log/vispac-fog.log
StandardError=append:/var/log/vispac-fog-error.log

[Install]
WantedBy=multi-user.target
EOF

# Create log files
touch /var/log/vispac-fog.log /var/log/vispac-fog-error.log
chown vispac:vispac /var/log/vispac-fog.log /var/log/vispac-fog-error.log

# Wait for Cloud to be ready
echo "Waiting for Cloud service to be ready..."
CLOUD_HOST=$(echo "$CLOUD_URL" | sed 's|http://||' | sed 's|:.*||')
CLOUD_PORT=$(echo "$CLOUD_URL" | sed 's|.*:||')

for i in {1..60}; do
    if nc -z "$CLOUD_HOST" "$CLOUD_PORT" 2>/dev/null; then
        echo "Cloud service is ready!"
        break
    fi
    echo "Waiting for Cloud service... ($i/60)"
    sleep 5
done

# Enable and start service
systemctl daemon-reload
systemctl enable vispac-fog
systemctl start vispac-fog

echo "======================================"
echo "Fog setup complete!"
echo "MQTT Broker: 0.0.0.0:1883"
echo "NEWS2 API: 0.0.0.0:8000"
echo "Cloud URL: $CLOUD_URL"
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
systemctl stop vispac-fog mosquitto || true

echo "Collecting logs..."
tar -czf "$LOG_ARCHIVE" \
    /var/log/vispac-*.log \
    /home/vispac/app/logs/ \
    /var/log/mosquitto/ \
    2>/dev/null || true

echo "Logs archived to: $LOG_ARCHIVE"
chown ubuntu:ubuntu "$LOG_ARCHIVE"

echo "Experiment completed at $(date)" > /home/ubuntu/experiment_complete.txt
COLLECT_SCRIPT

chmod +x /home/ubuntu/collect_logs.sh
chown ubuntu:ubuntu /home/ubuntu/collect_logs.sh

echo "/home/ubuntu/collect_logs.sh" | at now + $${EXPERIMENT_DURATION} hours

echo "Log collection scheduled in $${EXPERIMENT_DURATION} hours"
