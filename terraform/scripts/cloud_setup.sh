#!/bin/bash
# =============================================================================
# Cloud Instance Setup Script
# Cloud API + PostgreSQL Database
# =============================================================================

set -e

# Variables from Terraform
DB_PASSWORD="${db_password}"
GIT_REPO="${git_repo}"
GIT_BRANCH="${git_branch}"
EXPERIMENT_DURATION="${experiment_duration}"

LOG_FILE="/var/log/vispac-cloud-setup.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "======================================"
echo "VISPAC Cloud Setup"
echo "======================================"

# Update system
apt-get update
apt-get install -y git python3 python3-pip python3-venv postgresql postgresql-contrib

# =============================================================================
# PostgreSQL Setup
# =============================================================================

echo "Configuring PostgreSQL..."

# Start PostgreSQL
systemctl enable postgresql
systemctl start postgresql

# Create database and user
sudo -u postgres psql << EOF
CREATE USER vispac WITH PASSWORD '$DB_PASSWORD';
CREATE DATABASE vispac_db OWNER vispac;
GRANT ALL PRIVILEGES ON DATABASE vispac_db TO vispac;
EOF

# Configure PostgreSQL to accept local connections
echo "host    vispac_db    vispac    127.0.0.1/32    md5" >> /etc/postgresql/*/main/pg_hba.conf
systemctl restart postgresql

# =============================================================================
# Application Setup
# =============================================================================

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
# Create systemd service for Cloud API
# =============================================================================

cat > /etc/systemd/system/vispac-cloud.service << EOF
[Unit]
Description=VISPAC Cloud Service - Data Storage API
After=network.target postgresql.service
Requires=postgresql.service

[Service]
Type=simple
User=vispac
WorkingDirectory=/home/vispac/app
Environment="CLOUD_DB_URL=postgresql://vispac:$DB_PASSWORD@127.0.0.1:5432/vispac_db"
ExecStart=/home/vispac/app/venv/bin/python cloud_api.py

# Restart policy
Restart=always
RestartSec=5

# Logging
StandardOutput=append:/var/log/vispac-cloud.log
StandardError=append:/var/log/vispac-cloud-error.log

[Install]
WantedBy=multi-user.target
EOF

# Create log files
touch /var/log/vispac-cloud.log /var/log/vispac-cloud-error.log
chown vispac:vispac /var/log/vispac-cloud.log /var/log/vispac-cloud-error.log

# Enable and start service
systemctl daemon-reload
systemctl enable vispac-cloud
systemctl start vispac-cloud

echo "======================================"
echo "Cloud setup complete!"
echo "Cloud API: 0.0.0.0:9000"
echo "PostgreSQL: 127.0.0.1:5432"
echo "Database: vispac_db"
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
systemctl stop vispac-cloud postgresql || true

echo "Collecting logs..."
tar -czf "$LOG_ARCHIVE" \
    /var/log/vispac-*.log \
    /home/vispac/app/logs/ \
    /var/log/postgresql/ \
    2>/dev/null || true

echo "Logs archived to: $LOG_ARCHIVE"
chown ubuntu:ubuntu "$LOG_ARCHIVE"

echo "Experiment completed at $(date)" > /home/ubuntu/experiment_complete.txt
COLLECT_SCRIPT

chmod +x /home/ubuntu/collect_logs.sh
chown ubuntu:ubuntu /home/ubuntu/collect_logs.sh

# Cloud stops 1 hour after edges to ensure all data is stored
CLOUD_DURATION=$(($${EXPERIMENT_DURATION} + 1))
echo "/home/ubuntu/collect_logs.sh" | at now + $${CLOUD_DURATION} hours

echo "Log collection scheduled in $${CLOUD_DURATION}h (1h after edges)"
