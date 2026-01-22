#!/bin/bash
# =============================================================================
# Download VISPAC Logs from All Instances
# Usage: ./download_logs.sh <scenario_name>
# Example: ./download_logs.sh scenario1_baseline
# =============================================================================

set -e

SCENARIO=${1:-"experiment"}
SSH_KEY="~/.ssh/vispac"
OUTPUT_DIR="logs/${SCENARIO}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Downloading logs for: $SCENARIO"
echo "Output directory: $OUTPUT_DIR"
echo "=============================================="

# Get IPs from Terraform output
cd terraform
CLOUD_IP=$(terraform output -raw cloud_public_ip 2>/dev/null || echo "")
FOG_IP=$(terraform output -raw fog_public_ip 2>/dev/null || echo "")

# Get edge IPs (from terraform output as JSON)
EDGE_IPS=$(terraform output -json edge_instances 2>/dev/null | jq -r 'to_entries[] | "\(.key):\(.value.public_ip)"' || echo "")
cd ..

echo ""
echo "Discovered instances:"
echo "  Cloud: $CLOUD_IP"
echo "  Fog: $FOG_IP"
echo "  Edges: $EDGE_IPS"
echo ""

# Function to download logs from an instance
download_from_instance() {
    local NAME=$1
    local IP=$2
    local SERVICE_NAME=$3
    
    if [ -z "$IP" ]; then
        echo "⚠️  Skipping $NAME - No IP found"
        return
    fi
    
    echo "📥 Downloading from $NAME ($IP)..."
    
    # Download archived logs (from scheduled collection)
    scp -i $SSH_KEY -o StrictHostKeyChecking=no \
        "ubuntu@${IP}:/home/ubuntu/vispac_logs_*.tar.gz" \
        "${OUTPUT_DIR}/" 2>/dev/null || echo "  - No archived logs found"
    
    # Download service logs
    scp -i $SSH_KEY -o StrictHostKeyChecking=no \
        "ubuntu@${IP}:/var/log/vispac-${SERVICE_NAME}.log" \
        "${OUTPUT_DIR}/${NAME}_service.log" 2>/dev/null || echo "  - No service log found"
    
    # Download error logs
    scp -i $SSH_KEY -o StrictHostKeyChecking=no \
        "ubuntu@${IP}:/var/log/vispac-${SERVICE_NAME}-error.log" \
        "${OUTPUT_DIR}/${NAME}_error.log" 2>/dev/null || echo "  - No error log found"
    
    # Download setup logs
    scp -i $SSH_KEY -o StrictHostKeyChecking=no \
        "ubuntu@${IP}:/var/log/vispac-${SERVICE_NAME}-setup.log" \
        "${OUTPUT_DIR}/${NAME}_setup.log" 2>/dev/null || echo "  - No setup log found"
    
    # Download app logs if they exist
    scp -i $SSH_KEY -o StrictHostKeyChecking=no -r \
        "ubuntu@${IP}:/home/vispac/app/logs/" \
        "${OUTPUT_DIR}/${NAME}_app_logs/" 2>/dev/null || echo "  - No app logs found"
    
    echo "  ✅ Done"
}

# Download from Cloud
download_from_instance "cloud" "$CLOUD_IP" "cloud"

# Download from Fog
download_from_instance "fog" "$FOG_IP" "fog"

# Download from all Edges
if [ -n "$EDGE_IPS" ]; then
    while IFS= read -r line; do
        EDGE_ID=$(echo "$line" | cut -d: -f1)
        EDGE_IP=$(echo "$line" | cut -d: -f2)
        download_from_instance "$EDGE_ID" "$EDGE_IP" "edge"
    done <<< "$EDGE_IPS"
fi

# Summary
echo ""
echo "=============================================="
echo "Download complete!"
echo "=============================================="
echo "Logs saved to: $OUTPUT_DIR"
echo ""
ls -la "$OUTPUT_DIR"
echo ""
echo "To analyze, run:"
echo "  python analyze_logs.py $OUTPUT_DIR --output results/"
