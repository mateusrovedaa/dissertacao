#!/bin/bash
# =============================================================================
# VISPAC Infrastructure Cleanup Script
# Destroys all AWS resources created by Terraform
# =============================================================================
#
# This script provides a safe way to remove all VISPAC infrastructure.
# It can be run in different modes for safety.
#
# Usage:
#   ./cleanup.sh              # Dry-run mode (shows what would be destroyed)
#   ./cleanup.sh --execute    # Actually destroy resources
#   ./cleanup.sh --force      # Destroy without confirmation prompt
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}"
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║           VISPAC Infrastructure Cleanup Script                   ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Parse arguments
DRY_RUN=true
FORCE=false

for arg in "$@"; do
    case $arg in
        --execute)
            DRY_RUN=false
            ;;
        --force)
            FORCE=true
            DRY_RUN=false
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  (no args)    Dry-run mode - shows what would be destroyed"
            echo "  --execute    Actually destroy resources (with confirmation)"
            echo "  --force      Destroy without confirmation prompt"
            echo "  --help       Show this help message"
            echo ""
            echo "Resources that will be destroyed:"
            echo "  - EC2 Instances (Edge x4, Fog x1, Cloud x1)"
            echo "  - Security Groups"
            echo "  - VPC, Subnets, Internet Gateway"
            echo "  - Route Tables"
            echo "  - SSH Key Pair"
            exit 0
            ;;
    esac
done

# Check if Terraform is installed
if ! command -v terraform &> /dev/null; then
    echo -e "${RED}Error: Terraform is not installed${NC}"
    echo "Please install Terraform first: https://www.terraform.io/downloads"
    exit 1
fi

# Check if Terraform state exists
if [ ! -f "terraform.tfstate" ] && [ ! -d ".terraform" ]; then
    echo -e "${YELLOW}Warning: No Terraform state found in this directory${NC}"
    echo "Either infrastructure was never created, or state is stored remotely."
    echo ""
    echo "If using remote state, ensure you have proper credentials configured."
    echo ""
fi

# Show current state
echo -e "${GREEN}Current Infrastructure State:${NC}"
echo "------------------------------"

if terraform state list 2>/dev/null | head -20; then
    RESOURCE_COUNT=$(terraform state list 2>/dev/null | wc -l)
    echo ""
    echo -e "Total resources: ${YELLOW}$RESOURCE_COUNT${NC}"
else
    echo "Unable to list resources. State may not exist."
    RESOURCE_COUNT=0
fi

echo ""

# Dry-run mode
if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}=== DRY-RUN MODE ===${NC}"
    echo "Showing what would be destroyed..."
    echo ""
    
    terraform plan -destroy -out=destroy.tfplan 2>/dev/null || {
        echo -e "${RED}Failed to generate destroy plan${NC}"
        echo "This may happen if:"
        echo "  - No terraform.tfvars file exists"
        echo "  - AWS credentials are not configured"
        echo "  - Infrastructure was never created"
        exit 1
    }
    
    echo ""
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}This was a DRY-RUN. No resources were destroyed.${NC}"
    echo ""
    echo "To actually destroy resources, run:"
    echo -e "  ${GREEN}$0 --execute${NC}"
    echo ""
    echo "To destroy without confirmation:"
    echo -e "  ${RED}$0 --force${NC}"
    
    # Clean up plan file
    rm -f destroy.tfplan
    exit 0
fi

# Confirmation prompt (unless --force)
if [ "$FORCE" = false ]; then
    echo -e "${RED}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║                        ⚠️  WARNING ⚠️                              ║${NC}"
    echo -e "${RED}║  This will PERMANENTLY DESTROY all VISPAC infrastructure!        ║${NC}"
    echo -e "${RED}║                                                                  ║${NC}"
    echo -e "${RED}║  Resources to be destroyed:                                      ║${NC}"
    echo -e "${RED}║    - 4x Edge EC2 instances                                       ║${NC}"
    echo -e "${RED}║    - 1x Fog EC2 instance                                         ║${NC}"
    echo -e "${RED}║    - 1x Cloud EC2 instance (with PostgreSQL data!)               ║${NC}"
    echo -e "${RED}║    - VPC, Subnets, Security Groups                               ║${NC}"
    echo -e "${RED}║    - All associated networking resources                         ║${NC}"
    echo -e "${RED}╚══════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -n "Type 'DESTROY' to confirm: "
    read -r confirmation
    
    if [ "$confirmation" != "DESTROY" ]; then
        echo -e "${GREEN}Aborted. No resources were destroyed.${NC}"
        exit 0
    fi
fi

# Execute destroy
echo ""
echo -e "${RED}Destroying infrastructure...${NC}"
echo ""

terraform destroy -auto-approve

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              Infrastructure destroyed successfully!              ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Cleanup local files
echo "Cleaning up local files..."
rm -f destroy.tfplan
rm -f terraform.tfstate.backup

echo ""
echo "Remaining files:"
echo "  - terraform.tfstate (empty, can be deleted)"
echo "  - .terraform/ (provider cache, can be deleted with: rm -rf .terraform)"
echo "  - terraform.tfvars (your configuration, keep for future use)"
echo ""
echo -e "${GREEN}Cleanup complete!${NC}"
