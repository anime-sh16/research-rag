#!/usr/bin/env bash
# Idempotent security group creation.

set -euo pipefail
source "$(dirname "$0")/env.sh"

# Find the default VPC for this region.
VPC_ID=$(aws ec2 describe-vpcs \
    --filters "Name=isDefault,Values=true" \
    --query 'Vpcs[0].VpcId' --output text)
[[ "$VPC_ID" != "None" && -n "$VPC_ID" ]] || die "No default VPC found in $AWS_REGION."

# Create or find SG.
SG_ID=$(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$VPC_ID" \
    --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo "None")

if [[ "$SG_ID" == "None" || -z "$SG_ID" ]]; then
    log "Creating security group $SG_NAME in $VPC_ID"
    SG_ID=$(aws ec2 create-security-group \
        --group-name "$SG_NAME" \
        --description "research-rag: ports 80/443 inbound, no SSH" \
        --vpc-id "$VPC_ID" \
        --query GroupId --output text)
else
    log "Security group $SG_NAME exists ($SG_ID) — ensuring rules"
fi

# Authorize ingress rules. Errors swallowed if rule already exists.
authorize() {
    local port="$1"
    aws ec2 authorize-security-group-ingress \
        --group-id "$SG_ID" \
        --protocol tcp \
        --port "$port" \
        --cidr 0.0.0.0/0 \
        2>&1 | grep -v "InvalidPermission.Duplicate" || true
}
authorize 80
authorize 443

cache_set SG_ID "$SG_ID"
cache_set VPC_ID "$VPC_ID"
log "Security group: $SG_ID (VPC $VPC_ID)"
