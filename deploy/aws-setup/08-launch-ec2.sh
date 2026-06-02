#!/usr/bin/env bash
# Launches the t2.micro EC2 instance. Idempotent — re-runs detect an existing tagged instance.

set -euo pipefail
source "$(dirname "$0")/env.sh"

[[ -n "${SG_ID:-}" ]] || die "SG_ID not in cache. Run 07-create-security-group.sh first."

# Latest Amazon Linux 2023 AMI for x86_64.
AMI_ID=$(aws ssm get-parameter \
    --name "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64" \
    --query 'Parameter.Value' --output text)
log "Using AMI: $AMI_ID"

# Look up an existing instance by tag.
INSTANCE_ID=$(aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=$PROJECT_PREFIX" \
              "Name=instance-state-name,Values=running,pending,stopped" \
    --query 'Reservations[0].Instances[0].InstanceId' --output text 2>/dev/null || echo "None")

if [[ "$INSTANCE_ID" != "None" && -n "$INSTANCE_ID" ]]; then
    log "Instance $INSTANCE_ID already exists — skipping launch"
else
    log "Launching new t3.micro instance"
    INSTANCE_ID=$(aws ec2 run-instances \
        --image-id "$AMI_ID" \
        --instance-type t3.micro \
        --security-group-ids "$SG_ID" \
        --iam-instance-profile "Name=$IAM_INSTANCE_PROFILE" \
        --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=30,VolumeType=gp3,DeleteOnTermination=true}' \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$PROJECT_PREFIX}]" \
        --metadata-options 'HttpEndpoint=enabled,HttpTokens=required' \
        --count 1 \
        --query 'Instances[0].InstanceId' --output text)
fi

log "Instance ID: $INSTANCE_ID"
log "Waiting for instance to enter 'running' state (60-120s)..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"

log "Waiting for SSM agent to register (up to 3 min)..."
for i in {1..60}; do
    STATUS=$(aws ssm describe-instance-information \
        --filters "Key=InstanceIds,Values=$INSTANCE_ID" \
        --query 'InstanceInformationList[0].PingStatus' --output text 2>/dev/null || echo "")
    if [[ "$STATUS" == "Online" ]]; then
        log "SSM agent online"
        break
    fi
    sleep 5
done

PUBLIC_IP=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

cache_set INSTANCE_ID "$INSTANCE_ID"
cache_set PUBLIC_IP "$PUBLIC_IP"
log "Instance ready: $INSTANCE_ID at $PUBLIC_IP"
