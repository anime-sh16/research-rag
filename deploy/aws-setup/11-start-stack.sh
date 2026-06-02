#!/usr/bin/env bash
# Pulls the latest images from ECR and starts the research-rag stack on EC2.
# Idempotent — safe to re-run for redeploys.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]:-$0}")/env.sh"

[[ -n "${INSTANCE_ID:-}" ]] || die "INSTANCE_ID not in cache. Run 08-launch-ec2.sh first."
[[ -n "${ECR_REGISTRY:-}" ]] || die "ECR_REGISTRY not in cache. Run 02-create-ecr-repos.sh first."

IMAGE_TAG="${IMAGE_TAG:-latest}"

log "Starting stack on $INSTANCE_ID (tag: $IMAGE_TAG)"

CMD_ID=$(aws ssm send-command \
    --instance-ids "$INSTANCE_ID" \
    --document-name "AWS-RunShellScript" \
    --comment "research-rag start stack" \
    --parameters "commands=[
        \"set -euo pipefail\",
        \"cd /opt/research-rag/deploy\",
        \"AWS_REGION=$AWS_REGION aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_REGISTRY\",
        \"ECR_REGISTRY=$ECR_REGISTRY IMAGE_TAG=$IMAGE_TAG docker compose --env-file /opt/research-rag/.env pull\",
        \"ECR_REGISTRY=$ECR_REGISTRY IMAGE_TAG=$IMAGE_TAG docker compose --env-file /opt/research-rag/.env up -d --remove-orphans\",
        \"sleep 20\",
        \"docker compose --env-file /opt/research-rag/.env ps\",
        \"docker compose --env-file /opt/research-rag/.env logs --tail=30\"
    ]" \
    --query 'Command.CommandId' --output text)

log "Waiting for stack start (CommandId $CMD_ID) — up to 3 min..."
aws ssm wait command-executed \
    --command-id "$CMD_ID" \
    --instance-id "$INSTANCE_ID" \
    --cli-read-timeout 180

STATUS=$(aws ssm get-command-invocation \
    --command-id "$CMD_ID" --instance-id "$INSTANCE_ID" \
    --query 'Status' --output text)
OUTPUT=$(aws ssm get-command-invocation \
    --command-id "$CMD_ID" --instance-id "$INSTANCE_ID" \
    --query 'StandardOutputContent' --output text)
STDERR=$(aws ssm get-command-invocation \
    --command-id "$CMD_ID" --instance-id "$INSTANCE_ID" \
    --query 'StandardErrorContent' --output text)

log "Status: $STATUS"
echo "$OUTPUT"
[[ -n "$STDERR" ]] && warn "stderr: $STDERR"

[[ "$STATUS" == "Success" ]] || die "Stack start failed"

PUBLIC_IP="${PUBLIC_IP:-}"
[[ -z "$PUBLIC_IP" ]] && PUBLIC_IP=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

log "Stack is up."
log "API health : http://$PUBLIC_IP/health  (via Caddy → api:8000)"
log "UI         : http://$PUBLIC_IP/"
log "DuckDNS    : https://research-rag-animesh.duckdns.org/"
