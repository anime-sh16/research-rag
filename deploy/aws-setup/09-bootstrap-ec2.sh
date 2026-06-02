#!/usr/bin/env bash
# Bootstraps the running EC2 instance: installs Docker, ships deploy/ artifacts,
# installs systemd units, runs fetch-secrets once.

set -euo pipefail
source "$(dirname "$0")/env.sh"

[[ -n "${INSTANCE_ID:-}" ]] || die "INSTANCE_ID not in cache. Run 08-launch-ec2.sh first."

REPO_ROOT="$(cd "$AWS_SETUP_DIR/../.." && pwd)"
DEPLOY_DIR="$REPO_ROOT/deploy"

# ── Step A: install Docker ────────────────────────────────────────────────────
log "Installing Docker on $INSTANCE_ID"
CMD_ID=$(aws ssm send-command \
    --instance-ids "$INSTANCE_ID" \
    --document-name "AWS-RunShellScript" \
    --comment "research-rag bootstrap: install docker" \
    --parameters 'commands=[
        "set -euo pipefail",
        "dnf update -y",
        "dnf install -y docker",
        "mkdir -p /usr/local/lib/docker/cli-plugins",
        "curl -SL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 -o /usr/local/lib/docker/cli-plugins/docker-compose",
        "chmod +x /usr/local/lib/docker/cli-plugins/docker-compose",
        "systemctl enable --now docker",
        "usermod -aG docker ssm-user",
        "mkdir -p /opt/research-rag/deploy/systemd",
        "chown -R ssm-user:ssm-user /opt/research-rag",
        "docker --version"
    ]' \
    --query 'Command.CommandId' --output text)

log "Waiting for Docker install (CommandId $CMD_ID)..."
aws ssm wait command-executed --command-id "$CMD_ID" --instance-id "$INSTANCE_ID"
STATUS=$(aws ssm get-command-invocation \
    --command-id "$CMD_ID" --instance-id "$INSTANCE_ID" \
    --query 'Status' --output text)
OUTPUT=$(aws ssm get-command-invocation \
    --command-id "$CMD_ID" --instance-id "$INSTANCE_ID" \
    --query 'StandardOutputContent' --output text)
log "Docker install status: $STATUS"
echo "$OUTPUT" | tail -5
[[ "$STATUS" == "Success" ]] || die "Docker install failed"

# ── Step B: upload deploy/ artifacts via SSM ─────────────────────────────────
log "Uploading deploy/ artifacts"

upload_file() {
    local local_path="$1" remote_path="$2" mode="${3:-644}"
    local b64
    b64=$(base64 -i "$local_path")
    local cmd_id
    cmd_id=$(aws ssm send-command \
        --instance-ids "$INSTANCE_ID" \
        --document-name "AWS-RunShellScript" \
        --comment "upload $(basename "$remote_path")" \
        --parameters "commands=[
            \"mkdir -p \$(dirname $remote_path)\",
            \"echo '$b64' | base64 -d > $remote_path\",
            \"chmod $mode $remote_path\"
        ]" \
        --query 'Command.CommandId' --output text)
    aws ssm wait command-executed --command-id "$cmd_id" --instance-id "$INSTANCE_ID"
    local status
    status=$(aws ssm get-command-invocation \
        --command-id "$cmd_id" --instance-id "$INSTANCE_ID" \
        --query 'Status' --output text)
    [[ "$status" == "Success" ]] || die "Upload of $remote_path failed: $status"
    log "Uploaded $(basename "$remote_path")"
}

upload_file "$DEPLOY_DIR/docker-compose.yml"           /opt/research-rag/deploy/docker-compose.yml           644
upload_file "$DEPLOY_DIR/Caddyfile"                    /opt/research-rag/deploy/Caddyfile                    644
upload_file "$DEPLOY_DIR/fetch-secrets.sh"             /opt/research-rag/deploy/fetch-secrets.sh             755
upload_file "$DEPLOY_DIR/duckdns-update.sh"            /opt/research-rag/deploy/duckdns-update.sh            755
upload_file "$DEPLOY_DIR/systemd/research-rag.service"         /etc/systemd/system/research-rag.service         644
upload_file "$DEPLOY_DIR/systemd/research-rag-secrets.service" /etc/systemd/system/research-rag-secrets.service 644
upload_file "$DEPLOY_DIR/systemd/duckdns-update.service"       /etc/systemd/system/duckdns-update.service       644
upload_file "$DEPLOY_DIR/systemd/duckdns-update.timer"         /etc/systemd/system/duckdns-update.timer         644

# ── Step C: enable systemd units + first secrets fetch + DuckDNS update ──────
log "Enabling systemd units and running first secrets fetch"
CMD_ID=$(aws ssm send-command \
    --instance-ids "$INSTANCE_ID" \
    --document-name "AWS-RunShellScript" \
    --parameters "commands=[
        \"systemctl daemon-reload\",
        \"systemctl enable research-rag-secrets.service\",
        \"systemctl enable duckdns-update.timer\",
        \"systemctl enable research-rag.service\",
        \"AWS_REGION=$AWS_REGION /opt/research-rag/deploy/fetch-secrets.sh\",
        \"ls -la /opt/research-rag/.env\",
        \"systemctl start duckdns-update.service\",
        \"systemctl status duckdns-update.service --no-pager\"
    ]" \
    --query 'Command.CommandId' --output text)

aws ssm wait command-executed --command-id "$CMD_ID" --instance-id "$INSTANCE_ID"
STATUS=$(aws ssm get-command-invocation \
    --command-id "$CMD_ID" --instance-id "$INSTANCE_ID" \
    --query 'Status' --output text)
OUTPUT=$(aws ssm get-command-invocation \
    --command-id "$CMD_ID" --instance-id "$INSTANCE_ID" \
    --query 'StandardOutputContent' --output text)
log "Bootstrap status: $STATUS"
echo "$OUTPUT"
[[ "$STATUS" == "Success" ]] || die "Bootstrap step C failed"

log "EC2 box bootstrapped successfully"
