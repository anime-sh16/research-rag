#!/usr/bin/env bash
# Pulls /research-rag/* params from AWS SSM Parameter Store into /opt/research-rag/.env.
# Runs as root on EC2 via the research-rag-secrets.service systemd unit.

set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
PREFIX="/research-rag"
OUT="/opt/research-rag/.env"
TMP="$(mktemp)"

trap 'rm -f "$TMP"' EXIT

# get-parameters-by-path returns Name<TAB>Value pairs.
aws ssm get-parameters-by-path \
    --path "$PREFIX" \
    --with-decryption \
    --region "$REGION" \
    --query "Parameters[].[Name,Value]" \
    --output text \
| while IFS=$'\t' read -r name value; do
    [[ -z "$name" ]] && continue
    key=$(basename "$name" | tr '[:lower:]-' '[:upper:]_')
    # Escape value for shell: wrap in single quotes, escape embedded single quotes.
    escaped=${value//\'/\'\\\'\'}
    printf "%s='%s'\n" "$key" "$escaped"
done > "$TMP"

if [[ ! -s "$TMP" ]]; then
    echo "fetch-secrets: no parameters found under $PREFIX" >&2
    exit 1
fi

install -m 600 -o root -g root "$TMP" "$OUT"
echo "fetch-secrets: wrote $(wc -l < "$OUT") variables to $OUT"
