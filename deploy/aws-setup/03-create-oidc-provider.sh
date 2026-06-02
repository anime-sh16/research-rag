#!/usr/bin/env bash
# Idempotent OIDC provider creation for GitHub Actions.

set -euo pipefail
source "$(dirname "$0")/env.sh"

OIDC_URL="https://token.actions.githubusercontent.com"
OIDC_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:oidc-provider/token.actions.githubusercontent.com"

if aws iam get-open-id-connect-provider --open-id-connect-provider-arn "$OIDC_ARN" >/dev/null 2>&1; then
    log "OIDC provider already exists — skipping"
else
    log "Creating OIDC provider for GitHub Actions"
    aws iam create-open-id-connect-provider \
        --url "$OIDC_URL" \
        --client-id-list "sts.amazonaws.com" \
        --thumbprint-list "6938fd4d98bab03faadb97b34396831e3780aea1" \
        >/dev/null
fi

cache_set OIDC_PROVIDER_ARN "$OIDC_ARN"
log "OIDC provider ARN: $OIDC_ARN"
