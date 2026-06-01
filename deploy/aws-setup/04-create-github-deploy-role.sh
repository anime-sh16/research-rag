#!/usr/bin/env bash
# Idempotent IAM role + policy for the github-actions-deploy role.
# Requires GITHUB_REPO env var (e.g. "yourname/research-rag") on first run.

set -euo pipefail
source "$(dirname "$0")/env.sh"

if [[ -z "${GITHUB_REPO:-}" && -z "${GITHUB_REPO_CACHED:-}" ]]; then
    die "Set GITHUB_REPO=owner/repo (e.g. GITHUB_REPO=anime-sh16/research-rag) before running this script."
fi
GITHUB_REPO="${GITHUB_REPO:-$GITHUB_REPO_CACHED}"
cache_set GITHUB_REPO_CACHED "$GITHUB_REPO"

POLICY_DIR="$AWS_SETUP_DIR/policies"
TRUST=$(sed -e "s|__ACCOUNT_ID__|$AWS_ACCOUNT_ID|g" \
            -e "s|__GITHUB_REPO__|$GITHUB_REPO|g" \
            "$POLICY_DIR/github-deploy-trust.json")
PERMS=$(sed -e "s|__ACCOUNT_ID__|$AWS_ACCOUNT_ID|g" \
            -e "s|__AWS_REGION__|$AWS_REGION|g" \
            "$POLICY_DIR/github-deploy-permissions.json")

if aws iam get-role --role-name "$IAM_ROLE_DEPLOY" >/dev/null 2>&1; then
    log "Role $IAM_ROLE_DEPLOY already exists — updating trust policy"
    aws iam update-assume-role-policy \
        --role-name "$IAM_ROLE_DEPLOY" \
        --policy-document "$TRUST"
else
    log "Creating role $IAM_ROLE_DEPLOY"
    aws iam create-role \
        --role-name "$IAM_ROLE_DEPLOY" \
        --assume-role-policy-document "$TRUST" \
        --description "Assumed by GitHub Actions via OIDC; pushes to ECR + SSM Send-Command." \
        >/dev/null
fi

log "Putting inline permissions policy on $IAM_ROLE_DEPLOY"
aws iam put-role-policy \
    --role-name "$IAM_ROLE_DEPLOY" \
    --policy-name "github-actions-deploy-permissions" \
    --policy-document "$PERMS"

ROLE_ARN=$(aws iam get-role --role-name "$IAM_ROLE_DEPLOY" --query Role.Arn --output text)
cache_set IAM_ROLE_DEPLOY_ARN "$ROLE_ARN"
log "Role ARN: $ROLE_ARN"
