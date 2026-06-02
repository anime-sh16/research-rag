#!/usr/bin/env bash
# Sourced by every other script in this directory.
# Re-runs safe: cached values are read, missing ones are discovered.

set -euo pipefail

export AWS_REGION="${AWS_REGION:-ap-northeast-1}"
export AWS_PROFILE="${AWS_PROFILE:-research-rag}"

# Project naming
export PROJECT_PREFIX="research-rag"
export ECR_REPO_API="${PROJECT_PREFIX}-api"
export ECR_REPO_UI="${PROJECT_PREFIX}-ui"
export IAM_ROLE_DEPLOY="github-actions-deploy"
export IAM_ROLE_INSTANCE="ec2-rag-instance-role"
export IAM_INSTANCE_PROFILE="ec2-rag-instance-profile"
export SG_NAME="rag-sg"
export SSM_PREFIX="/research-rag"

# Cache file
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export AWS_SETUP_DIR="$THIS_DIR"
export AWS_SETUP_CACHE="$THIS_DIR/.cache/state.env"
mkdir -p "$(dirname "$AWS_SETUP_CACHE")"
touch "$AWS_SETUP_CACHE"

# Read previously-discovered values, if any.
# shellcheck disable=SC1090
source "$AWS_SETUP_CACHE"

# Account ID + region — discover once, then persist.
if [[ -z "${AWS_ACCOUNT_ID:-}" ]]; then
    AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    echo "AWS_ACCOUNT_ID=$AWS_ACCOUNT_ID" >> "$AWS_SETUP_CACHE"
fi
export AWS_ACCOUNT_ID

if ! grep -q "^AWS_REGION=" "$AWS_SETUP_CACHE"; then
    echo "AWS_REGION=$AWS_REGION" >> "$AWS_SETUP_CACHE"
fi

# Helper: persist a key=value into the cache (overwrites existing key).
cache_set() {
    local key="$1" value="$2"
    grep -v "^${key}=" "$AWS_SETUP_CACHE" > "${AWS_SETUP_CACHE}.tmp" || true
    echo "${key}=${value}" >> "${AWS_SETUP_CACHE}.tmp"
    mv "${AWS_SETUP_CACHE}.tmp" "$AWS_SETUP_CACHE"
    export "${key}=${value}"
}

# Pretty logging
log()  { printf "\033[1;34m[aws-setup]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[aws-setup]\033[0m %s\n" "$*" >&2; }
die()  { printf "\033[1;31m[aws-setup]\033[0m %s\n" "$*" >&2; exit 1; }
