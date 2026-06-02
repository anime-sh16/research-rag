#!/usr/bin/env bash
# Idempotent ECR repo creation.

set -euo pipefail
source "$(dirname "$0")/env.sh"

create_repo() {
    local name="$1"
    if aws ecr describe-repositories --repository-names "$name" >/dev/null 2>&1; then
        log "ECR repo $name already exists — skipping"
    else
        log "Creating ECR repo $name"
        aws ecr create-repository \
            --repository-name "$name" \
            --image-tag-mutability MUTABLE \
            --image-scanning-configuration scanOnPush=true \
            >/dev/null
    fi
}

create_repo "$ECR_REPO_API"
create_repo "$ECR_REPO_UI"

ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
cache_set ECR_REGISTRY "$ECR_REGISTRY"
log "ECR registry: $ECR_REGISTRY"
