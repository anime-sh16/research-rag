#!/usr/bin/env bash
# Build linux/amd64 images and push to ECR.
# Run from anywhere — paths are resolved relative to this script.
# Requires: Docker daemon running (colima start on Mac), AWS CLI, correct profile.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]:-$0}")/env.sh"

[[ -n "${ECR_REGISTRY:-}" ]] || die "ECR_REGISTRY not in cache. Run 02-create-ecr-repos.sh first."

REPO_ROOT="$(cd "$AWS_SETUP_DIR/../.." && pwd)"

# Use git SHA as the image tag so every push is uniquely addressable.
GIT_SHA=$(git -C "$REPO_ROOT" rev-parse --short HEAD)
IMAGE_TAG="$GIT_SHA"

log "Repo root : $REPO_ROOT"
log "ECR       : $ECR_REGISTRY"
log "Image tag : $IMAGE_TAG"

# ── ECR login ────────────────────────────────────────────────────────────────
log "Authenticating with ECR"
aws ecr get-login-password --region "$AWS_REGION" \
    | docker login --username AWS --password-stdin "$ECR_REGISTRY"

# ── Build + push helper ───────────────────────────────────────────────────────
build_and_push() {
    local name="$1"        # e.g. research-rag-api
    local dockerfile="$2"  # relative to REPO_ROOT
    local full_name="$ECR_REGISTRY/$name"

    log "Building $name (linux/amd64)"
    docker build \
        --platform linux/amd64 \
        --file "$REPO_ROOT/$dockerfile" \
        --tag "$full_name:$IMAGE_TAG" \
        --tag "$full_name:latest" \
        "$REPO_ROOT"

    log "Pushing $name:$IMAGE_TAG"
    docker push "$full_name:$IMAGE_TAG"

    log "Pushing $name:latest"
    docker push "$full_name:latest"
}

build_and_push "research-rag-api" "deploy/Dockerfile.api"
build_and_push "research-rag-ui"  "deploy/Dockerfile.ui"

# Persist the tag so subsequent scripts can reference it.
cache_set IMAGE_TAG "$IMAGE_TAG"

log "Done. Images tagged $IMAGE_TAG and latest in ECR."
log "Next: run 11-start-stack.sh to pull and start the stack on EC2."
