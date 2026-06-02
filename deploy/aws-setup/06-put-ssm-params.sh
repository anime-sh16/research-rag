#!/usr/bin/env bash
# Uploads runtime secrets to SSM Parameter Store under /research-rag/.
# Reads sensitive values from the repo-root .env; takes DuckDNS values as flags.

set -euo pipefail
source "$(dirname "$0")/env.sh"

DUCKDNS_HOST_ARG=""
DUCKDNS_TOKEN_ARG=""
SKIP_DUCKDNS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --duckdns-host)  DUCKDNS_HOST_ARG="$2";  shift 2;;
        --duckdns-token) DUCKDNS_TOKEN_ARG="$2"; shift 2;;
        --skip-duckdns)  SKIP_DUCKDNS=1;          shift;;
        *) die "Unknown arg: $1";;
    esac
done

ENV_FILE="$(cd "$AWS_SETUP_DIR/../.." && pwd)/.env"
[[ -f "$ENV_FILE" ]] || die ".env not found at $ENV_FILE"

# shellcheck disable=SC1090
source "$ENV_FILE"

# PIPELINE_VERSION is deliberately NOT seeded: the deployed app reads it from the
# baked-in VERSION file (see deploy/Dockerfile.api), so /health always reflects the
# commit it was built from instead of a drifting SSM value.
for key in GOOGLE_API_KEY JINA_API_KEY QDRANT_URL QDRANT_API_KEY LANGSMITH_API_KEY LANGSMITH_PROJECT; do
    if [[ -z "${!key:-}" ]]; then
        die "$key is empty in $ENV_FILE"
    fi
done

[[ -n "${ECR_REGISTRY:-}" ]] || die "ECR_REGISTRY not in cache. Run 02-create-ecr-repos.sh first."

put() {
    local name="$1" type="$2" value="$3"
    log "Putting $name ($type)"
    aws ssm put-parameter \
        --name "$name" \
        --type "$type" \
        --value "$value" \
        --overwrite \
        >/dev/null
}

put "$SSM_PREFIX/google-api-key"    SecureString "$GOOGLE_API_KEY"
put "$SSM_PREFIX/jina-api-key"      SecureString "$JINA_API_KEY"
put "$SSM_PREFIX/qdrant-url"        String       "$QDRANT_URL"
put "$SSM_PREFIX/qdrant-api-key"    SecureString "$QDRANT_API_KEY"
put "$SSM_PREFIX/langsmith-api-key" SecureString "$LANGSMITH_API_KEY"
put "$SSM_PREFIX/langsmith-project"  String       "$LANGSMITH_PROJECT"
put "$SSM_PREFIX/ecr-registry"      String       "$ECR_REGISTRY"

if [[ "$SKIP_DUCKDNS" -eq 0 ]]; then
    [[ -n "$DUCKDNS_HOST_ARG" ]]  || die "Pass --duckdns-host <subdomain>.duckdns.org or --skip-duckdns"
    [[ -n "$DUCKDNS_TOKEN_ARG" ]] || die "Pass --duckdns-token <token> or --skip-duckdns"
    put "$SSM_PREFIX/duckdns-host"  String       "$DUCKDNS_HOST_ARG"
    put "$SSM_PREFIX/duckdns-token" SecureString "$DUCKDNS_TOKEN_ARG"
fi

log "All parameters uploaded to $SSM_PREFIX/"
