#!/usr/bin/env bash
# Updates the DuckDNS A record with the EC2 instance's current public IPv4.
# Reads DUCKDNS_HOST and DUCKDNS_TOKEN from /opt/research-rag/.env.

set -euo pipefail

ENV_FILE="${ENV_FILE:-/opt/research-rag/.env}"

if [[ ! -r "$ENV_FILE" ]]; then
    echo "duckdns-update: missing $ENV_FILE" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "$ENV_FILE"

: "${DUCKDNS_HOST:?DUCKDNS_HOST not set}"
: "${DUCKDNS_TOKEN:?DUCKDNS_TOKEN not set}"

# DuckDNS expects only the subdomain (the part before .duckdns.org).
domain="${DUCKDNS_HOST%%.duckdns.org}"

response=$(curl -fsS "https://www.duckdns.org/update?domains=${domain}&token=${DUCKDNS_TOKEN}&ip=" || true)

if [[ "$response" != "OK" ]]; then
    echo "duckdns-update: unexpected response: $response" >&2
    exit 1
fi

echo "duckdns-update: $domain.duckdns.org updated"
