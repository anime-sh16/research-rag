# deploy/

All deployment artifacts for the research-rag EC2 stack. App-side code is in `../src/`; AWS infrastructure provisioning scripts are in `aws-setup/`.

**Live stack:** [https://research-rag-animesh.duckdns.org](https://research-rag-animesh.duckdns.org)
**EC2 instance:** `i-02680d0acb8704ca0` (t3.micro, ap-northeast-1)
**ECR registry:** `421765950243.dkr.ecr.ap-northeast-1.amazonaws.com`

---

## Files

| File | Where it runs | Purpose |
|---|---|---|
| `Dockerfile.api` | local build / ECR | Multi-stage uv image for FastAPI |
| `Dockerfile.ui` | local build / ECR | Multi-stage uv image for Streamlit |
| `docker-compose.yml` | EC2 `/opt/research-rag/deploy/` | Production 3-service stack: caddy + api + ui |
| `docker-compose.local.yml` | local | Override: skips Caddy, exposes ports 8000 + 8501 directly |
| `Caddyfile` | inside `caddy` container | Reverse proxy config + auto Let's Encrypt TLS |
| `fetch-secrets.sh` | EC2, as root, via systemd | Pulls all `/research-rag/*` SSM params into `/opt/research-rag/.env` |
| `duckdns-update.sh` | EC2, as root, via systemd timer | Updates DuckDNS A record with current public IP |
| `systemd/research-rag-secrets.service` | EC2 | One-shot: runs `fetch-secrets.sh` before the compose stack starts |
| `systemd/research-rag.service` | EC2 | Starts the compose stack on boot |
| `systemd/duckdns-update.service` | EC2 | One-shot: runs `duckdns-update.sh` |
| `systemd/duckdns-update.timer` | EC2 | Triggers the DNS updater at boot + every 30 minutes |

---

## Local development

Pre-reqs: Colima running (`colima start`), `.env` populated at repo root.

```bash
# From repo root
DUCKDNS_HOST=localhost docker compose \
  --project-directory . \
  -f deploy/docker-compose.yml \
  -f deploy/docker-compose.local.yml \
  up --build
```

- API → `http://localhost:8000/health`
- UI  → `http://localhost:8501`

Tear down:

```bash
docker compose --project-directory . \
  -f deploy/docker-compose.yml \
  -f deploy/docker-compose.local.yml \
  down -v
```

---

## Deploying to EC2

### First-time setup

Run the `aws-setup/` scripts in order (01 → 11). See [aws-setup/README.md](aws-setup/README.md) for the full runbook.

### Re-deploying after code changes

```bash
# 1. Build and push new images to ECR
bash deploy/aws-setup/10-first-image-push.sh

# 2. Pull and restart the stack on EC2
bash deploy/aws-setup/11-start-stack.sh
```

### Checking stack status on EC2

Open an SSM session (no SSH required):

```bash
aws ssm start-session --target i-02680d0acb8704ca0
```

Then inside the session:

```bash
cd /opt/research-rag/deploy
docker compose --env-file /opt/research-rag/.env ps
docker compose --env-file /opt/research-rag/.env logs --tail=50
```

---

## Environment variables

| Variable | Required | EC2 source | Notes |
|---|---|---|---|
| `GOOGLE_API_KEY` | yes | SSM `/research-rag/google-api-key` | Gemini embedding + LLM |
| `JINA_API_KEY` | yes | SSM `/research-rag/jina-api-key` | Reranker |
| `QDRANT_URL` | yes | SSM `/research-rag/qdrant-url` | Qdrant Cloud cluster |
| `QDRANT_API_KEY` | yes | SSM `/research-rag/qdrant-api-key` | |
| `LANGSMITH_API_KEY` | yes | SSM `/research-rag/langsmith-api-key` | Tracing |
| `LANGSMITH_PROJECT` | yes | SSM `/research-rag/langsmith-project` | |
| `DUCKDNS_HOST` | yes | SSM `/research-rag/duckdns-host` | Caddy uses this as the TLS hostname |
| `DUCKDNS_TOKEN` | yes | SSM `/research-rag/duckdns-token` | DNS updater token |
| `ECR_REGISTRY` | yes | SSM `/research-rag/ecr-registry` | Image pull source |
| `IMAGE_TAG` | no | compose default `latest` | Pin to a git SHA for rollback |
| `API_BASE_URL` | no | compose default `http://api:8000` | Streamlit → FastAPI |
| `REPO_URL` | no | compose default `https://github.com/` | Sidebar link in UI |

Secrets are fetched at boot by `fetch-secrets.sh` and written to `/opt/research-rag/.env` (mode `600`, owned by root). The compose stack reads them via `--env-file /opt/research-rag/.env`.

---

## What this directory does NOT do

- Does not provision AWS resources — see `aws-setup/`
- Does not configure GitHub Actions — see `.github/workflows/` (Plan C)
- Does not ingest data into Qdrant — that is a local CLI workflow (`uv run python -m src.ingestion.pipeline`)
