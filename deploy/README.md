# deploy/

All deployment artifacts for the research-rag EC2 stack. App-side code is in `../src/`; runtime infrastructure (AWS resources, GitHub Actions) lives elsewhere — see the design spec at [`docs/superpowers/specs/2026-05-21-ec2-free-tier-deployment-design.md`](../docs/superpowers/specs/2026-05-21-ec2-free-tier-deployment-design.md).

## Files

| File | Where it runs | Purpose |
|---|---|---|
| `Dockerfile.api` | local build, ECR | Multi-stage image for FastAPI |
| `Dockerfile.ui` | local build, ECR | Multi-stage image for Streamlit |
| `../.dockerignore` (repo root) | docker build context | Keeps images lean — must live at the build context root |
| `docker-compose.yml` | EC2 (`/opt/research-rag/deploy/`) and local smoke | 3-service stack: caddy + api + ui |
| `Caddyfile` | inside the `caddy` container | Reverse proxy + auto-TLS |
| `fetch-secrets.sh` | EC2, as root, via systemd | Pulls `/research-rag/*` SSM params into `/opt/research-rag/.env` |
| `duckdns-update.sh` | EC2, as root, via systemd timer | Updates DuckDNS A record with current public IP |
| `systemd/research-rag-secrets.service` | EC2 | One-shot: runs `fetch-secrets.sh` before compose |
| `systemd/research-rag.service` | EC2 | Brings up the compose stack on boot |
| `systemd/duckdns-update.service` | EC2 | One-shot: runs `duckdns-update.sh` |
| `systemd/duckdns-update.timer` | EC2 | Calls the service at boot + every 30 minutes |

## Local smoke test

Pre-reqs: Colima running (`colima status` shows `Running`), `.env` populated at repo root (`GOOGLE_API_KEY`, `JINA_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`).

Build the images (from repo root):

```bash
docker buildx build --platform linux/amd64 -f deploy/Dockerfile.api -t research-rag-api:dev --load .
docker buildx build --platform linux/amd64 -f deploy/Dockerfile.ui  -t research-rag-ui:dev  --load .
```

Bring the stack up:

```bash
DUCKDNS_HOST=:80 \
ECR_REGISTRY=local \
IMAGE_TAG=dev \
ENV_FILE=$(pwd)/.env \
  docker compose -f deploy/docker-compose.yml up -d
```

Verify:

```bash
curl -fsS http://localhost/api/health
# → {"status":"ok","pipeline_version":"..."}
```

Open `http://localhost/` for the Streamlit UI.

Tear down:

```bash
docker compose -f deploy/docker-compose.yml down -v
```

## Environment variables consumed at runtime

| Variable | Required | Source on EC2 | Notes |
|---|---|---|---|
| `GOOGLE_API_KEY` | yes | SSM `/research-rag/google-api-key` | Gemini embedding + LLM |
| `JINA_API_KEY` | yes | SSM `/research-rag/jina-api-key` | Reranker |
| `QDRANT_URL` | yes | SSM `/research-rag/qdrant-url` | Qdrant Cloud cluster |
| `QDRANT_API_KEY` | yes | SSM `/research-rag/qdrant-api-key` | |
| `LANGSMITH_API_KEY` | yes | SSM `/research-rag/langsmith-api-key` | Tracing |
| `LANGSMITH_PROJECT` | yes | SSM `/research-rag/langsmith-project` | |
| `DUCKDNS_HOST` | yes | SSM `/research-rag/duckdns-host` | Caddy uses this as the cert hostname |
| `DUCKDNS_TOKEN` | yes | SSM `/research-rag/duckdns-token` | Updater token |
| `ECR_REGISTRY` | yes | SSM `/research-rag/ecr-registry` | `<acct>.dkr.ecr.us-east-1.amazonaws.com` |
| `IMAGE_TAG` | no | compose default `latest` | Pin to a sha for rollback |
| `API_BASE_URL` | no | compose default `http://api:8000` | Streamlit → FastAPI |
| `REPO_URL` | no | compose default `https://github.com/` | Sidebar link |

## What this directory does NOT do

- It does not provision AWS resources. See Plan B.
- It does not configure GitHub Actions. See Plan C.
- It does not ingest data into Qdrant. That stays a local-CLI workflow (`uv run python main.py`).
