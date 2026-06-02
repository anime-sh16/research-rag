# GitHub Actions Workflows

## ci.yml — Continuous Integration
**Triggers:** every push to any branch, every PR to `main`

Runs ruff lint, ruff format check, and pytest. Must pass before merging to `main`.

| Branch | CI | Deploy |
|---|---|---|
| any branch (e.g. `dev`, `feature/*`) | ✅ runs | ❌ never |
| `main` | ✅ runs | ✅ runs |

---

## deploy.yml — Build & Deploy
**Triggers:** push to `main` (automatic) | manual via workflow_dispatch

### Normal flow (every merge to main)
```
build-and-push → ragas-gate (skipped) → deploy
```
Builds `linux/amd64` Docker images, pushes SHA tag to ECR, restarts EC2 stack via SSM.

### Gated deploy (manual — for portfolio releases)
Go to **Actions → Deploy → Run workflow → check `run_ragas_gate`**
```
build-and-push ──┐
                 ├── both must pass → tag latest in ECR → deploy
ragas-gate ──────┘
```
Runs full RAGAS evaluation, compares composite score against baseline. Blocks deploy if score drops more than 0.02 below baseline. Results appear in the job summary and as a downloadable artifact.

**When to use:** before a portfolio release, when you want a CI-logged proof that the deployed version meets the quality bar.

### latest tag behaviour
`latest` in ECR is only updated after a successful deploy — never during build alone. Safe to redeploy manually at any time.

### Required secrets
| Secret | Used by |
|---|---|
| `AWS_ROLE_ARN` | all AWS steps (OIDC, no static keys) |
| `AWS_REGION` | all AWS steps |
| `ECR_REGISTRY` | image push + EC2 pull |
| `EC2_INSTANCE_ID` | SSM deploy command |
| `GOOGLE_API_KEY` | RAGAS gate only |
| `QDRANT_URL` | RAGAS gate only |
| `QDRANT_API_KEY` | RAGAS gate only |
| `JINA_API_KEY` | RAGAS gate only |
| `LANGSMITH_API_KEY` | RAGAS gate only |
| `LANGSMITH_PROJECT` | RAGAS gate only |

---

## ragas-ondemand.yml — Manual RAGAS Evaluation
**Triggers:** manual via workflow_dispatch only

Run from **Actions → RAGAS Evaluation → Run workflow**.

Inputs:
- `experiment` — name for this run (e.g. `v3-mmr`). Must match the `pipeline_version` set in `src/config/config.py`.
- `run_gate` — whether to check scores against baseline after evaluation (default: true)
- `baseline` — path to baseline results JSON (default: `v2-hybrid-rerank-v2`)

Results are uploaded as a workflow artifact and logged to LangSmith.

**When to use:** after implementing a retrieval or generation improvement on `dev`, before merging to `main`.

---

## Typical workflow

```
dev branch
  │
  ├── implement improvement
  ├── run ragas-ondemand.yml  ← evaluate, check scores
  ├── satisfied? merge to main
  │
main branch
  └── deploy.yml fires automatically ← build + deploy, no re-evaluation
```

For portfolio release:
```
main branch
  └── Actions → Deploy → Run workflow → check run_ragas_gate
        └── gated deploy with CI-logged proof
```
