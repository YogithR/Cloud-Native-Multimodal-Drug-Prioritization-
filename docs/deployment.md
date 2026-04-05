# Deployment (Phase 8 — Docker, local-first)

**Default path:** run the API with **Docker Compose on your machine** (below). This is the **main** deployment path for the project.

**Optional (Phase 12):** a **temporary** demo on **AWS Free Tier–eligible EC2** is documented separately. It is **not** required for development or for completing the project: [cloud-aws-free-tier.md](cloud-aws-free-tier.md).

---

## Prerequisites

- Docker Desktop (or Docker Engine + Compose v2) installed
- Trained model artifacts on the host, e.g.:
  - `artifacts/models/baseline_logreg.joblib`
  - `artifacts/models/multimodal_fusion_rf.joblib`
  - `artifacts/models/graph_fusion_nn.joblib` (if `ranking.yaml` uses `model_key: graph_fusion`)

The API loads paths from `configs/model.yaml` and `configs/ranking.yaml`. Those paths are relative to the container working directory (`/app`), which matches the repo layout when you mount `artifacts/` read-only.

## Build and run (recommended)

From the **repository root** (not inside `docker/`):

### Windows (PowerShell)

```powershell
docker compose -f docker/docker-compose.yml up --build
```

Or:

```powershell
.\scripts\start_local_stack.ps1
```

### Linux / macOS (bash)

```bash
docker compose -f docker/docker-compose.yml up --build
```

Or:

```bash
./scripts/start_local_stack.sh
```

## Verify

- Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

## What gets mounted

| Host path      | Container path   | Purpose                          |
|----------------|------------------|----------------------------------|
| `./configs`    | `/app/configs`   | Model + ranking + data YAML      |
| `./artifacts`  | `/app/artifacts` | Trained `.joblib` models, reports |

## Notes

- `data/processed/` is **not** required for inference if the model files under `artifacts/models/` are present.
- Re-training on the host updates files on disk; restart the container (or rely on file bind mounts) to pick up new artifacts.
