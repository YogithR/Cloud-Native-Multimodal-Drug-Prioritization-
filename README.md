# Cloud-Native Multimodal AI Platform for Drug Candidate Prioritization

Local-first, free-tooling ML platform: **multimodal features** from SMILES (RDKit descriptors + fingerprints + metadata), **three model tiers** (baseline logistic regression, Random Forest fusion, **graph + tabular neural fusion**), **evaluation with comparison reports**, and a **FastAPI** service for **prediction and deterministic ranking** with reason codes and confidence. Optional **Docker** and **MLflow** (local file store).

---

## What this project does

| Layer | What you get |
|--------|----------------|
| **Data** | Ingestion and validation for a BBBP-style permeability dataset → processed tables. |
| **Features** | Per-candidate tabular features + **molecular graphs** from SMILES for the neural model. |
| **Models** | Baseline sklearn → multimodal Random Forest → **GCN-style graph branch + MLP tabular branch + fusion head** (PyTorch). |
| **Evaluation** | Locked metrics (ROC-AUC, PR-AUC, F1, balanced accuracy) + **three-way comparison** (metrics, rank correlation, error analysis). |
| **Ranking** | Priority score from model probability + descriptor penalties; **deterministic** tie-break; **`prediction_confidence`** on API responses. |
| **Serving** | FastAPI: `/health`, `/predict`, `/rank`, `/batch-rank`, `/docs`, `/metrics`. |
| **Ops** | Config-driven paths, pytest, Ruff/Black/isort, GitHub Actions CI, Docker Compose. |

Advanced upgrade details: [docs/advanced-upgrade-graph-fusion.md](docs/advanced-upgrade-graph-fusion.md).

---

## Repository layout (short)

| Path | Role |
|------|------|
| `configs/` | `data.yaml`, `model.yaml`, `ranking.yaml`, `mlflow.yaml`, … |
| `src/` | Features, models, ranking, API, monitoring, utils |
| `scripts/` | One-shot pipelines (ingest, features, train, eval, Docker helpers) |
| `docker/` | `Dockerfile.api`, `docker-compose.yml` |
| `tests/` | Unit/integration tests |
| `data/` | Raw / interim / processed (generated; see GitHub note below) |
| `artifacts/` | Trained models and reports (generated) |

---

## Prerequisites

- **Python 3.11+**
- **Docker Desktop** (optional, for containerized API)
- **Git**

On Windows, use **PowerShell** for the commands below.

---

## Quick start (full pipeline)

From the repository root (e.g. `D:\Cloud-Native-Multimodal-Drug-Prioritization`):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e ".[dev]"
```

### 1) Data and features

```powershell
python scripts/run_data_ingestion.py
python scripts/run_feature_pipeline.py
```

**Expected:** CSVs under `data/raw/`, `data/interim/`, `data/processed/` (e.g. `bbbp_*_features.csv`).

### 2) Train models

```powershell
python scripts/run_train_pipeline.py
python scripts/run_graph_fusion_train.py
```

**Expected artifacts:**

- `artifacts/models/baseline_logreg.joblib`
- `artifacts/models/multimodal_fusion_rf.joblib`
- `artifacts/models/graph_fusion_nn.joblib`

### 3) Evaluate and compare

```powershell
python scripts/run_evaluation.py
python scripts/run_graph_fusion_evaluation.py
python scripts/run_three_way_comparison.py
```

**Expected reports:** `artifacts/reports/*.json` including `three_way_model_comparison.json`.

### 4) Tests

```powershell
$env:MLFLOW_DISABLE = "1"
pytest tests -q
```

### 5) Run the API locally

Set the active model in **`configs/ranking.yaml`**:

```yaml
model_key: graph_fusion   # or: baseline | multimodal
```

Then:

```powershell
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

- Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

Stop the server with **Ctrl+C**.

---

## Docker

**Prerequisites:** Same `artifacts/models/*.joblib` files on the host (train locally first). Compose file lives under **`docker/`**, not the repo root.

From **repository root**:

```powershell
$env:COMPOSE_BAKE = "false"
docker compose -f ".\docker\docker-compose.yml" build
docker compose -f ".\docker\docker-compose.yml" up
```

Background:

```powershell
docker compose -f ".\docker\docker-compose.yml" up -d
docker compose -f ".\docker\docker-compose.yml" down
```

More detail: [docs/deployment.md](docs/deployment.md).

**First build** can take a long time (large dependencies including PyTorch). If **C:** disk is tight, move Docker’s **disk image location** to another drive in Docker Desktop → Settings → Resources → Advanced.

---

## MLflow (optional)

Local file store under `./mlruns`. Disable in tests via `MLFLOW_DISABLE=1` (see `tests/conftest.py`).

```powershell
mlflow ui --backend-store-uri file:./mlruns
```

Details: [docs/mlops.md](docs/mlops.md).

---

## CI

GitHub Actions: lint, tests (`MLFLOW_DISABLE=1`), Docker build (no push). See [docs/ci.md](docs/ci.md).

---

## GitHub: cloning vs. what you commit

**`.gitignore`** excludes generated paths such as `data/raw/`, `data/processed/`, `artifacts/models/`, `artifacts/reports/`, and `mlruns/`. After cloning, **run the pipeline above** (or copy artifacts privately) to produce models and reports. This keeps the repo small and avoids committing large binaries.

For **description, topics, and first push**, see [docs/github-setup.md](docs/github-setup.md).

---

## More documentation

- [docs/deployment.md](docs/deployment.md) — Docker mounts and verification  
- [docs/advanced-upgrade-graph-fusion.md](docs/advanced-upgrade-graph-fusion.md) — graph fusion architecture and commands  
- [docs/monitoring.md](docs/monitoring.md) — logging and Prometheus metrics  
- [docs/cloud-aws-free-tier.md](docs/cloud-aws-free-tier.md) — optional EC2 demo (not required)

