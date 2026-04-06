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

---

## Future Goal: Toward a Real Biotech Production System

The repository today is a **strong, end-to-end, local-first platform** for **drug candidate prioritization**: reproducible data and features, **multiple comparable models** (including graph-enhanced multimodal fusion), **honest evaluation**, **deterministic ranking**, and a **container-ready API**. That scope is intentional—it demonstrates how ML platforms are structured without claiming a regulated or proprietary production deployment.

The **long-term direction** is to evolve this foundation toward a **more realistic biotech production system**. The items below are **planned next-step improvements**, not capabilities shipped in this repo. They reflect how industrial R&D and informatics teams typically mature such systems.

### 1. Real-world biotech data expansion

- Move beyond the current **benchmark-style** dataset to **curated internal or licensed** data where appropriate.
- Integrate richer **assay**, **ADMET**, **bioactivity**, and **target-related** signals as they become available.
- Improve **data lineage** and **dataset versioning** so training and evaluation are traceable over time.

### 2. Stronger biological context

- Add **protein / target**–aware features or linked modalities where the use case requires it.
- Support **richer multimodal inputs** (e.g. additional structured assay readouts, pathway-level context—scoped to real data agreements).
- Align modeling and ranking more closely with **actual drug-discovery decision points** (not only a single public benchmark task).

### 3. More production-grade modeling

- Extend the **graph-enhanced multimodal** stack (architecture search, pretraining, or domain-specific encoders—subject to data and compute).
- Support **multi-objective** prediction (several endpoints or constraints) instead of relying on a **single-task** setup alone.
- Add **calibration** and **stronger uncertainty estimation** where decisions depend on reliable confidence.

### 4. Better ranking and decision support

- Make **ranking** more **biologically meaningful** (e.g. tie-break rules and penalties grounded in program-specific ADME/Tox strategy).
- Support **multi-factor prioritization** (portfolio constraints, risk–benefit composites).
- Deepen **explainability** for **scientist-facing** workflows (auditable reasons, sensitivity analysis, cohort-level reporting).

### 5. Validation for real-world use

- Introduce **scaffold splits**, **temporal splits**, and **external validation** when external benchmarks or partner data allow.
- Strengthen **robustness checks** (subgroups, label noise, domain shift).
- Report evaluation that better reflects **unseen chemistry** and deployment risk.

### 6. Production MLOps and platform maturity

- Strengthen **data and model lineage** (what was trained on what, when, and by whom).
- Improve **model registry**, **promotion workflows**, and **reproducible** training pipelines at scale.
- Harden **deployment**, **monitoring**, **drift detection**, and **auditability** for long-running services.

### 7. Security and enterprise readiness

- Add **authentication** and **authorization** for APIs and batch jobs.
- Enforce **secure handling** of **sensitive and proprietary** structures and assay data.
- Adopt **enterprise-grade** deployment patterns (network isolation, secrets management, change control) where required.

