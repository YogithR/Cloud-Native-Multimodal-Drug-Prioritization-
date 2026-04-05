# Continuous integration (Phase 10)

## GitHub Actions workflow

Workflow file: [`.github/workflows/ci.yml`](../.github/workflows/ci.yml).

**Triggers:** pushes and pull requests to any branch, plus manual **workflow_dispatch**.

**Jobs:**

1. **Lint and test (Python 3.11)**  
   - Pip cache keyed on `pyproject.toml`  
   - `ruff check`, `black --check`, `isort --check-only`  
   - `pytest` with **`MLFLOW_DISABLE=1`** (no MLflow writes in CI)

2. **Docker build**  
   - Builds `docker/Dockerfile.api` with Buildx  
   - **Does not push** images (free, no registry required)  
   - Uses GitHub Actions cache for Docker layers when available  

**Concurrency:** duplicate runs for the same branch are cancelled to save minutes.

## Local parity

From the repo root (Git Bash / WSL / macOS / Linux):

```bash
make ci
```

On **Windows PowerShell** (no `make`), run the same steps manually:

```powershell
ruff check src tests
black --check src tests
isort --check-only src tests
$env:MLFLOW_DISABLE = "1"
pytest -q
```

## Requirements for green CI

- No paid services  
- No secrets required for default workflow  
- Training data and large artifacts are **not** downloaded in CI; tests are self-contained  
