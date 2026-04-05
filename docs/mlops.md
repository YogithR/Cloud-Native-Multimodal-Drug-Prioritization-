# MLOps (Phase 9 — local MLflow)

## What is tracked

- **Training runs** (`baseline_train`, `multimodal_train`): hyperparameters, tags, sklearn model artifact, training summary JSON.
- **Evaluation runs** (`baseline_eval`, `multimodal_eval`): validation/test metrics (`val_*`, `test_*`), threshold.

Storage is **local and free**: `file:./mlruns` (see `configs/mlflow.yaml`).

## Disable logging

- Environment: `MLFLOW_DISABLE=1`
- Or set `mlflow.enabled: false` in `configs/mlflow.yaml`

## View the UI

From the **repository root** (with venv activated):

```powershell
mlflow ui --backend-store-uri file:./mlruns
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) and select experiment `drug-prioritization-bbbp`.

## Makefile

```powershell
make mlflow-ui
```

## Docker

`docker-compose` mounts `./mlruns` to `/app/mlruns` so runs persist on the host. Run training/eval on the host or inside the container with the same `configs/mlflow.yaml` paths.
