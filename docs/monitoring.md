# Monitoring and observability (Phase 11)

## What is included

1. **Structured-style request logs** — each HTTP request logs `method`, `path`, `status_code`, `duration_ms` (see `src/monitoring/middleware.py`).
2. **Root logging** — `configure_logging()` in `src/monitoring/logging_config.py`. Override level with **`LOG_LEVEL`** (default `INFO`).
3. **Prometheus metrics** — **`GET /metrics`** exposes the default HTTP and process metrics from [prometheus-fastapi-instrumentator](https://github.com/trallnag/prometheus-fastapi-instrumentator). `/health` is excluded from HTTP histograms to reduce noise.

## Endpoints

| Path | Purpose |
|------|---------|
| `GET /health` | Liveness; includes `version` from `configs/serving.yaml` |
| `GET /metrics` | Prometheus scrape target (text format) |

## Local checks

```powershell
Invoke-WebRequest -Uri http://127.0.0.1:8000/metrics -UseBasicParsing | Select-Object -ExpandProperty Content | Select-Object -First 20
```

## Docker

The API process logs to stdout; container orchestrators can aggregate logs. Prometheus can scrape `http://<host>:8000/metrics` when the port is exposed.
