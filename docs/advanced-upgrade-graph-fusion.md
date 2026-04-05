# Advanced upgrade: graph-enhanced multimodal fusion

This phase sits **before Phase 13** (final polish). The project title is unchanged: **Cloud-Native Multimodal AI Platform for Drug Candidate Prioritization**.

## What was added

1. **Graph branch** — RDKit builds atom-bond graphs from SMILES (deterministic atom ordering, padded to `max_atoms`). A small **GCN stack** encodes the graph to a fixed vector.
2. **Tabular branch** — Same multimodal columns as the RF model (metadata + RDKit descriptors + Morgan fingerprint bits), **standardized** with `StandardScaler` fit on the training split only.
3. **Fusion** — Concatenate graph embedding and tabular MLP embedding, then an MLP classification head. Training uses `BCEWithLogitsLoss` with class-balanced `pos_weight` and early stopping on validation ROC-AUC.
4. **Ranking** — Same priority formula as before (`base_score - descriptor_penalties`). **Tie-breaking** is deterministic: `priority_score` descending, then `prediction_confidence` (distance from 0.5) descending, then `candidate_id` ascending. Responses include `prediction_confidence` and optional reason code `uncertain_prediction` near p≈0.5.
5. **Evaluation** — `artifacts/reports/graph_fusion_metrics.json` plus `artifacts/reports/three_way_model_comparison.json` for baseline vs RF multimodal vs graph fusion (metrics from reports, Spearman rank correlations, confusion-style counts, error analysis).

## Commands

**Train** (after processed feature CSVs exist):

```text
python scripts/run_graph_fusion_train.py
```

**Evaluate** (requires trained artifact at `artifacts/models/graph_fusion_nn.joblib`):

```text
python scripts/run_graph_fusion_evaluation.py
```

**Three-way comparison** (requires all three model artifacts and preferably all three eval JSONs):

```text
python scripts/run_three_way_comparison.py
```

**Use graph fusion in the API** — set `ranking.model_key: graph_fusion` in `configs/ranking.yaml` and ensure the graph fusion artifact path in `configs/model.yaml` points to your trained file.

## MLflow

Training and evaluation log to the same local experiment when `configs/mlflow.yaml` has `enabled: true` and `MLFLOW_DISABLE` is unset. Tests set `MLFLOW_DISABLE=1` in `tests/conftest.py`.

## Dependencies

- **PyTorch** (CPU) — neural encoder and fusion head. If imports fail, reinstall a matching CPU wheel (e.g. `pip install --upgrade torch`).

The three-way comparison report uses **pandas** rank correlations (Spearman-equivalent via ranked Pearson), so no extra numeric stack is required beyond scikit-learn’s usual dependencies.

No paid services or proprietary datasets are required.
