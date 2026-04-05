# Resume and portfolio copy — Drug candidate prioritization platform

Use this file to paste into your resume, LinkedIn “About,” or portfolio site. Tweak numbers only if you re-ran eval and have different metrics.

**Official project title (do not change for branding consistency):**  
**Cloud-Native Multimodal AI Platform for Drug Candidate Prioritization**

---

## One-line pitch

End-to-end **multimodal ML** platform for **drug-like candidates**: RDKit features and **molecular graphs**, **sklearn baselines** plus a **graph–tabular neural fusion** model, **three-way evaluation**, and a **FastAPI** service with **deterministic ranking**, **confidence**, and **reason codes** — **Docker**-ready and **MLflow**-tracked locally.

---

## Resume bullets (choose 3–4)

- Built a **local-first multimodal AI platform** for **candidate prioritization** using **SMILES**-based **RDKit** descriptors, **Morgan fingerprints**, and **metadata**, with reproducible **train/val/test** splits and **YAML-driven** configuration.

- Implemented **three model tiers** for comparison: **logistic regression** baseline, **Random Forest** multimodal fusion, and **PyTorch** **graph encoder (GCN-style) + tabular MLP + fusion head**; logged training/eval with **MLflow** (local file store).

- Delivered **honest evaluation**: shared metrics (**ROC-AUC, PR-AUC, F1, balanced accuracy**) and a **three-way report** combining **metric tables**, **ranking agreement** (Spearman-style on ranked probabilities/priorities), and **error-style analysis** on the test split.

- Shipped a **FastAPI** inference API (**health, predict, rank, batch-rank, OpenAPI docs, Prometheus metrics**) with **deterministic ranking** (descriptor penalties, tie-breaking) and **`prediction_confidence`** plus **transparent reason codes**.

- Added **pytest** coverage, **Ruff/Black/isort**, **GitHub Actions** CI, and **Docker Compose** deployment with **host-mounted configs and artifacts**; documented **PowerShell** workflows for Windows.

---

## LinkedIn / portfolio short paragraph

I built a **cloud-native-style** (container-ready) **multimodal ML** system for **drug candidate prioritization**. The stack combines **classical** and **neural** models: a **graph branch** over **RDKit-derived molecular graphs** fused with **tabular** chemistry features, compared against **baseline** and **ensemble** models. The service exposes **REST** endpoints for **scoring and ranking** with **interpretability-friendly** outputs, backed by **automated tests** and **CI**.

---

## Interview talking points (30-second version)

1. **Problem framing:** Binary classification on a **permeability-style** (BBBP-like) task; label semantics drive whether high probability is “good” or “risk” in ranking — configurable in YAML.

2. **Multimodal design:** **Tabular** modality = metadata + descriptors + fingerprints; **graph** modality = atoms/bonds from SMILES → **GCN-style** encoder; **fusion** = concatenate embeddings → MLP head.

3. **Why not only Random Forest:** The neural model lets you **separate** structure (graph) from hand-engineered/tabular signals and shows **modern DL + chemistry** literacy without claiming unrealistic scale.

4. **Ranking vs. raw probability:** We keep **deterministic** rules (penalties, tie-breaks) so product behavior is **auditable**; the model supplies **P(y=1)** and we add **confidence** for UX and debugging.

5. **Engineering:** **Artifacts on disk**, **no paid services**, **local MLflow**, **Docker** for parity with deploy; tests gate regressions.

---

## Skills tags (for ATS or GitHub Topics)

`Python` · `PyTorch` · `scikit-learn` · `FastAPI` · `RDKit` · `Docker` · `MLflow` · `pytest` · `GitHub Actions` · `Prometheus` · `MLOps` · `cheminformatics` · `multimodal ML`

---

## Optional “metrics” sentence (only if asked)

After training locally, evaluation JSONs under `artifacts/reports/` report **val/test** ROC-AUC, PR-AUC, F1, and balanced accuracy for **baseline**, **multimodal RF**, and **graph fusion**; exact numbers depend on split seed and run — cite **your** `graph_fusion_metrics.json` if an interviewer wants a number.
