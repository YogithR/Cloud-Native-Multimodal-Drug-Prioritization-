# GitHub setup — first push and repository metadata

Use this checklist when you create the repository on GitHub and push this project.

---

## 1) Create the repository on GitHub

1. GitHub → **New repository**.
2. Name example: `cloud-native-multimodal-drug-prioritization` (or your preference).
3. **Do not** add a README if you already have one locally (avoids merge conflicts).
4. Create the repo (empty).

---

## 2) Suggested short description (GitHub “About” box)

Paste into the repository **Description** field:

```text
Multimodal ML platform for drug candidate prioritization: RDKit + graph neural fusion, sklearn baselines, FastAPI ranking API, Docker, MLflow, pytest.
```

Alternative (shorter):

```text
FastAPI + PyTorch graph fusion + sklearn baselines for SMILES-based candidate ranking; Docker & CI included.
```

---

## 3) Suggested topics (GitHub Topics)

Add under **About → Topics** (click the gear icon):

```text
fastapi
pytorch
scikit-learn
rdkit
machine-learning
mlops
docker
cheminformatics
drug-discovery
python
github-actions
mlflow
prometheus
```

You can remove or add tags to match your audience.

---

## 4) Website / portfolio (optional)

If you have a personal site or deployed demo, add it under **About → Website**. Otherwise leave blank.

---

## 5) First push from your machine (PowerShell)

From your project root (where `.git` lives or will live):

```powershell
Set-Location "d:\Cloud-Native-Multimodal-Drug-Prioritization"

git init
git add .
git status
```

Review `git status`: you should **not** see `artifacts/models/`, large `data/processed/`, or `mlruns/` if `.gitignore` is working.

```powershell
git commit -m "Initial commit: multimodal drug prioritization platform with graph fusion and FastAPI"
git branch -M main
git remote add origin https://github.com/YOUR_USER/YOUR_REPO.git
git push -u origin main
```

Replace `YOUR_USER/YOUR_REPO` with your GitHub username and repository name.

---

## 6) What stays out of Git (by design)

`.gitignore` is set to exclude:

- Virtual env (`.venv/`)
- **`mlruns/`** (local MLflow)
- **`data/raw/`**, **`data/interim/`**, **`data/processed/`** (generated datasets)
- **`artifacts/models/`**, **`artifacts/reports/`** (trained weights and run reports)

**Why:** Keeps the repo small and avoids licensing/binary clutter. Cloners follow **README.md** to regenerate data and artifacts, or you can document a optional release with artifacts (separate from this default layout).

---

## 7) Optional: GitHub Actions

The workflow under `.github/workflows/` should run on push/PR once the repo is public or Actions are enabled for private repos.

---

## 8) Optional: LICENSE

Add a `LICENSE` file (e.g. MIT) if you want others to reuse the code clearly; GitHub can generate one when creating the repo or you can add the file locally and push.
