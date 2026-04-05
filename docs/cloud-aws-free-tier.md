# Phase 12 — Optional AWS deployment (AWS Free Tier only)

This document is **optional**. The project’s **default and main** deployment path remains **local Docker** (see [deployment.md](deployment.md)).

**Scope:** a **short, temporary demo** on **one small EC2 instance** running the existing API container. This is **not** an always-on production architecture.

**Rules used here:**

- Only steps that target **AWS Free Tier–eligible** usage for **eligible accounts** (see [AWS Free Tier](https://aws.amazon.com/free/)).
- No dependency on upgrading, subscriptions, or non-free-tier services for this path.
- **Delete everything** after your demo window ends.

---

## 1) Selected option: single EC2 instance + Docker

**What:** One **Amazon EC2** instance in the **Free Tier–eligible** size (**`t2.micro`** or, where `t2.micro` is unavailable, **`t3.micro`**) running **Docker Engine** and your **`docker/Dockerfile.api`** image (or `docker compose` with the same image and bind mounts).

**Why this fits AWS Free Tier (eligible accounts):**

- AWS documents **750 hours per month** of **Linux `t2.micro` / `t3.micro`** usage for **new accounts** for **12 months** under the **Free Plan** (confirm current terms on the official AWS Free Tier page for your account type and region).
- **No Application Load Balancer**, **no RDS**, **no ECS/EKS control plane**, **no extra managed services** — only one small VM and Docker, which keeps the footprint minimal and easier to tear down.

---

## 2) What you need before starting

- An AWS account that **shows as eligible** for **Free Tier** in the AWS Billing / Free Tier dashboard.
- **AWS CLI v2** installed and configured (`aws configure`) with **least privilege** IAM user for this demo (optional but recommended).
- Your **project built locally**: `artifacts/models/*.joblib` present, `configs/` ready (same as local Docker).
- **SSH key pair** (.pem) created in the target region (EC2 → Key Pairs).

---

## 3) Exact deployment steps (minimal)

These steps are **manual** (AWS Console + SSH). Adjust region and names as you like.

### A. Networking and security (keep it tight)

1. In **EC2 → Security Groups**, create **`sg-drug-api-demo`** in your VPC (default VPC is fine for a short demo).
2. **Inbound rules** (example — restrict to **your** public IP for SSH and HTTP to the API port):
   - **SSH (22)** — source: **My IP** (not `0.0.0.0/0` unless you have no alternative).
   - **Custom TCP (8000)** — source: **My IP** (same idea).
3. **Outbound:** default (allow HTTPS for package installs if needed).

### B. Launch the instance

1. **EC2 → Launch instance**
2. **Name:** `drug-prioritization-demo` (example)
3. **AMI:** **Amazon Linux 2023** (or **Ubuntu 22.04 LTS**)
4. **Instance type:** **`t2.micro`** or **`t3.micro`** (pick the one marked **Free tier eligible** in the wizard)
5. **Key pair:** your existing key
6. **Network:** default subnet + **Auto-assign public IP: Enable** (for a simple public demo URL)
7. **Storage:** **30 GiB or less** **gp2** (stay within Free Tier storage limits documented for your account)
8. **Security group:** `sg-drug-api-demo`
9. **Launch**

Wait until **Instance state = Running**. Note the **Public IPv4 address**.

### C. Install Docker on the instance (SSH)

```bash
# Example: Amazon Linux 2023 — verify current Docker install docs in AWS docs if this changes
sudo dnf update -y
sudo dnf install -y docker
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user
# Log out and SSH back in so group membership applies
```

### D. Copy project files to the instance

From your **laptop** (repo root), use **SCP** or **rsync** (replace host and key path):

```bash
# Example: copy configs + artifacts only (smallest cloud footprint for inference)
scp -i /path/to/key.pem -r configs artifacts ec2-user@YOUR_PUBLIC_IP:~/
```

If you prefer the **full repo**, copy the whole project folder instead; the image build needs `pyproject.toml`, `src/`, `configs/`, etc.

### E. Build and run the API container (same as local philosophy)

On the EC2 instance (after copying files):

```bash
mkdir -p app && cd app
# If you copied the full repo:
# git clone <your-repo-url> .   # OR scp the full tree

# Build from docker/Dockerfile.api at repo root (adjust paths if different)
docker build -f docker/Dockerfile.api -t drug-api:demo .

docker run -d --name drug-api \
  -p 8000:8000 \
  -v "$HOME/configs:/app/configs:ro" \
  -v "$HOME/artifacts:/app/artifacts:ro" \
  drug-api:demo
```

**Monitoring (free-tier-safe):** the app already exposes **`GET /health`** and **`GET /metrics`** (Phase 11). Do **not** turn on optional AWS observability add-ons beyond what **your** account’s **Free Tier** explicitly includes.

### F. Verify from your laptop

```bash
curl -fsS http://YOUR_PUBLIC_IP:8000/health
curl -fsS http://YOUR_PUBLIC_IP:8000/metrics | head
```

Open `http://YOUR_PUBLIC_IP:8000/docs` in a browser (only if your security group allows **your IP** on port **8000**).

---

## 4) Exact teardown / cleanup (mandatory after the demo)

Do these in order and **confirm in the AWS Console** that nothing is left running.

| Step | Action |
|------|--------|
| 1 | **Stop and remove** the Docker container on the instance (SSH): `docker rm -f drug-api` |
| 2 | **Terminate** the EC2 instance: **EC2 → Instances → Instance state → Terminate** |
| 3 | **Elastic IP:** If you **allocated** a static Elastic IP for this demo, **release** it after the instance is gone. **Unattached** Elastic IPs can incur charges — **always release** if you created one |
| 4 | **Security group:** Delete **`sg-drug-api-demo`** if it is **not** used by anything else |
| 5 | **Key pair:** Optional — delete the demo key pair **only if** you will not reuse it |
| 6 | **Volumes:** After termination, confirm the root volume is **deleted** (default delete on terminate should be **on** for the demo instance) |

---

## 5) Billing-safety checklist (Free Tier–focused)

Complete **before** and **after** the demo:

- [ ] In **AWS Billing → Free Tier**, confirm your account shows **Free Plan / Free Tier** eligibility the way AWS documents it for your account.
- [ ] In **AWS Budgets**, create a **budget with an alert** (for example **$1 USD** actual or forecasted) emailed to you so you get notified if anything unexpected appears. (Budgets alerts themselves are part of normal account hygiene for demos.)
- [ ] **Do not** leave the instance running 24/7 unless you have verified it remains within **your** Free Tier monthly hour allowance.
- [ ] **Do not** create **NAT Gateways**, **ALB**, **RDS**, **ElastiCache**, **MSK**, or other services not required for this single-VM demo.
- [ ] After the demo: **terminate the EC2 instance**, **release Elastic IP** if any, **verify** **EC2 → Instances** shows **no running** demo instances.

---

## 6) Resources that must be removed after testing

Remove **all** of the following if you created them for this demo:

- The **EC2 instance** (terminate)
- Any **Elastic IP** you allocated (release)
- The **security group** created only for this demo (if unused)
- Optional: the **key pair** (only if disposable)

Keep **local** development unchanged: your laptop repo, local Docker, and CI workflows do not require AWS.

---

## 7) How to verify the optional cloud deployment works

1. **`/health`** returns JSON with **`status`** and **`version`** (same contract as local).
2. **`/docs`** loads in the browser when allowed by your security group.
3. **`POST /predict`** with a small JSON body returns probabilities (same as local), proving the container sees **`artifacts/models/`** and **`configs/`**.

If any step fails, **prefer fixing locally first**, then re-copy files and restart the container — the cloud path should mirror local Docker behavior.

---

## 8) Relationship to the main project workflow

| Path | Role |
|------|------|
| **Local `uvicorn` / local Docker Compose** | **Default** — primary way to run and develop |
| **This document** | **Optional** — short-lived AWS Free Tier demo only |

Do **not** treat this AWS path as required to complete the project. **Local-first** remains the source of truth.
