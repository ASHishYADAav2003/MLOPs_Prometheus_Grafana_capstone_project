# 🚀 MLOps Capstone Project — Flask ML App with Prometheus & Grafana Monitoring

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey?logo=flask)](https://flask.palletsprojects.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.15-orange)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-3.53-purple)](https://dvc.org/)
[![Docker](https://img.shields.io/badge/Docker-containerized-blue?logo=docker)](https://www.docker.com/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-EKS-326CE5?logo=kubernetes)](https://kubernetes.io/)
[![Prometheus](https://img.shields.io/badge/Prometheus-monitoring-E6522C?logo=prometheus)](https://prometheus.io/)
[![Grafana](https://img.shields.io/badge/Grafana-dashboards-F46800?logo=grafana)](https://grafana.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A production-grade MLOps capstone project implementing the complete machine learning lifecycle — from experiment tracking and data versioning to containerized deployment on AWS EKS with real-time monitoring using **Prometheus** and **Grafana**.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Phase 1 — Project Setup](#phase-1--project-setup)
- [Phase 2 — MLflow on DagsHub](#phase-2--mlflow-on-dagshub)
- [Phase 3 — DVC Pipeline](#phase-3--dvc-pipeline)
- [Phase 4 — Flask App & Docker](#phase-4--flask-app--docker)
- [Phase 5 — CI/CD with GitHub Actions](#phase-5--cicd-with-github-actions)
- [Phase 6 — AWS EKS Deployment](#phase-6--aws-eks-deployment)
- [Phase 7 — Prometheus Setup](#phase-7--prometheus-setup)
- [Phase 8 — Grafana Setup](#phase-8--grafana-setup)
- [AWS Resource Cleanup](#aws-resource-cleanup)
- [License](#license)

---

## Overview

This project demonstrates an end-to-end MLOps workflow for a machine learning model served via a Flask REST API. The project is scaffolded using the **Cookiecutter Data Science** template and covers:

- Experiment tracking with **MLflow** and **DagsHub**
- Reproducible ML pipelines with **DVC** (backed by **AWS S3**)
- Containerization with **Docker** and image registry on **AWS ECR**
- Scalable deployment on **AWS EKS** (Kubernetes, managed via `eksctl`)
- Automated CI/CD via **GitHub Actions**
- Real-time metrics scraping with **Prometheus** (hosted on EC2)
- Visualization and alerting with **Grafana** (hosted on EC2)

---

## Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│                        GitHub Actions CI/CD                           │
│      Test → DVC Repro → Docker Build → Push to ECR → Deploy to EKS   │
└────────────────────────────────┬──────────────────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │       AWS EKS Cluster    │
                    │   (flask-app-cluster)    │
                    │  ┌─────────────────────┐ │
                    │  │  Flask App Pods (x2) │ │  ◄── AWS ECR Docker Image
                    │  │  Gunicorn :5000      │ │
                    │  │  /metrics endpoint   │ │
                    │  └──────────┬──────────┘ │
                    │  ┌──────────▼──────────┐  │
                    │  │  LoadBalancer (ELB) │  │
                    │  └──────────┬──────────┘  │
                    └────────────┼──────────────┘
                                 │
               ┌─────────────────┴──────────────────┐
               │                                     │
   ┌───────────▼──────────────┐        ┌─────────────▼────────────┐
   │   Prometheus EC2         │        │   Flask App (ELB)        │
   │   (Ubuntu, t3.medium)    │◄scrape─│   <elb-hostname>:5000    │
   │   Port: 9090             │        │   /metrics               │
   └───────────┬──────────────┘        └──────────────────────────┘
               │
   ┌───────────▼──────────────┐
   │   Grafana EC2            │
   │   (Ubuntu, t3.medium)    │
   │   Port: 3000             │
   │   Dashboards & Alerts    │
   └──────────────────────────┘
```

---

## Tech Stack

| Category | Tools |
|---|---|
| **ML / Data Science** | scikit-learn, pandas, numpy, NLTK, matplotlib |
| **Experiment Tracking** | MLflow 2.15, DagsHub |
| **Data Versioning** | DVC 3.53 + AWS S3 |
| **Web Framework** | Flask 3.0, Gunicorn |
| **Monitoring** | Prometheus v2.46 (`prometheus_client`), Grafana 10.1.5 |
| **Containerization** | Docker (python:3.10-slim) |
| **Image Registry** | AWS ECR |
| **Orchestration** | AWS EKS, eksctl, kubectl |
| **CI/CD** | GitHub Actions |
| **Environment** | Conda (`atlas`, Python 3.10), Cookiecutter Data Science |

---

## Project Structure

```
MLOPs_Prometheus_Grafana_capstone_project/
│
├── .dvc/                        # DVC configuration & cache
├── .github/
│   └── workflows/
│       └── ci.yaml              # GitHub Actions CI/CD pipeline
│
├── flask_app/                   # Flask application
│   ├── app.py                   # Main app with Prometheus /metrics endpoint
│   └── requirements.txt         # Flask-specific deps (generated by pipreqs)
│
├── models/
│   └── vectorizer.pkl           # Serialized text vectorizer
│
├── notebooks/                   # Exploratory analysis & experiment notebooks
│
├── src/                         # Core ML pipeline source code
│   ├── logger/                  # Logging configuration
│   ├── data_ingestion.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_building.py
│   ├── model_evaluation.py
│   └── register_model.py
│
├── scripts/                     # Utility & CI test scripts
├── tests/                       # Unit & integration tests
├── docs/                        # Project documentation
├── reports/                     # Generated metrics & figures
├── references/                  # Reference materials
│
├── Dockerfile                   # Container definition
├── deployment.yaml              # Kubernetes Deployment + Service manifest
├── dvc.yaml                     # DVC pipeline stage definitions
├── dvc.lock                     # Locked pipeline state
├── params.yaml                  # Model hyperparameters
├── requirements.txt             # Full Python dependencies
├── Makefile                     # Common task shortcuts
├── setup.py                     # Package installation
└── tox.ini                      # Test environment configuration
```

---

## Phase 1 — Project Setup

```bash
# 1. Create and activate conda environment
conda create -n atlas python=3.10
conda activate atlas

# 2. Scaffold project with Cookiecutter Data Science template
pip install cookiecutter
cookiecutter -c v1 https://github.com/drivendata/cookiecutter-data-science

# 3. Rename src/models → src/model (project convention)

# 4. Push initial structure to GitHub
git add . && git commit -m "Initial project structure" && git push
```

---

## Phase 2 — MLflow on DagsHub

### Connect DagsHub to Your GitHub Repo

1. Go to [dagshub.com/dashboard](https://dagshub.com/dashboard)
2. Click **Create → New Repo → Connect a repo → GitHub → Connect**
3. Select your repository and click **Connect**
4. Copy the MLflow tracking URL and code snippet shown on the DagsHub repo page

### Generate an Auth Token

Go to your DagsHub repo → **Settings → Tokens → Generate new token**. Save it securely and add it to your GitHub repository secrets.

### Install & Configure

```bash
pip install dagshub mlflow

# Set MLflow tracking environment variables
export MLFLOW_TRACKING_URI=https://dagshub.com/<your_username>/MLOPs_Prometheus_Grafana_capstone_project.mlflow
export MLFLOW_TRACKING_USERNAME=<your_dagshub_username>
export MLFLOW_TRACKING_PASSWORD=<your_dagshub_token>

# Run experiment notebooks — metrics and artifacts will log to DagsHub
```

---

## Phase 3 — DVC Pipeline

### Initialize DVC and Configure Remote Storage

```bash
dvc init

# Temporary local remote for initial testing
dvc remote add -d mylocal local_s3

# Switch to AWS S3 once the bucket is ready
pip install "dvc[s3]" awscli
aws configure              # Enter IAM credentials

dvc remote add -d myremote s3://<your-bucket-name>

# Useful remote management commands
# dvc remote list
# dvc remote remove <name>
```

> **AWS Prerequisite:** Create an IAM user with `AmazonS3FullAccess` permissions and create a dedicated S3 bucket for DVC artifacts.

### DVC Pipeline Stages (`dvc.yaml`)

```
data_ingestion → data_preprocessing → feature_engineering → model_building → model_evaluation
```

Hyperparameters are centrally managed in `params.yaml` and tracked by both DVC and MLflow.

```bash
# Run the full ML pipeline
dvc repro

# Check which stages are out of date
dvc status

# Push all artifacts to S3
dvc push
```

---

## Phase 4 — Flask App & Docker

### Flask App

```bash
# Create flask_app directory and add application files

pip install flask

# Auto-generate flask_app/requirements.txt
pip install pipreqs
cd flask_app && pipreqs . --force
```

### Build & Run Docker Image

```bash
# Build from project root
docker build -t capstone-app:latest .

# Run locally with the required environment variable
docker run -p 8888:5000 \
  -e CAPSTONE_TEST=<your_secret_value> \
  capstone-app:latest

# App: http://localhost:8888
# Metrics: http://localhost:8888/metrics
```

The `Dockerfile` uses `python:3.10-slim`, copies `flask_app/` and `models/vectorizer.pkl`, installs dependencies, downloads NLTK corpora (`stopwords`, `wordnet`), and starts the app with Gunicorn in production mode:

```dockerfile
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]
```

---

## Phase 5 — CI/CD with GitHub Actions

The workflow in `.github/workflows/ci.yaml` triggers on every push to `main` and runs the following stages:

```
push to main
     │
     ▼
Run Tests (pytest / tox)
     │
     ▼
DVC Repro (retrain ML pipeline if params/data changed)
     │
     ▼
Docker Build & Push → AWS ECR
     │
     ▼
kubectl apply -f deployment.yaml → AWS EKS
```

### Required GitHub Secrets

| Secret | Description |
|---|---|
| `AWS_ACCESS_KEY_ID` | IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret key |
| `AWS_REGION` | e.g., `us-east-1` |
| `ECR_REPOSITORY` | ECR repository name (e.g., `capstone-proj`) |
| `AWS_ACCOUNT_ID` | Your 12-digit AWS account ID |
| `CAPSTONE_TEST` | App secret — injected into pods as a Kubernetes Secret |

> **IAM Permission needed:** Ensure the IAM user has `AmazonEC2ContainerRegistryFullAccess`.

---

## Phase 6 — AWS EKS Deployment

### Install Required CLI Tools (Windows PowerShell)

```powershell
# kubectl
Invoke-WebRequest -Uri "https://dl.k8s.io/release/v1.28.2/bin/windows/amd64/kubectl.exe" -OutFile "kubectl.exe"
Move-Item -Path .\kubectl.exe -Destination "C:\Windows\System32"

# eksctl
Invoke-WebRequest -Uri "https://github.com/weaveworks/eksctl/releases/download/v0.158.0/eksctl_Windows_amd64.zip" -OutFile "eksctl.zip"
Expand-Archive -Path .\eksctl.zip -DestinationPath .
Move-Item -Path .\eksctl.exe -Destination "C:\Windows\System32\eksctl.exe"

# Verify all tools
aws --version && kubectl version --client && eksctl version
```

> **Windows Note:** If `aws` resolves to the Anaconda-installed version, uninstall it with `pip uninstall awscli` and ensure `C:\Program Files\Amazon\AWSCLIV2\` is in your system PATH.

### Create the EKS Cluster

```bash
eksctl create cluster \
  --name flask-app-cluster \
  --region us-east-1 \
  --nodegroup-name flask-app-nodes \
  --node-type t3.small \
  --nodes 1 --nodes-min 1 --nodes-max 1 \
  --managed
```

> `eksctl` creates the cluster using **AWS CloudFormation** stacks behind the scenes: one for the EKS control plane and one for the managed node group.

### Configure kubectl & Verify

```bash
# Point kubectl to the new cluster
aws eks --region us-east-1 update-kubeconfig --name flask-app-cluster

# Verify cluster
aws eks list-clusters
kubectl get nodes
kubectl get namespaces
```

### Deploy the Application

```bash
# Deploy via CI/CD (recommended), or manually:
kubectl apply -f deployment.yaml

# Monitor deployment
kubectl get pods
kubectl get svc

# Get the LoadBalancer external hostname
kubectl get svc flask-app-service

# Test the live endpoint
curl http://<external-elb-hostname>:5000
```

**`deployment.yaml` provisions:**
- **2 replicas** of the Flask/Gunicorn pod (image pulled from AWS ECR)
- Resource requests: `256Mi` memory / `250m` CPU; limits: `512Mi` / `1` CPU
- **LoadBalancer** Service on port `5000`
- `CAPSTONE_TEST` secret injected via a Kubernetes Secret (`capstone-secret`)

> **Security Group:** Add an inbound rule for port `5000` on the EKS node security group so the LoadBalancer can reach your pods.

---

## Phase 7 — Prometheus Setup

### Launch EC2 Instance

| Setting | Value |
|---|---|
| Instance type | t3.medium |
| OS | Ubuntu |
| Storage | 20 GB SSD |
| Inbound ports | `9090` (Prometheus UI), `22` (SSH) |

### Install Prometheus

```bash
# SSH into the instance
ssh -i your-key.pem ubuntu@<ec2-public-ip>

sudo apt update && sudo apt upgrade -y

# Download and extract Prometheus v2.46.0
wget https://github.com/prometheus/prometheus/releases/download/v2.46.0/prometheus-2.46.0.linux-amd64.tar.gz
tar -xvzf prometheus-2.46.0.linux-amd64.tar.gz
mv prometheus-2.46.0.linux-amd64 prometheus

# Move to standard paths
sudo mv prometheus /etc/prometheus
sudo mv /etc/prometheus/prometheus /usr/local/bin/
```

### Configure Scrape Target

```bash
sudo nano /etc/prometheus/prometheus.yml
```

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "flask-app"
    static_configs:
      - targets: ["<your-elb-hostname>:5000"]   # EKS LoadBalancer external hostname
```

```bash
# Verify configuration
cat /etc/prometheus/prometheus.yml

# Start Prometheus
/usr/local/bin/prometheus --config.file=/etc/prometheus/prometheus.yml

# Access Prometheus UI at: http://<ec2-public-ip>:9090
```

---

## Phase 8 — Grafana Setup

### Launch EC2 Instance

| Setting | Value |
|---|---|
| Instance type | t3.medium |
| OS | Ubuntu |
| Storage | 20 GB SSD |
| Inbound ports | `3000` (Grafana UI), `22` (SSH) |

### Install & Start Grafana

```bash
# SSH into the instance
ssh -i your-key.pem ubuntu@<ec2-public-ip>

sudo apt update && sudo apt upgrade -y

# Download and install Grafana v10.1.5
wget https://dl.grafana.com/oss/release/grafana_10.1.5_amd64.deb
sudo apt install ./grafana_10.1.5_amd64.deb -y

# Start and enable on boot
sudo systemctl start grafana-server
sudo systemctl enable grafana-server
sudo systemctl status grafana-server

# Access Grafana UI at: http://<ec2-public-ip>:3000
# Default credentials: admin / admin
```

### Connect Prometheus as a Data Source

1. Open Grafana → **Connections → Data Sources → Add new data source**
2. Select **Prometheus**
3. Set URL: `http://<prometheus-ec2-public-ip>:9090`
4. Click **Save & Test**

### Example PromQL Queries

```promql
# Prediction request rate (per second over last 1 minute)
rate(flask_request_count_total[1m])

# 95th percentile response latency
histogram_quantile(0.95, rate(flask_request_latency_seconds_bucket[5m]))
```

---

## AWS Resource Cleanup

Tear down all resources in order to avoid ongoing charges:

```bash
# 1. Delete Kubernetes resources
kubectl delete deployment flask-app
kubectl delete service flask-app-service
kubectl delete secret capstone-secret

# 2. Delete EKS cluster (also removes CloudFormation stacks)
eksctl delete cluster --name flask-app-cluster --region us-east-1

# Verify deletion
eksctl get cluster --region us-east-1

# 3. Manually clean up via AWS Console:
#    - Delete ECR repository and images
#    - Empty and delete the S3 bucket used for DVC
#    - Terminate Prometheus and Grafana EC2 instances
#    - Confirm all CloudFormation stacks are fully deleted
#    - Verify no lingering resources via AWS Support Chat if needed
```

> **CloudFormation Note:** `eksctl create cluster` provisions two stacks: `eksctl-flask-app-cluster-cluster` (control plane) and `eksctl-flask-app-cluster-nodegroup-flask-app-nodes` (worker nodes). Both are automatically deleted when you run `eksctl delete cluster`.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

> Built as an end-to-end MLOps capstone demonstrating: experiment tracking → reproducible ML pipelines → containerized deployment → production monitoring with Prometheus & Grafana.
