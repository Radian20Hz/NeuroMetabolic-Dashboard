# 🧠 NeuroMetabolic Dashboard (NMD)

> AI-driven decision-support system for Type 1 Diabetes management — predicting glycemic trends using Temporal Fusion Transformers and closed-loop pump data.

[![CI](https://github.com/Radian20Hz/NeuroMetabolic-Dashboard/actions/workflows/ci.yml/badge.svg)](https://github.com/Radian20Hz/NeuroMetabolic-Dashboard/actions)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active%20Development-orange)](https://github.com/Radian20Hz/NeuroMetabolic-Dashboard)

> 🚧 **Active Development** — Phase 2 in progress (Q2 2026). This repository documents a 24-month engineering journey toward a production-ready clinical decision-support tool.

---

## 💡 Motivation

This project is built by a T1D patient, not just *for* T1D patients.

The developer wears a **Medtronic 780G closed-loop insulin pump** every day — which means this is not an academic exercise. The data this system processes is the same data that determines insulin delivery in real life. That personal stake is what drives the engineering decisions: precision over convenience, explainability over black-box accuracy, and patient safety above all else.

---

## 📌 Overview

The **NeuroMetabolic Dashboard** integrates real-time CGM data from the **Medtronic 780G** closed-loop system with a **Temporal Fusion Transformer (TFT)** model to provide:

- 📈 High-precision **blood glucose predictions** (up to 60-min horizon)
- 🔬 **"What-If" metabolic simulator** — visualize the impact of meals and exercise *before* they happen
- 🚨 **Proactive alert system** — 20-minute lead time before predicted hypoglycemia
- 🧩 **Explainable AI (XAI)** — TFT attention maps for variable importance visualization

> ⚠️ **Medical Disclaimer:** NMD is a decision-support tool only. It is not a replacement for professional medical advice or automated insulin delivery (AID) logic.

---

## 🏗️ Architecture

```
graph TD
  A[Medtronic 780G Pump] -->|Bluetooth| B[MiniMed Mobile App]
  B -->|HTTPS| C[CareLink Cloud]
  C -->|API Scraper / CSV Fallback| D[Data Ingestion Service]
  D --> E[(InfluxDB)]
  E --> F[TFT Model - PyTorch/MLflow]
  F --> G[ONNX Runtime - Edge AI]
  H[User Inputs: Meals/Activity] --> G
  G --> I[React Dashboard UI]
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3.11, FastAPI |
| **Database** | InfluxDB 2.7 (time-series) |
| **ML Framework** | PyTorch, Temporal Fusion Transformer |
| **Inference** | ONNX Runtime (Edge AI) |
| **Frontend** | React.js, Tailwind CSS, Recharts |
| **MLOps** | MLflow, DVC |
| **CI/CD** | GitHub Actions, Flake8, MyPy, PyTest |
| **Containerization** | Docker, Docker Compose |
| **Security** | OAuth2 + JWT, AES-256 |

---

## 📁 Project Structure

```
neurometabolic-dashboard/
├── backend/                  # FastAPI application
│   ├── app/
│   │   ├── api/              # Route handlers
│   │   ├── core/             # Config, security, dependencies
│   │   ├── models/           # Pydantic schemas
│   │   ├── services/         # Business logic
│   │   └── utils/            # Helpers
│   └── tests/
│       ├── unit/
│       └── integration/
├── frontend/                 # React dashboard UI (Phase 3)
│   └── src/
│       ├── components/
│       ├── pages/
│       ├── hooks/
│       └── utils/
├── ml/                       # ML pipeline (Phase 2)
│   ├── data/
│   │   ├── raw/              # OhioT1DM dataset (gitignored)
│   │   └── processed/
│   ├── models/               # Saved model weights (gitignored)
│   ├── notebooks/            # Exploratory analysis
│   └── scripts/              # Training & evaluation scripts
├── docker-compose.yml        # InfluxDB local setup
└── .github/
    └── workflows/            # CI/CD pipelines
```

---

## ✅ Progress

### Phase 1 — ETL Pipeline (complete)

- [x] CareLink CSV parser — handles real Medtronic 780G export format
- [x] InfluxDB service — write and query glucose time-series data
- [x] REST API — `/upload`, `/latest` endpoints
- [x] Pydantic response models
- [x] GitHub Actions CI/CD pipeline (flake8 + mypy + pytest)
- [x] 30 unit tests — all passing

### Phase 2 — Clinical Intelligence Layer (in progress)

- [x] Glucose Validator — ADA 2024 clinical zone classification
- [x] Time-in-Range calculator
- [x] Glycemic statistics engine (min/max/avg/std_dev/TIR)
- [x] `/classify` endpoint — single reading classification
- [x] `/statistics` endpoint — full CSV statistical analysis
- [x] Docker Compose + InfluxDB — end-to-end pipeline working with real pump data
- [ ] CareLink API scraper — automated data ingestion
- [ ] Stats enrichment on upload response

### Phase 3 — TFT Model (planned Q4 2026)

- [ ] OhioT1DM dataset preprocessing
- [ ] TFT architecture implementation
- [ ] ONNX conversion for edge inference
- [ ] MARD validation < 10%

### Phase 4 — Frontend & Production (planned Q1 2027)

- [ ] React dashboard with CGM chart
- [ ] "What-If" metabolic simulator
- [ ] Proactive hypoglycemia alert system
- [ ] Bilingual documentation (EN/JP)

---

## 🚀 Roadmap

| Phase | Timeline | Focus | Status |
|---|---|---|---|
| **Phase 1** | Q1–Q2 2026 | ETL pipeline + REST API | ✅ Complete |
| **Phase 2** | Q2–Q3 2026 | Clinical intelligence layer | 🔄 In Progress |
| **Phase 3** | Q4 2026 | TFT model + ONNX inference | 📅 Planned |
| **Phase 4** | Q1 2027 | Frontend + production hardening | 📅 Planned |

---

## ⚙️ Getting Started

### Prerequisites

- Python 3.11+
- Docker + Docker Compose
- Node.js 18+ (Phase 4)

### Installation

```bash
# Clone repository
git clone https://github.com/Radian20Hz/NeuroMetabolic-Dashboard.git
cd neurometabolic-dashboard

# Start InfluxDB
docker compose up -d

# Backend setup
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Create .env file
cp .env.example .env  # fill in your InfluxDB credentials
```

### Running locally

```bash
# Backend
cd backend && uvicorn app.main:app --reload
# API docs available at http://localhost:8000/docs

# InfluxDB UI
# Available at http://localhost:8086
```

### Running tests

```bash
cd backend
pytest tests/unit/ -v
```

---

## 🔒 Data Privacy & Compliance

- **GDPR/RODO compliant**: Full de-identification at the extraction layer
- **OhioT1DM Dataset**: Fully de-identified academic benchmark dataset
- **AES-256** encryption for data at rest
- **OAuth2 + JWT** for stateless authentication

---

## 📊 Model Validation Targets

| Metric | Target |
|---|---|
| **MARD** (Mean Absolute Relative Difference) | < 10% |
| **Clarke Error Grid** zones A+B | > 95% |
| **Time-in-Range** prediction accuracy | > 90% |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built with ❤️ for the T1D community.*
