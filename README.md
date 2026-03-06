# 🧠 NeuroMetabolic Dashboard (NMD)

> AI-driven decision-support system for Type 1 Diabetes management — predicting glycemic trends using Temporal Fusion Transformers and closed-loop pump data.

[![CI](https://github.com/Radian20Hz/NeuroMetabolic-Dashboard/actions/workflows/ci.yml/badge.svg)](https://github.com/Radian20Hz/NeuroMetabolic-Dashboard/actions)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active%20Development-orange)](https://github.com/Radian20Hz/NeuroMetabolic-Dashboard)

> 🚧 **Active Development** — Phase 1 in progress (Q2 2026). This repository documents a 24-month engineering journey toward a production-ready clinical decision-support tool.

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
| **Database** | InfluxDB (time-series) |
| **ML Framework** | PyTorch, Temporal Fusion Transformer |
| **Inference** | ONNX Runtime (Edge AI) |
| **Frontend** | React.js, Tailwind CSS, Recharts |
| **MLOps** | MLflow, DVC |
| **CI/CD** | GitHub Actions, Flake8, MyPy, PyTest |
| **Security** | OAuth2 + JWT, AES-256 |

---

## 📁 Project Structure

```
neurometabolic-dashboard/
├── backend/                  # FastAPI application
│   ├── app/
│   │   ├── api/              # Route handlers
│   │   ├── core/             # Config, security, dependencies
│   │   ├── models/           # Pydantic schemas & DB models
│   │   ├── services/         # Business logic
│   │   └── utils/            # Helpers
│   └── tests/
│       ├── unit/
│       └── integration/
├── frontend/                 # React dashboard UI
│   └── src/
│       ├── components/
│       ├── pages/
│       ├── hooks/
│       └── utils/
├── ml/                       # ML pipeline
│   ├── data/
│   │   ├── raw/              # OhioT1DM dataset (gitignored)
│   │   └── processed/
│   ├── models/               # Saved model weights (gitignored)
│   ├── notebooks/            # Exploratory analysis
│   └── scripts/              # Training & evaluation scripts
├── infra/
│   ├── docker/               # Docker configs
│   └── github-actions/
└── .github/
    └── workflows/            # CI/CD pipelines
```

---

## 🚀 Roadmap

| Phase | Timeline | Focus |
|---|---|---|
| **Phase 1** | Q2–Q3 2026 | ETL pipeline + OhioT1DM data aggregation |
| **Phase 2** | Q4 2026 | TFT architecture + MARD validation |
| **Phase 3** | Q1 2027 | ONNX conversion + "What-If" simulator UI |
| **Phase 4** | Q2 2027 | Bilingual documentation (EN/JP) |

---

## ⚙️ Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker (recommended)
- InfluxDB 2.x

### Installation

```bash
# Clone repository
git clone https://github.com/Radian20Hz/NeuroMetabolic-Dashboard.git
cd neurometabolic-dashboard

# Backend setup
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Frontend setup
cd ../frontend
npm install
```

### Running locally

```bash
# Backend
cd backend && uvicorn app.main:app --reload

# Frontend
cd frontend && npm run dev
```

---

## 🔒 Data Privacy & Compliance

- **GDPR/RODO compliant**: Full de-identification at the extraction layer
- **OhioT1DM Dataset**: Fully de-identified academic benchmark dataset
- **AES-256** encryption for data at rest
- **OAuth2 + JWT** for stateless authentication

---

## 📊 Model Validation

Target metrics:
- **MARD** (Mean Absolute Relative Difference): < 10%
- **Clarke Error Grid**: > 95% in zones A+B

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built with ❤️ for the T1D community.*
