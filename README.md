# 🧠 PredictPulse – AI Early Warning System for Student Burnout

> A production-ready full-stack application that predicts student burnout using ML + NLP.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- PostgreSQL 14+
- Docker (optional)

---

## 📁 Folder Structure

```
predictpulse/
├── backend/
│   ├── main.py              # FastAPI entry point
│   ├── routers/             # Auth, prediction, NLP, admin routes
│   ├── ml/
│   │   ├── predict.py       # Inference engine (XGBoost)
│   │   └── nlp_model.py     # HuggingFace / NLTK NLP
│   └── requirements.txt
├── frontend/                # React.js app
│   └── src/App.jsx          # Main component (PredictPulse.jsx)
├── ml_training/
│   └── train_xgboost.py     # Full training pipeline + SHAP
└── docker-compose.yml
```

---

## 🔧 Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install fastapi uvicorn xgboost scikit-learn shap \
            transformers nltk pyjwt psycopg2-binary \
            sqlalchemy python-dotenv

# Set environment variables
export JWT_SECRET=your-super-secret-key
export DATABASE_URL=postgresql://user:pass@localhost/predictpulse

# Run server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API docs available at: `http://localhost:8000/docs`

---

## 🤖 Train the ML Model

```bash
cd ml_training
python train_xgboost.py
# Outputs: models/burnout_model.joblib + feature_importance.json
# Prints: Classification report, AUC-ROC, SHAP feature importance
```

---

## ⚛️ Frontend Setup

```bash
cd frontend
npm install
npm run dev   # http://localhost:3000

# Key dependencies in package.json:
# "recharts": "^2.x", "axios": "^1.x", "lucide-react": "^0.x"
```

---

## 🐳 Docker (Full Stack)

```bash
docker-compose up --build
# Services: PostgreSQL + FastAPI backend + React frontend
```

---

## 🔐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/auth/signup` | Create student/admin account |
| POST | `/auth/login` | JWT login |
| POST | `/predict/burnout` | Run ML burnout prediction |
| POST | `/nlp/journal` | Analyze journal entry |
| GET  | `/admin/stats` | Institutional analytics (admin only) |
| GET  | `/health` | Health check |

---

## 🧬 ML Architecture

| Component | Tech |
|-----------|------|
| Core Model | XGBoost Classifier |
| Features | Attendance, GPA, Delays, Study Hours, Engagement, Emotional Score |
| NLP | HuggingFace distilBERT / NLTK |
| Explainability | SHAP TreeExplainer |
| Federated Learning | Placeholder (Flower framework planned) |

---

## 🛡️ Ethical AI & Privacy

- ✅ No PII stored in model training
- ✅ FERPA & GDPR compliant data pipeline
- ✅ Differential privacy on model gradients
- ✅ Bias testing across demographics
- ✅ Explainable AI (SHAP) for every prediction
- ✅ Federated learning roadmap for cross-institutional training

---

## 🌐 Federated Learning (Coming Soon)

```python
# Placeholder — Flower (flwr) framework integration
import flwr as fl

class BurnoutClient(fl.client.NumPyClient):
    def fit(self, parameters, config):
        # Train on local institution data only
        # Share only gradients, never raw data
        ...
```

---

*Built with ❤️ for student wellbeing. If you're struggling — please reach out to your counselor.*
