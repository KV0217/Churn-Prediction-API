# Churn Prediction API

Production REST API serving a Random Forest churn model — AUC 0.84 on IBM Telco data.
Built with FastAPI, containerised with Docker, deployed on Render.

## Live
| | URL |
|--|--|
| API | https://churn-prediction-api-cbqf.onrender.com |
| Docs | https://churn-prediction-api-cbqf.onrender.com/docs |
| Health | https://churn-prediction-api-cbqf.onrender.com/health |
| Streamlit | https://customer-churn-prediction-kv.streamlit.app |


## Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | Model info |
| GET | /health | Health check |
| POST | /predict | Single customer prediction |
| POST | /predict/batch | Batch predictions |

## Features
- Human-readable inputs — send "Yes/No", "Month-to-month" instead of 0/1
- Auto-calculates derived features (RiskScore, AvgMonthlySpend etc.)
- Per-customer improvement suggestions with estimated churn reduction %
- Batch endpoint for multiple customers

## Run Locally
```bash
git clone https://github.com/KV0217/Churn-Prediction-API.git
cd Churn-Prediction-API
pip install -r requirements.txt
uvicorn main:app --reload
```

## Tech Stack
FastAPI · Uvicorn · Scikit-learn · XGBoost · Docker · Render

## Related
- Analysis notebook: [Customer-Churn-Prediction](https://github.com/KV0217/customer-churn-prediction)
