from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import json
import pandas as pd

app = FastAPI(
    title="Churn Prediction API",
    description="Predicts customer churn using Random Forest trained on IBM Telco data — AUC 0.84",
    version="1.0.0"
)

model = joblib.load("churn_model.joblib")

FEATURE_NAMES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges', 'AvgMonthlySpend', 'IsNewCustomer',
    'HighSpender', 'HasSupport', 'ChargesPerTenure', 'RiskScore'
]


class CustomerData(BaseModel):
    gender: int              # 0 = Female, 1 = Male
    SeniorCitizen: int       # 0 or 1
    Partner: int             # 0 or 1
    Dependents: int          # 0 or 1
    tenure: float            # months as customer
    PhoneService: int        # 0 or 1
    MultipleLines: int       # 0 or 1
    InternetService: int     # label encoded
    OnlineSecurity: int      # 0 or 1
    OnlineBackup: int        # 0 or 1
    DeviceProtection: int    # 0 or 1
    TechSupport: int         # 0 or 1
    StreamingTV: int         # 0 or 1
    StreamingMovies: int     # 0 or 1
    Contract: int            # label encoded
    PaperlessBilling: int    # 0 or 1
    PaymentMethod: int       # label encoded
    MonthlyCharges: float
    TotalCharges: float
    AvgMonthlySpend: float   # TotalCharges / (tenure + 1)
    IsNewCustomer: int       # 1 if tenure <= 6
    HighSpender: int         # 1 if MonthlyCharges > 75
    HasSupport: int          # 1 if OnlineSecurity or TechSupport == 1
    ChargesPerTenure: float  # MonthlyCharges / (tenure + 1)
    RiskScore: int           # 0-5 composite risk score

    class Config:
        json_schema_extra = {
            "example": {
                "gender": 1,
                "SeniorCitizen": 0,
                "Partner": 1,
                "Dependents": 0,
                "tenure": 12,
                "PhoneService": 1,
                "MultipleLines": 0,
                "InternetService": 1,
                "OnlineSecurity": 0,
                "OnlineBackup": 1,
                "DeviceProtection": 0,
                "TechSupport": 0,
                "StreamingTV": 1,
                "StreamingMovies": 0,
                "Contract": 0,
                "PaperlessBilling": 1,
                "PaymentMethod": 2,
                "MonthlyCharges": 65.5,
                "TotalCharges": 786.0,
                "AvgMonthlySpend": 61.25,
                "IsNewCustomer": 0,
                "HighSpender": 0,
                "HasSupport": 0,
                "ChargesPerTenure": 5.04,
                "RiskScore": 3
            }
        }


@app.get("/")
def root():
    return {
        "message": "Churn Prediction API is live",
        "model": "Random Forest — AUC 0.84 on IBM Telco (7,043 records)",
        "docs": "/docs",
        "features": FEATURE_NAMES
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(customer: CustomerData):
    try:
        input_df = pd.DataFrame([customer.dict()])[FEATURE_NAMES]
        prob = model.predict_proba(input_df)[0][1]
        prob = round(float(prob), 3)

        if prob >= 0.7:
            risk = "high"
            recommendation = "Immediate outreach — offer contract upgrade or discount"
        elif prob >= 0.4:
            risk = "medium"
            recommendation = "Monitor closely — send loyalty reward within 7 days"
        else:
            risk = "low"
            recommendation = "No immediate action needed"

        return {
            "churn_probability": prob,
            "risk_level": risk,
            "recommendation": recommendation
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/batch")
def predict_batch(customers: list[CustomerData]):
    results = []
    for customer in customers:
        result = predict(customer)
        results.append(result)
    return {"predictions": results, "count": len(results)}