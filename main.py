from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from typing import Literal
import joblib
import pandas as pd

app = FastAPI(
    title="Churn Prediction API",
    description="Predicts customer churn using Random Forest trained on IBM Telco data — AUC 0.84",
    version="2.0.0"
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

# --- Encoding maps ---
YES_NO = {"yes": 1, "no": 0}
GENDER_MAP = {"male": 1, "female": 0}
INTERNET_MAP = {"fiber optic": 2, "dsl": 1, "no": 0}
CONTRACT_MAP = {"month-to-month": 0, "one year": 1, "two year": 2}
PAYMENT_MAP = {
    "electronic check": 0,
    "mailed check": 1,
    "bank transfer": 2,
    "credit card": 3
}


class CustomerData(BaseModel):
    gender:             Literal["Male", "Female", "male", "female"]
    SeniorCitizen:      Literal["Yes", "No", "yes", "no"]
    Partner:            Literal["Yes", "No", "yes", "no"]
    Dependents:         Literal["Yes", "No", "yes", "no"]
    tenure:             float
    PhoneService:       Literal["Yes", "No", "yes", "no"]
    MultipleLines:      Literal["Yes", "No", "yes", "no"]
    InternetService:    Literal["Fiber optic", "DSL", "No", "fiber optic", "dsl", "no"]
    OnlineSecurity:     Literal["Yes", "No", "yes", "no"]
    OnlineBackup:       Literal["Yes", "No", "yes", "no"]
    DeviceProtection:   Literal["Yes", "No", "yes", "no"]
    TechSupport:        Literal["Yes", "No", "yes", "no"]
    StreamingTV:        Literal["Yes", "No", "yes", "no"]
    StreamingMovies:    Literal["Yes", "No", "yes", "no"]
    Contract:           Literal["Month-to-month", "One year", "Two year", "month-to-month", "one year", "two year"]
    PaperlessBilling:   Literal["Yes", "No", "yes", "no"]
    PaymentMethod:      Literal["Electronic check", "Mailed check", "Bank transfer", "Credit card",
                                "electronic check", "mailed check", "bank transfer", "credit card"]
    MonthlyCharges:     float
    TotalCharges:       float

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Male",
                "SeniorCitizen": "No",
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 65.5,
                "TotalCharges": 786.0
            }
        }


def encode(customer: CustomerData) -> pd.DataFrame:
    """Convert human-readable inputs to model-ready numbers"""
    g = customer.gender.lower()
    c = customer.Contract.lower()
    pm = customer.PaymentMethod.lower()
    i = customer.InternetService.lower()

    gender          = GENDER_MAP[g]
    SeniorCitizen   = YES_NO[customer.SeniorCitizen.lower()]
    Partner         = YES_NO[customer.Partner.lower()]
    Dependents      = YES_NO[customer.Dependents.lower()]
    tenure          = customer.tenure
    PhoneService    = YES_NO[customer.PhoneService.lower()]
    MultipleLines   = YES_NO[customer.MultipleLines.lower()]
    InternetService = INTERNET_MAP[i]
    OnlineSecurity  = YES_NO[customer.OnlineSecurity.lower()]
    OnlineBackup    = YES_NO[customer.OnlineBackup.lower()]
    DeviceProtection= YES_NO[customer.DeviceProtection.lower()]
    TechSupport     = YES_NO[customer.TechSupport.lower()]
    StreamingTV     = YES_NO[customer.StreamingTV.lower()]
    StreamingMovies = YES_NO[customer.StreamingMovies.lower()]
    Contract        = CONTRACT_MAP[c]
    PaperlessBilling= YES_NO[customer.PaperlessBilling.lower()]
    PaymentMethod   = PAYMENT_MAP[pm]
    MonthlyCharges  = customer.MonthlyCharges
    TotalCharges    = customer.TotalCharges

    # Derived features (same as your notebook)
    AvgMonthlySpend  = round(TotalCharges / (tenure + 1), 2)
    IsNewCustomer    = 1 if tenure <= 6 else 0
    HighSpender      = 1 if MonthlyCharges > 75 else 0
    HasSupport       = 1 if (OnlineSecurity == 1 or TechSupport == 1) else 0
    ChargesPerTenure = round(MonthlyCharges / (tenure + 1), 2)
    RiskScore        = (
        (1 if Contract == 0 else 0) +          # month-to-month
        (1 if PaymentMethod == 0 else 0) +      # electronic check
        (1 if OnlineSecurity == 0 else 0) +
        (1 if TechSupport == 0 else 0) +
        (1 if tenure <= 12 else 0)
    )

    row = {
        'gender': gender, 'SeniorCitizen': SeniorCitizen,
        'Partner': Partner, 'Dependents': Dependents,
        'tenure': tenure, 'PhoneService': PhoneService,
        'MultipleLines': MultipleLines, 'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity, 'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection, 'TechSupport': TechSupport,
        'StreamingTV': StreamingTV, 'StreamingMovies': StreamingMovies,
        'Contract': Contract, 'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod, 'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges, 'AvgMonthlySpend': AvgMonthlySpend,
        'IsNewCustomer': IsNewCustomer, 'HighSpender': HighSpender,
        'HasSupport': HasSupport, 'ChargesPerTenure': ChargesPerTenure,
        'RiskScore': RiskScore
    }
    return pd.DataFrame([row])[FEATURE_NAMES]


@app.get("/")
def root():
    return {
        "message": "Churn Prediction API is live",
        "model": "Random Forest — AUC 0.84 on IBM Telco (7,043 records)",
        "docs": "/docs",
        "version": "2.0 — human-readable inputs"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(customer: CustomerData):
    try:
        input_df = encode(customer)
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