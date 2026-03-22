from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal, List, Dict
import joblib
import pandas as pd

app = FastAPI(
    title="Churn Prediction API",
    description="Predicts customer churn using Random Forest trained on IBM Telco data — AUC 0.84",
    version="3.0.0"
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

YES_NO      = {"yes": 1, "no": 0}
GENDER_MAP  = {"male": 1, "female": 0}
INTERNET_MAP= {"fiber optic": 2, "dsl": 1, "no": 0}
CONTRACT_MAP= {"month-to-month": 0, "one year": 1, "two year": 2}
PAYMENT_MAP = {
    "electronic check": 0, "mailed check": 1,
    "bank transfer": 2,    "credit card": 3
}


class CustomerData(BaseModel):
    gender:           Literal["Male","Female","male","female"]
    SeniorCitizen:    Literal["Yes","No","yes","no"]
    Partner:          Literal["Yes","No","yes","no"]
    Dependents:       Literal["Yes","No","yes","no"]
    tenure:           float
    PhoneService:     Literal["Yes","No","yes","no"]
    MultipleLines:    Literal["Yes","No","yes","no"]
    InternetService:  Literal["Fiber optic","DSL","No","fiber optic","dsl","no"]
    OnlineSecurity:   Literal["Yes","No","yes","no"]
    OnlineBackup:     Literal["Yes","No","yes","no"]
    DeviceProtection: Literal["Yes","No","yes","no"]
    TechSupport:      Literal["Yes","No","yes","no"]
    StreamingTV:      Literal["Yes","No","yes","no"]
    StreamingMovies:  Literal["Yes","No","yes","no"]
    Contract:         Literal["Month-to-month","One year","Two year","month-to-month","one year","two year"]
    PaperlessBilling: Literal["Yes","No","yes","no"]
    PaymentMethod:    Literal["Electronic check","Mailed check","Bank transfer","Credit card",
                              "electronic check","mailed check","bank transfer","credit card"]
    MonthlyCharges:   float
    TotalCharges:     float

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Male",
                "SeniorCitizen": "No",
                "Partner": "No",
                "Dependents": "No",
                "tenure": 2,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 85.0,
                "TotalCharges": 170.0
            }
        }


def get_suggestions(customer: CustomerData, prob: float, encoded: dict) -> List[Dict]:
    """Generate ranked improvement suggestions based on customer profile"""
    suggestions = []

    # Contract type — highest impact
    if customer.Contract.lower() == "month-to-month":
        suggestions.append({
            "area": "Contract Type",
            "current": "Month-to-month",
            "recommended": "One year or Two year contract",
            "impact": "High",
            "estimated_churn_reduction": "25-35%",
            "action": "Offer 10% discount for annual contract upgrade"
        })

    # Payment method
    if customer.PaymentMethod.lower() == "electronic check":
        suggestions.append({
            "area": "Payment Method",
            "current": "Electronic check",
            "recommended": "Bank transfer or Credit card (auto-pay)",
            "impact": "Medium",
            "estimated_churn_reduction": "10-15%",
            "action": "Incentivise auto-pay setup with one month free"
        })

    # Online security
    if customer.OnlineSecurity.lower() == "no":
        suggestions.append({
            "area": "Online Security",
            "current": "Not subscribed",
            "recommended": "Add Online Security package",
            "impact": "Medium",
            "estimated_churn_reduction": "8-12%",
            "action": "Offer 3-month free trial of security package"
        })

    # Tech support
    if customer.TechSupport.lower() == "no":
        suggestions.append({
            "area": "Tech Support",
            "current": "Not subscribed",
            "recommended": "Add Tech Support package",
            "impact": "Medium",
            "estimated_churn_reduction": "8-12%",
            "action": "Bundle tech support with next billing cycle at no cost"
        })

    # Tenure — new customer
    if customer.tenure <= 6:
        suggestions.append({
            "area": "Customer Tenure",
            "current": f"{int(customer.tenure)} months (new customer)",
            "recommended": "Engage with loyalty programme",
            "impact": "High",
            "estimated_churn_reduction": "15-20%",
            "action": "Assign dedicated onboarding support for first 3 months"
        })

    # High charges
    if customer.MonthlyCharges > 75:
        suggestions.append({
            "area": "Monthly Charges",
            "current": f"${customer.MonthlyCharges} (high spender)",
            "recommended": "Review plan value perception",
            "impact": "Medium",
            "estimated_churn_reduction": "5-10%",
            "action": "Proactively offer bundle discount or loyalty credit"
        })

    # Online backup
    if customer.OnlineBackup.lower() == "no":
        suggestions.append({
            "area": "Online Backup",
            "current": "Not subscribed",
            "recommended": "Add Online Backup",
            "impact": "Low",
            "estimated_churn_reduction": "3-5%",
            "action": "Include backup in next promotional offer"
        })

    # If low risk — nothing critical
    if not suggestions:
        suggestions.append({
            "area": "Overall",
            "current": "Good profile",
            "recommended": "Maintain current engagement",
            "impact": "Low",
            "estimated_churn_reduction": "0%",
            "action": "Send loyalty reward to reinforce satisfaction"
        })

    return suggestions


def encode(customer: CustomerData) -> tuple:
    g  = customer.gender.lower()
    c  = customer.Contract.lower()
    pm = customer.PaymentMethod.lower()
    i  = customer.InternetService.lower()

    gender           = GENDER_MAP[g]
    SeniorCitizen    = YES_NO[customer.SeniorCitizen.lower()]
    Partner          = YES_NO[customer.Partner.lower()]
    Dependents       = YES_NO[customer.Dependents.lower()]
    tenure           = customer.tenure
    PhoneService     = YES_NO[customer.PhoneService.lower()]
    MultipleLines    = YES_NO[customer.MultipleLines.lower()]
    InternetService  = INTERNET_MAP[i]
    OnlineSecurity   = YES_NO[customer.OnlineSecurity.lower()]
    OnlineBackup     = YES_NO[customer.OnlineBackup.lower()]
    DeviceProtection = YES_NO[customer.DeviceProtection.lower()]
    TechSupport      = YES_NO[customer.TechSupport.lower()]
    StreamingTV      = YES_NO[customer.StreamingTV.lower()]
    StreamingMovies  = YES_NO[customer.StreamingMovies.lower()]
    Contract         = CONTRACT_MAP[c]
    PaperlessBilling = YES_NO[customer.PaperlessBilling.lower()]
    PaymentMethod    = PAYMENT_MAP[pm]
    MonthlyCharges   = customer.MonthlyCharges
    TotalCharges     = customer.TotalCharges

    AvgMonthlySpend  = round(TotalCharges / (tenure + 1), 2)
    IsNewCustomer    = 1 if tenure <= 6 else 0
    HighSpender      = 1 if MonthlyCharges > 75 else 0
    HasSupport       = 1 if (OnlineSecurity == 1 or TechSupport == 1) else 0
    ChargesPerTenure = round(MonthlyCharges / (tenure + 1), 2)
    RiskScore        = (
        (1 if Contract == 0 else 0) +
        (1 if PaymentMethod == 0 else 0) +
        (1 if OnlineSecurity == 0 else 0) +
        (1 if TechSupport == 0 else 0) +
        (1 if tenure <= 12 else 0)
    )

    encoded = {
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
    return pd.DataFrame([encoded])[FEATURE_NAMES], encoded


@app.get("/")
def root():
    return {
        "message": "Churn Prediction API is live",
        "model": "Random Forest — AUC 0.84 on IBM Telco (7,043 records)",
        "docs": "/docs",
        "version": "3.0 — human inputs + improvement suggestions"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(customer: CustomerData):
    try:
        input_df, encoded = encode(customer)
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

        suggestions = get_suggestions(customer, prob, encoded)

        return {
            "churn_probability": prob,
            "risk_level": risk,
            "recommendation": recommendation,
            "risk_score": encoded['RiskScore'],
            "improvement_suggestions": suggestions,
            "summary": f"{len(suggestions)} area(s) identified for retention improvement"
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
