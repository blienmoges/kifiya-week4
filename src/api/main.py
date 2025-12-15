# src/api/main.py
from fastapi import FastAPI
from src.api.pydantic_models import CustomerData, PredictionResponse
import pandas as pd
import mlflow
import mlflow.sklearn

app = FastAPI(title="Customer Risk Prediction API")

# Load best model from MLflow registry
model_uri = "models:/YourBestModelName/Production"  # Replace with your MLflow model name
model = mlflow.sklearn.load_model(model_uri)

@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData):
    # Convert input to DataFrame
    input_df = pd.DataFrame([customer.dict()])
    
    # Predict probability for the positive class
    probability = model.predict_proba(input_df)[:, 1][0]
    
    return PredictionResponse(CustomerId=customer.CustomerId, RiskProbability=probability)
