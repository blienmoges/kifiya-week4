# src/api/pydantic_models.py
from pydantic import BaseModel
from typing import Optional

class CustomerData(BaseModel):
    CustomerId: str
    Amount: float
    Value: float
    CurrencyCode: str
    CountryCode: str
    ProviderId: str
    ProductCategory: str
    ChannelId: str
    PricingStrategy: str

class PredictionResponse(BaseModel):
    CustomerId: str
    RiskProbability: float
