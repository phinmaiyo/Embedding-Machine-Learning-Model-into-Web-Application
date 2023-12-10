# Building Machine Learning APIs With FastAPI
# Imports
from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import pandas as pd

# Loading Models
pipeline = joblib.load('./src/xgb_pipeline.pkl')
encoder = joblib.load('./src/encoder.pkl')

# FastAPI Application Setup
app = FastAPI(
    title="Sepsis Analysis API"
)

# Data Model (Pydantic BaseModel)
class SepsisFeatures(BaseModel):
    PRG: int
    PL: int
    PR:int
    SK :int
    TS :int
    M11:float
    BD2:float
    Age:int
    Insurance:int

# Endpoints
@app.get('/')
def home():
   return "Sepsis Anaysis"

@app.get('/info')
def appinfo():
    return 'Sepsis Analysis API: This is my interface'
 
# Prediction Endpoint (/predict_sepsis) 
@app.post('/predict_sepsis') 
def predict_sepsis(sepsis_features: SepsisFeatures):
    # Convert input features to a DataFrame
    df = pd.DataFrame([sepsis_features.model_dump()])

    # Perform prediction using the pre-trained pipeline
    prediction = pipeline.predict(df)

    # Inversely transform the predicted value using the loaded encoder to get human-readable output
    encoder_prediction = encoder.inverse_transform([prediction])[0]
    
    # Return a dictionary with the predicted value
    return {'prediction' : encoder_prediction}






    
