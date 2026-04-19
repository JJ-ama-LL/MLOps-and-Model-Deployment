from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time
import joblib
from pydantic import BaseModel, Field
from typing import List
import pandas as pd
import numpy as np
import uuid

ml_model = None
preprocessor = None
start_time = None

MODEL_NAME = "taxi-tip-predictor"
MODEL_VERSION = "1"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ml_model, preprocessor, start_time
    # STARTUP: Load Model once
    ml_model = joblib.load("models/registered_model.pkl")
    preprocessor = joblib.load("models/preprocessor.pkl")
    start_time = time.time()
    print("Model and Preprocessor Loaded Successfully!")
    yield
    print("Shutting down...")

app = FastAPI(title = "NYC Taxi Trip Predictor", version = "0.1.0", lifespan=lifespan)

@app.get("/")
def root():
    return {"message": "Welcome to the NYC Taxi Trip Predictor API!"}

class TripInput(BaseModel):
    trip_distance: float = Field(..., gt=0, description="Trip distance in miles")
    fare_amount: float = Field(..., gt=0, le=500, description="Fare in dollars")
    pickup_hour: int = Field(..., ge=0, le=23, description="Pickup hour (0-23)")
    passenger_count: int = Field(..., ge=1, le=9, description="Number of passengers")
    trip_duration_minutes: float = Field(..., gt=0, description="Trip duration in minutes")
    total_amount: float = Field(..., gt=0, description="Total amount charged")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "trip_distance": 3.5,
                "fare_amount": 14.5,
                "pickup_hour": 14,
                "passenger_count": 1,
                "trip_duration_minutes": 12.0,
                "total_amount": 18.8
            }]
        }
    }

class PredictionResponse(BaseModel):
    predicted_tip_amount: float
    model_version: str
    prediction_id: str

class BatchInput(BaseModel):
    records: List[TripInput] = Field(..., max_length=100)

class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]
    count: int
    processing_time_ms: float

def prepare_features(trip: TripInput) -> pd.DataFrame:
    log_trip_distance = np.log1p(trip.trip_distance)
    trip_speed_mph = trip.trip_distance / max(trip.trip_duration_minutes / 60, 0.001)
    fare_per_mile = trip.fare_amount / trip.trip_distance
    fare_per_minute = trip.fare_amount / max(trip.trip_duration_minutes, 0.001)
    is_weekend = 0
    pickup_day_of_week = 0

    row = {
        "VendorID": 1,
        "passenger_count": trip.passenger_count,
        "trip_distance": trip.trip_distance,
        "RatecodeID": 1,
        "PULocationID": 161,
        "DOLocationID": 236,
        "payment_type": 1,
        "fare_amount": trip.fare_amount,
        "extra": 0.0,
        "mta_tax": 0.5,
        "tolls_amount": 0.0,
        "improvement_surcharge": 0.3,
        "total_amount": trip.total_amount,
        "congestion_surcharge": 0.0,
        "Airport_fee": 0.0,
        "pickup_hour": trip.pickup_hour,
        "pickup_day_of_week": pickup_day_of_week,
        "is_weekend": is_weekend,
        "trip_duration_minutes": trip.trip_duration_minutes,
        "log_trip_distance": log_trip_distance,
        "trip_speed_mph": trip_speed_mph,
        "fare_per_mile": fare_per_mile,
        "fare_per_minute": fare_per_minute,
        "pickup_borough_Bronx": 0.0, "pickup_borough_Brooklyn": 0.0,
        "pickup_borough_EWR": 0.0, "pickup_borough_Manhattan": 0.0,
        "pickup_borough_N/A": 0.0, "pickup_borough_Queens": 0.0,
        "pickup_borough_Staten Island": 0.0, "pickup_borough_Unknown": 1.0,
        "dropoff_borough_Bronx": 0.0, "dropoff_borough_Brooklyn": 0.0,
        "dropoff_borough_EWR": 0.0, "dropoff_borough_Manhattan": 0.0,
        "dropoff_borough_N/A": 0.0, "dropoff_borough_Queens": 0.0,
        "dropoff_borough_Staten Island": 0.0, "dropoff_borough_Unknown": 1.0,
        "store_and_fwd_flag": "N",
    }
    return pd.DataFrame([row])

@app.post("/predict", response_model=PredictionResponse)
def predict(trip: TripInput):
    features = prepare_features(trip)
    processed = preprocessor.transform(features)
    prediction = ml_model.predict(processed)[0]
    return PredictionResponse(
        predicted_tip_amount=round(float(prediction), 2),
        model_version=MODEL_VERSION, 
        prediction_id=str(uuid.uuid4())
    )

@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchInput):
    start = time.time()
    predictions = []
    for trip in batch.records:
        features = prepare_features(trip)
        processed = preprocessor.transform(features)
        pred = ml_model.predict(processed)[0]
        predictions.append(PredictionResponse(
            predicted_tip_amount=round(float(pred), 2),
            model_version=MODEL_VERSION, 
            prediction_id=str(uuid.uuid4())
        ))

    elapsed = (time.time() - start) * 1000
    return BatchResponse(
        predictions=predictions,
        count=len(predictions),
        processing_time_ms=round(elapsed, 2)
    )

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": ml_model is not None,
        "model_version": MODEL_VERSION,
        "uptime_seconds": round(time.time() - start_time, 1)
    }

@app.get("/model/info")
def model_info():
    return {        
        "model_name": MODEL_NAME,
        "version": MODEL_VERSION,
        "features": ["trip_distance", "fare_amount", "pickup_hour",
                    "passenger_count", "trip_duration_minutes",
                    "total_amount"],
        "metrics": {
            "mae": 0.0750,
            "rmse": 0.2739,
            "r2": 0.9967
        }
    }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred. Please try again."
        }
    )