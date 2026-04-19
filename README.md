# COMP 3610 Assignment 4: MLOps & Model Deployment

A containerized ML prediction service for NYC Yellow Taxi tip amount prediction, built with MLflow, FastAPI, and Docker.

## Prerequisites

- Python 3.11+
- Docker Desktop installed and running
- Git

## Project Structure

    assignment4/
    ├── assignment4.ipynb       # Main notebook documenting all work
    ├── app.py                  # FastAPI application
    ├── test_app.py             # pytest test suite
    ├── Dockerfile              # Container definition
    ├── docker-compose.yml      # Service orchestration
    ├── requirements.txt        # Python dependencies
    ├── README.md               # This file
    ├── .gitignore              # Excludes data, models, mlruns
    ├── .dockerignore           # Excludes unnecessary files from image
    └── models/                 # Saved model files (gitignored)

## Setup & Installation

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd assignment4
```

### 2. Create and Activate a Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Data and Train Models
Open and run `assignment4.ipynb` from top to bottom. The notebook will:
- Download the NYC Yellow Taxi Trip Records dataset automatically
- Clean and preprocess the data
- Train and evaluate models
- Save the model and preprocessor to `models/`
- Log experiments to MLflow

## Running the Project

### Option A: Local Development
```bash
# Start MLflow tracking server
mlflow ui --port 5000

# In a separate terminal, start the API
uvicorn app:app --reload --port 8000
```

### Option B: Docker Compose (Recommended)
```bash
docker compose up --build
```

This starts the API service on port 8000.

### Option C: Docker Only
```bash
# Build the image
docker build -t taxi-tip-api .

# Run the container
docker run -d --name taxi-api -p 8000:8000 taxi-tip-api
```

## API Usage

### Single Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "trip_distance": 3.5,
    "fare_amount": 14.5,
    "pickup_hour": 14,
    "passenger_count": 1,
    "trip_duration_minutes": 12.0,
    "total_amount": 18.8
  }'
```

### Batch Prediction
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {
        "trip_distance": 3.5,
        "fare_amount": 14.5,
        "pickup_hour": 14,
        "passenger_count": 1,
        "trip_duration_minutes": 12.0,
        "total_amount": 18.8
      }
    ]
  }'
```

### Health Check
```bash
curl http://localhost:8000/health
```

### API Documentation
Visit `http://localhost:8000/docs` for the auto-generated Swagger UI.

## Running Tests
```bash
pytest test_app.py -v
```

## Environment Variables

| Variable           | Default                           | Description               |
|--------------------|-----------------------------------|---------------------------|
| MODEL_PATH         | /app/models/registered_model.pkl  | Path to the trained model |
| PREPROCESSOR_PATH  | /app/models/preprocessor.pkl      | Path to the preprocessor  |

## Container Info

| Field        | Value                |
|--------------|----------------------|
| Base Image   | python:3.11-slim     |
| Content Size | 348MB                |
| API Port     | 8000                 |

## Dataset

NYC Yellow Taxi Trip Records (January 2024) — downloaded automatically by the notebook from the NYC TLC Trip Record Data portal. Do not commit the raw data files.

## Model

A Linear Regression model trained to predict `tip_amount` from NYC taxi trip features. Tracked and registered using MLflow.

| Metric | Value  |
|--------|--------|
| MAE    | 0.0750 |
| RMSE   | 0.2739 |
| R²     | 0.9967 |

## Course Information

- **Course:** COMP 3610 - Big Data Analytics
- **Institution:** The University of the West Indies
- **Semester:** II, 2025-2026