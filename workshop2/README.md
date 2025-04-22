# Hands-on Project - End-to-End ML Pipeline

This repository contains an end-to-end machine learning pipeline using the UCI Combined Cycle Power Plant dataset. It covers data download, ingestion, preprocessing, model training, evaluation, testing, and deployment via FastAPI and Docker.

## Directory Structure
```
.
├── README.md
├── requirements.txt
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── pytest.ini
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
│   ├── raw/          # Raw downloaded and converted CSV
│   └── processed/    # Processed train/test CSV files
├── src/
│   ├── download_data.py  # Download & XLSX→CSV converter
│   ├── ingest.py         # Ingest CSV to raw_data.csv
│   ├── preprocess.py     # Train/test split & scaling
│   ├── train.py          # Model training & MLflow logging
│   ├── evaluate.py       # Model evaluation metrics
│   └── deploy.py         # FastAPI service for inference
├── models/
│   └── model.pkl         # Saved trained model
└── tests/
    ├── test_preprocess.py
    └── test_train.py
```

## Getting Started

1. **Clone** the repository:  
   ```bash
   git clone https://github.com/your-org/end_to_end_ml_pipeline_workshop.git
   cd end_to_end_ml_pipeline_workshop
   ```

2. **Install dependencies**:  
   ```bash
   pip install -r requirements.txt
   ```

3. **Download and prepare the dataset**:  
   ```bash
   python src/download_data.py
   ```  
   This downloads the UCI CCPP zip, extracts the `.xlsx` file, and converts it to `data/raw/dataset.csv`.

4. **Ingest raw data**:  
   ```bash
   python src/ingest.py --input data/raw/dataset.csv --output data/raw/
   ```  
   Produces `data/raw/raw_data.csv`.

5. **Preprocess data**:  
   ```bash
   python src/preprocess.py --input data/raw/ --output data/processed/
   ```  
   Creates:
   - `data/processed/train_processed.csv`
   - `data/processed/test_processed.csv`
   - `data/processed/scaler.pkl`

6. **Train the model**:  
   ```bash
   python src/train.py --input data/processed/ --model-output models/
   ```  
   Trains a `RandomForestRegressor`, logs parameters & model to MLflow, and saves `models/model.pkl`.

7. **Evaluate the model**:  
   ```bash
   python src/evaluate.py --model models/model.pkl --test-data data/processed/
   ```  
   Prints:
   - RMSE
   - MAE
   - R²

8. **Run the test suite**:  
   ```bash
   pytest --maxfail=1 --disable-warnings -q
   ```  
   This will execute:
   - `tests/test_preprocess.py` — verifies preprocessing outputs and scaler file.
   - `tests/test_train.py` — checks that the model is trained and persisted correctly.  
   Make sure all tests pass before building the Docker image or merging any changes.

9. **Deploy the model**:

   **Option A: Bake the model into the image**  
   ```bash
   docker build -t ml_pipeline_app .
   docker run -p 8000:8000 ml_pipeline_app
   ```

   **Option B: Mount the model at runtime**  
   ```bash
   docker build -t ml_pipeline_app .
   docker run -p 8000:8000 \
     -v $(pwd)/models:/app/models \
     ml_pipeline_app
   ```

10. **Verify deployment**:  
    - Check container is running: `docker ps`  
    - Access docs: `curl http://localhost:8000/docs`  
    - Test prediction:
      ```bash
      curl -X POST http://localhost:8000/predict \
           -H 'Content-Type: application/json' \
           -d '{
             "data": [
               {"AT":14.96,"V":41.76,"AP":1024.07,"RH":73.17},
               {"AT":25.18,"V":62.96,"AP":1020.04,"RH":59.08}
             ]
           }'
      ```

## CI/CD

A GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push and pull request to:

- Install dependencies
- Run unit tests (`pytest`)
- Validate code quality

---
