# Hands-on Project - End-to-End ML Pipeline

This repository contains an automated machine learning pipeline for real-world datasets, covering data ingestion, preprocessing, model training, evaluation, and deployment.

## Directory Structure
```
├── README.md
├── requirements.txt
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
│   └── raw/          # Raw input data
│   └── processed/    # Preprocessed data
├── src/
│   ├── ingest.py     # Data ingestion logic
│   ├── preprocess.py # Data preprocessing functions
│   ├── train.py      # Model training and MLflow logging
│   ├── evaluate.py   # Model evaluation
│   └── deploy.py     # FastAPI app for serving
└── tests/
    ├── test_preprocess.py
    └── test_train.py
```

## Getting Started
### Power Plant Dataset
To fetch the Combined Cycle Power Plant dataset from UCI:
```bash
python src/download_data.py
```


1. **Clone** the repository:
   ```bash
   git clone https://github.com/your-org/end_to_end_ml_pipeline_workshop.git
   cd end_to_end_ml_pipeline_workshop
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the pipeline**:
   ```bash
   python src/ingest.py --input data/raw/dataset.csv --output data/processed/
   python src/preprocess.py --input data/raw/ --output data/processed/
   python src/train.py --input data/processed/ --model-output models/
   python src/evaluate.py --model models/model.pkl --test-data data/processed/
   ```

4. **Deploy** the model:
   ```bash
   docker build -t ml_pipeline_app .
   docker run -p 8000:8000 ml_pipeline_app
   ```

## CI/CD

A GitHub Actions workflow (`.github/workflows/ci.yml`) is included to automatically install dependencies, run tests, and validate code on each push or pull request.
