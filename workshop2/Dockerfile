FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and trained model
COPY src ./src
COPY models ./models

EXPOSE 8000

CMD ["uvicorn", "src.deploy:app", "--host", "0.0.0.0", "--port", "8000"]
