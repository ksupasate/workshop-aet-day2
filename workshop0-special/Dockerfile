# Use official Python image
FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

EXPOSE 5000

CMD ["mlflow", "ui", "--host", "0.0.0.0"]
