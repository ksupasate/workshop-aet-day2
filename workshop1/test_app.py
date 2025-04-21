
import requests

# Replace with your local or deployed URL
URL = "http://127.0.0.1:8000/predict"

# Sample payload for testing
sample_data = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}

# Send POST request
response = requests.post(URL, json=sample_data)

# Output
print("Status Code:", response.status_code)
print("Response JSON:", response.json())
