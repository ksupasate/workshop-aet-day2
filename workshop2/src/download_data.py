import requests
import zipfile
import io
import os

def download_and_extract():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip"
    r = requests.get(url)
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall("data/raw/")
    print("Downloaded and extracted power plant dataset to data/raw/")

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    download_and_extract()
