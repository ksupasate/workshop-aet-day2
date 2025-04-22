import requests, zipfile, io, os, glob
import pandas as pd

def download_and_prepare():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip"
    response = requests.get(url)
    # Extract the zip contents to raw data directory
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall("data/raw/")
    # Find the xlsx file
    xlsx_files = glob.glob("data/raw/CCPP/*.xlsx")
    if not xlsx_files:
        print("No XLSX file found in data/raw/CCPP")
        return
    xlsx_path = xlsx_files[0]
    # Read and convert to CSV
    df = pd.read_excel(xlsx_path)
    csv_path = "data/raw/CCPP/dataset.csv"
    df.to_csv(csv_path, index=False)
    print(f"Converted {os.path.basename(xlsx_path)} to {csv_path}")

if __name__ == "__main__":
    os.makedirs("data/raw/CCPP", exist_ok=True)
    download_and_prepare()
