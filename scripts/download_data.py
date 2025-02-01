import os
import requests
import zipfile
import tarfile
from pathlib import Path
from config import DATA_DIR

dataset_path = DATA_DIR
DATA_URL = "https://github.com/your_username/your_repo/releases/download/v1.0/data.zip"  # Replace with your URL

def download_data():
    DATA_DIR.mkdir(exist_ok=True)
    archive_path = DATA_DIR / "data.zip"  # Or "data.tar.gz" based on your format

    # Download the file
    print(f"Downloading data from {DATA_URL}...")
    response = requests.get(DATA_URL, stream=True)

    if response.status_code == 200:
        with open(archive_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download completed.")

        # Extract the archive
        print("Extracting data...")
        if archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(DATA_DIR)
        elif archive_path.suffix in [".tar", ".gz", ".tgz"]:
            with tarfile.open(archive_path, "r:gz") as tar_ref:
                tar_ref.extractall(DATA_DIR)
        print(f"Data extracted to {DATA_DIR}.")
    else:
        print(f"Failed to download data. HTTP Status Code: {response.status_code}")

if __name__ == "__main__":
    download_data()
 