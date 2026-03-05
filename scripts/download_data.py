import gdown
import zipfile
import os
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_data(url, output_zip, data_dir):
    if not url:
        logging.error("URL not set.")
        raise ValueError("URL not set.")
    
    logging.info(f"Downloading data...")
    gdown.download(url, output_zip, quiet=False)

    os.makedirs(data_dir, exist_ok=True)

    logging.info(f"Extracting {output_zip} to {data_dir}...")
    with zipfile.ZipFile(output_zip, "r") as zip_ref:
        zip_ref.extractall(data_dir)

    logging.info(f"Downloaded and extracted to {data_dir}")
if __name__ == "__main__":
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'), override=True)
    url = os.getenv("URL")
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    output_zip = os.path.join(data_dir, 'data.zip')
    download_data(url, output_zip, data_dir)