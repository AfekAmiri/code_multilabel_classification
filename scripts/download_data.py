import gdown
import zipfile
import os
import logging
from dotenv import load_dotenv
import json
from glob import glob


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def download_data(url, output_zip, data_dir):
    if not url:
        logging.error("URL not set.")
        raise ValueError("URL not set.")

    logging.info("Downloading data...")
    gdown.download(url, output_zip, quiet=False)

    os.makedirs(data_dir, exist_ok=True)

    logging.info(f"Extracting {output_zip} to {data_dir}...")
    with zipfile.ZipFile(output_zip, "r") as zip_ref:
        zip_ref.extractall(data_dir)

    logging.info(f"Downloaded and extracted to {data_dir}")


def combine_jsons(input_dir, output_path):
    json_files = glob(os.path.join(input_dir, "*.json"))
    logging.info(f"Found {len(json_files)} JSON files in {input_dir}")
    with open(output_path, "w", encoding="utf-8") as outfile:
        for idx, file in enumerate(json_files):
            with open(file, "r", encoding="utf-8") as infile:
                obj = json.load(infile)
                json.dump(obj, outfile, ensure_ascii=False)
                outfile.write("\n")
            if (idx + 1) % 500 == 0:
                logging.info(f"Processed {idx + 1} files...")
    logging.info(f"All JSON files combined into {output_path}")


if __name__ == "__main__":
    load_dotenv(
        dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"), override=True
    )
    url = os.getenv("URL")
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    output_zip = os.path.join(data_dir, "data.zip")
    download_data(url, output_zip, data_dir)
    input_dir = os.path.join(data_dir, "code_classification_dataset")
    output_path = os.path.join(data_dir, "dataset.json")
    combine_jsons(input_dir, output_path)
