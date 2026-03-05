import os
import json
import logging
from glob import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def combine_jsons(input_dir, output_path):
    json_files = glob(os.path.join(input_dir, '*.json'))
    logging.info(f"Found {len(json_files)} JSON files in {input_dir}")
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for idx, file in enumerate(json_files):
            with open(file, 'r', encoding='utf-8') as infile:
                obj = json.load(infile)
                json.dump(obj, outfile, ensure_ascii=False)
                outfile.write('\n')
            if (idx + 1) % 500 == 0:
                logging.info(f"Processed {idx + 1} files...")
    logging.info(f"All JSON files combined into {output_path}")

if __name__ == "__main__":
    input_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'code_classification_dataset')
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset.json')
    combine_jsons(input_dir, output_path)
