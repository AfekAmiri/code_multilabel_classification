import os
import numpy as np
import pandas as pd
import logging
import yaml
from utils.utils import load_data, extract_features_labels, compute_metrics, load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def evaluate(config_path="config/model_configs.yaml"):
	with open(config_path, "r") as f:
		configs = yaml.safe_load(f)
	test_df, mlb = load_data(train=False)
	results = []
	for model_cfg in configs["models"]:
		model_path = model_cfg["weights_path"]
		model_type = model_cfg["model_type"]
		features = model_cfg["features"]
		X, y_true = extract_features_labels(test_df, features)
		model = load_model(model_type, model_path, X.shape[1], y_true.shape[1])
		y_pred = model.predict(X)
		acc, f1, prec, rec, report = compute_metrics(y_true, y_pred, mlb)
		results.append({
			'model': model_type,
			'features': ','.join(features),
			'accuracy': acc,
			'f1_weighted': f1,
			'precision_weighted': prec,
			'recall_weighted': rec,
			'report': report
		})
	df_results = pd.DataFrame(results)
	logging.info(df_results.sort_values('f1_weighted', ascending=False).to_markdown(index=False))

if __name__ == "__main__":
	evaluate()
