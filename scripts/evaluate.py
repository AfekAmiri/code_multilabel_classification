import os
import numpy as np
import pandas as pd
import logging
import yaml
from utils.utils import load_data, extract_features_labels, compute_metrics, load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

feature_map = {'prob_desc_description_embedding': 'desc',
                'source_code_embedding': 'code',
                'prob_desc_input_spec_combined_embedding': 'input',
                'prob_desc_output_spec_combined_embedding': 'output',
                'prob_desc_notes_embedding': 'notes',
                'difficulty': 'difficulty'}

def evaluate(config_path="config/model_configs.yaml"):
	with open(config_path, "r") as f:
		configs = yaml.safe_load(f)
	test_df, mlb = load_data(train=False)
	results = []
	for model_cfg in configs["models"]:
		model_name = model_cfg["name"]
		model_path = model_cfg["weights_path"]
		model_type = model_cfg["model_type"]
		features = model_cfg["features"]
		X, y_true = extract_features_labels(test_df, features)
		model = load_model(model_type, model_path, X.shape[1], y_true.shape[1])
		y_pred = model.predict(X)
		acc, f1, prec, rec, report = compute_metrics(y_true, y_pred, mlb)
		mapped_features = [feature_map.get(f, f) for f in features]
		results.append({
			'model': model_name,
			'features': ','.join(mapped_features),
			'accuracy': acc,
			'f1_weighted': f1,
			'precision_weighted': prec,
			'recall_weighted': rec,
			'report': report
		})
	df_results = pd.DataFrame(results)
	with open("docs/results.md", "w", encoding="utf-8") as f:
		f.write("# Résultats des modèles\n\n")
		f.write(df_results.drop(columns=['report']).sort_values('f1_weighted', ascending=False).to_markdown(index=False))
		f.write("\n\n## Détails des métriques\n\n")
		for idx, row in df_results.sort_values('f1_weighted', ascending=False).iterrows():
			f.write(f"\n### {row['model']}\n\n")
			f.write(row['report'])
			f.write("\n\n")

if __name__ == "__main__":
	evaluate()
