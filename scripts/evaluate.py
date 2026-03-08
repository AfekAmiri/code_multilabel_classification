import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from models.br_classifier import BRClassifier
from models.cc_classifier import CCClassifier
from models.dl_classifier import DLClassifier
from sklearn.linear_model import LogisticRegression

# Map feature keywords to column names
feature_map = {
	'code': 'source_code_embedding',
	'desc': 'prob_desc_description_embedding',
	'input': 'prob_desc_input_spec_combined_embedding',
	'output': 'prob_desc_output_spec_combined_embedding',
	'notes': 'prob_desc_notes_embedding',
	'difficulty': 'difficulty'
}

checkpoint_dir = os.path.join('models', 'checkpoints')
model_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('_weights.pkl')]

# Load test set and mlb
with open(os.path.join('data', 'test_df.pkl'), 'rb') as f:
	test_df = pickle.load(f)
with open(os.path.join('data', 'mlb.pkl'), 'rb') as f:
	mlb = pickle.load(f)

results = []

for model_file in model_files:
	model_path = os.path.join(checkpoint_dir, model_file)
	name_parts = model_file.replace('_weights.pkl', '').split('_')
	model_type = None
	features_abbr = []
	features = []
	for part in name_parts:
		if part in ['br', 'cc', 'dl']:
			model_type = part
		elif part in feature_map:
			features.append(feature_map[part])
			features_abbr.append(part)

	# Build feature matrix
	feature_arrays = []
	for col in features:
		vals = test_df[col].values
		if isinstance(vals[0], np.ndarray):
			feature_arrays.append(np.stack(vals))
		else:
			feature_arrays.append(np.array(vals).reshape(-1, 1))
	X = np.concatenate(feature_arrays, axis=1)
	y_true = np.array(test_df['tags_encoded'].tolist())

	# Load model
	if model_type == 'br':
		model = BRClassifier()
		model.load(model_path)
	elif model_type == 'cc':
		model = CCClassifier()
		model.load(model_path)
	elif model_type == 'dl':
		model = DLClassifier(
            embedding_dim=X.shape[1],
            num_labels=y_true.shape[1],
        )
		model.load(model_path)
	else:
		continue

	# Predict
	y_pred = model.predict(X)

	# Compute metrics
	acc = accuracy_score(y_true, y_pred)
	f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
	prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
	rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)

	results.append({
		'model': model_type,
		'features': ','.join(features_abbr),
		'accuracy': acc,
		'f1_weighted': f1,
		'precision_weighted': prec,
		'recall_weighted': rec
	})

# Create DataFrame and display
df_results = pd.DataFrame(results)

# Print markdown table
print("\nMarkdown Table:\n")
print(df_results.sort_values('f1_weighted', ascending=False).to_markdown(index=False))

# Print LaTeX table
#print("\nLaTeX Table:\n")
#print(df_results.sort_values('f1_weighted', ascending=False).to_latex(index=False))
