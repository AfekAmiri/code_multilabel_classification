import pickle
import os
import numpy as np
from models.br_classifier import BRClassifier
from models.cc_classifier import CCClassifier
from models.dl_classifier import DLClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

def load_data(train=True):
    if train:
        with open(os.path.join("data", "train_df.pkl"), "rb") as f:
            df = pickle.load(f)
    else:
        with open(os.path.join("data", "test_df.pkl"), "rb") as f:
            df = pickle.load(f)
    with open(os.path.join("data", "mlb.pkl"), "rb") as f:
        mlb = pickle.load(f)
    return df, mlb

def extract_features_labels(df, features):
    feature_arrays = []
    for col in features:
        vals = df[col].values
        if isinstance(vals[0], np.ndarray):
            feature_arrays.append(np.stack(vals))
        else:
            feature_arrays.append(np.array(vals).reshape(-1, 1))
    X = np.concatenate(feature_arrays, axis=1)
    y = np.array(df["tags_encoded"].tolist())
    return X, y

def compute_metrics(y_true, y_pred, label_encoder):
	acc = accuracy_score(y_true, y_pred)
	f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
	prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
	rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
	labels = label_encoder.classes_
	return acc, f1, prec, rec, classification_report(y_true, y_pred, target_names=labels, zero_division=0)

def load_model(model_type, model_path, embedding_dim, num_labels):
	if model_type == 'br':
		model = BRClassifier()
		model.load(model_path)
	elif model_type == 'cc':
		model = CCClassifier()
		model.load(model_path)
	elif model_type == 'dl':
		model = DLClassifier(
			embedding_dim=embedding_dim,
			num_labels=num_labels)
		model.load(model_path)
	else:
		raise ValueError(f"Unknown model type: {model_type}")
	return model