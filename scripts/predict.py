import yaml
import numpy as np
import logging
import pickle
import os
from utils.utils import load_data, extract_features_labels, load_model, embed

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def predict_with_model(model_name=None, input_sample=None):
    with open("config/model_configs.yaml", "r") as f:
        configs = yaml.safe_load(f)
    test_df, mlb = load_data(train=False)

    # Load scaler and imputer
    scaler_path = os.path.join("data", "scaler.pkl")
    imputer_path = os.path.join("data", "imputer.pkl")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(imputer_path, "rb") as f:
        imputer = pickle.load(f)

    # Select model config
    if model_name:
        model_cfg = next((m for m in configs["models"] if m["name"] == model_name), configs["models"][0])
    else:
        model_cfg = configs["models"][0]

    model_path = model_cfg["weights_path"]
    model_type = model_cfg["model_type"]
    features = model_cfg["features"]

    # Prepare input features
    if input_sample is not None:
        processed = []
        for col in features:
            val = input_sample.get(col, None)
            if col == "difficulty":
                # Impute and scale
                val = np.array([[val]])
                val = imputer.transform(val)
                val = scaler.transform(val)[0][0]
                processed.append(val)
            elif col.startswith("prob_desc") or col == "source_code":
                # Embed textual columns
                processed.append(embed(val))
            else:
                processed.append(val)
        X = np.array(processed).reshape(1, -1)
    else:
        X, _ = extract_features_labels(test_df, features)

    model = load_model(model_type, model_path, X.shape[1], mlb.classes_.shape[0])
    y_pred = model.predict(X[0].reshape(1, -1))
    y_proba = model.predict_proba(X[0].reshape(1, -1))[0] if hasattr(model, "predict_proba") else None
    print("proba", y_proba)
    tags_pred = mlb.inverse_transform(y_pred)[0]
    return y_pred, tags_pred, y_proba, model_cfg

if __name__ == "__main__":
    model_name = "cc_logistic_desc_code_input_weights"  # Change as needed
    # input_sample = { ... }  # Fill with actual values
    y_pred, tags_pred, y_proba, model_cfg = predict_with_model(model_name)  # or predict_with_model(model_name, input_sample)
    logging.info("Predictions for model %s:", model_cfg['name'])
    logging.info("%s", y_pred)
    logging.info("Decoded tags: %s", tags_pred)
    logging.info("Prediction probabilities: %s", y_proba)
