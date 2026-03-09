import yaml
import numpy as np
import logging
from utils.utils import load_data, extract_features_labels, load_model

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def predict_with_model(model_name=None, input_sample=None):
    with open("config/model_configs.yaml", "r") as f:
        configs = yaml.safe_load(f)
    test_df, mlb = load_data(train=False)

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
        X = np.array([input_sample[col] for col in features]).reshape(1, -1)
    else:
        X, _ = extract_features_labels(test_df, features)

    model = load_model(model_type, model_path, X.shape[1], mlb.classes_.shape[0])
    y_pred = model.predict(X[1].reshape(1, -1))
    tags_pred = mlb.inverse_transform(y_pred)[0]
    return y_pred, tags_pred, model_cfg

if __name__ == "__main__":
    model_name = "cc_logistic_desc_code_input_weights"  # Change as needed
    # input_sample = {"prob_desc_description_embedding": ..., "source_code_embedding": ..., "prob_desc_input_spec_combined_embedding": ...}  # Fill with actual values
    y_pred, tags_pred, model_cfg = predict_with_model(model_name)  # or predict_with_model(model_name, input_sample)
    logging.info("Predictions for model %s:", model_cfg['name'])
    logging.info("%s", y_pred)
    logging.info("Decoded tags: %s", tags_pred)
