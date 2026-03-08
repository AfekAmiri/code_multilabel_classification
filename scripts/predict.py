import pickle
import os
import numpy as np
from models.br_classifier import BRClassifier
from models.cc_classifier import CCClassifier
from models.dl_classifier import DLClassifier
from sklearn.linear_model import LogisticRegression

def predict_model(features, model_type, base_classifier='logistic', loss='bce', weights_path=None, input_data=None):

    # Load dataset for feature columns (optional, for column order)
    with open(os.path.join('data', 'test_df.pkl'), 'rb') as f:
        df = pickle.load(f)
    
    with open(os.path.join('data', 'mlb.pkl'), 'rb') as f:
        mlb = pickle.load(f)

    # Prepare input features
    if input_data is None:
        raise ValueError('No input data provided')
    X = np.array([input_data[col] for col in features]).reshape(1, -1)

    # Model selection
    if model_type == 'br':
        base = LogisticRegression() if base_classifier == 'logistic' else None
        model = BRClassifier(base_estimator=base)
    elif model_type == 'cc':
        base = LogisticRegression() if base_classifier == 'logistic' else None
        model = CCClassifier(base_estimator=base)
    elif model_type == 'dl':
        model = DLClassifier(num_labels=len(features), loss_type=loss)
    else:
        raise ValueError('Unknown model type')

    # Load weights
    if weights_path:
        model.load(weights_path)
    else:
        raise ValueError('No weights_path provided')

    # Predict
    y_pred = model.predict(X)
    return y_pred
