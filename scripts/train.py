import pickle
import os
from xml.parsers.expat import features, model
import numpy as np
from models.br_classifier import BRClassifier
from models.cc_classifier import CCClassifier
from models.dl_classifier import DLClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score, classification_report



def train_model(features, model_type, base_classifier='logistic', loss='bce', weights_path=None, dropout=0.2, focal_alpha=1, focal_gamma=2, hidden_dims=(256,128), embedding_dim=None, num_labels=None):
    # Load dataset
    with open(os.path.join('data', 'train_df.pkl'), 'rb') as f:
        df = pickle.load(f)
    
    with open(os.path.join('data', 'mlb.pkl'), 'rb') as f:
        mlb = pickle.load(f)

    # Select features and labels
    #X = np.vstack([df[col] for col in features]).T
    #X = np.concatenate([np.stack(df[col].values) for col in features], axis=1)

    feature_arrays = []
    for col in features:
        vals = df[col].values
        if isinstance(vals[0], np.ndarray):  # Embedding column
            feature_arrays.append(np.stack(vals))
        else:  # Scalar column
            feature_arrays.append(np.array(vals).reshape(-1, 1))

    X = np.concatenate(feature_arrays, axis=1)
    y = np.array(df['tags_encoded'].tolist())

    # Model selection
    if model_type == 'br':
        base = LogisticRegression(max_iter=1000) if base_classifier == 'logistic' else None
        model = BRClassifier(base_estimator=base)
    elif model_type == 'cc':
        base = LogisticRegression(max_iter=1000) if base_classifier == 'logistic' else None
        model = CCClassifier(base_estimator=base)
    elif model_type == 'dl':
        model = DLClassifier(
            embedding_dim=X.shape[1],
            num_labels=y.shape[1],
            hidden_dims=hidden_dims,
            dropout=dropout,
            loss_type=loss,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma
        )
    else:
        raise ValueError('Unknown model type')

    # Cross-validation
    #cv_results = cross_validate(
    #model, X, y,
    #cv=5,
    #scoring = {
    #'accuracy': 'accuracy',
    #'f1_weighted': make_scorer(f1_score, average='weighted', zero_division=0),
    #'precision_weighted': make_scorer(precision_score, average='weighted', zero_division=0),
    #'recall_weighted': make_scorer(recall_score, average='weighted', zero_division=0)
    #},
    #return_train_score=True)

    #print("Train scores:", cv_results['train_accuracy'], cv_results['train_f1_weighted'], cv_results['train_recall_weighted'], cv_results['train_precision_weighted'])
    #print("Validation scores:", cv_results['test_accuracy'], cv_results['test_f1_weighted'], cv_results['test_recall_weighted'], cv_results['test_precision_weighted'])

    # Fit final model on all data
    model.fit(X, y)
    y_pred = model.predict(X)
    tag_names = mlb.classes_
    print(classification_report(y, y_pred, target_names=tag_names, zero_division=0))

    # Save weights
    if weights_path:
        model.save(weights_path)

    return model

if __name__ == '__main__':
    # Example usage
    weights_path='models/checkpoints/dl_desc_code_input_500_weights.pkl'
    model = train_model(
        features=['prob_desc_description_embedding','prob_desc_input_spec_combined_embedding','source_code_embedding'],
        model_type='dl',
        base_classifier='logistic',
        loss='bce',
        weights_path=weights_path
    )
    print("Model trained and weights saved in ", weights_path)