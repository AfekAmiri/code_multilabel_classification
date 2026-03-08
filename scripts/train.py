import yaml
import logging
from utils.utils import load_data, extract_features_labels, compute_metrics
from models.br_classifier import BRClassifier
from models.cc_classifier import CCClassifier
from models.dl_classifier import DLClassifier
from sklearn.linear_model import LogisticRegression

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def select_model(
    model_type,
    base_classifier,
    loss,
    focal_alpha,
    focal_gamma,
    hidden_dims,
    embedding_dim,
    num_labels,
    dropout,
):
    if model_type in ["br", "cc"]:
        if base_classifier == "logistic":
            base = LogisticRegression(max_iter=1000)
            if model_type == "br":
                model = BRClassifier(base_estimator=base)
            else:
                model = CCClassifier(base_estimator=base)
        else:
            model = BRClassifier() if model_type == "br" else CCClassifier()
    elif model_type == "dl":
        model = DLClassifier(
            embedding_dim=embedding_dim,
            num_labels=num_labels,
            hidden_dims=hidden_dims,
            dropout=dropout,
            loss_type=loss,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
        )
    else:
        raise ValueError("Unknown model type")
    return model


def train_model(
    features,
    model_type,
    base_classifier="logistic",
    loss="bce",
    weights_path=None,
    dropout=0.2,
    focal_alpha=1,
    focal_gamma=2,
    hidden_dims=(256, 128),
    embedding_dim=None,
    num_labels=None,
    num_epochs=500,
    lr=1e-3,
):
    # Load dataset
    df, mlb = load_data(train=True)

    # Select features and labels
    X, y = extract_features_labels(df, features)

    # Select model
    model = select_model(
        model_type,
        base_classifier,
        loss,
        focal_alpha,
        focal_gamma,
        hidden_dims,
        X.shape[1],
        y.shape[1],
        dropout,
    )

    # Fit model
    if model_type == "dl":
        model.fit(X, y, num_epochs=num_epochs, lr=lr)
    else:
        model.fit(X, y)

    # Evaluate on training set
    y_pred = model.predict(X)
    acc, f1, prec, rec, report = compute_metrics(y, y_pred, mlb)
    logging.info("Classification report:\n%s", report)
    logging.info("Accuracy: %.4f | F1: %.4f | Precision: %.4f | Recall: %.4f", acc, f1, prec, rec)

    # Save weights
    if weights_path:
        model.save(weights_path)

    return model


if __name__ == "__main__":
    with open("config/model_configs.yaml", "r") as f:
        configs = yaml.safe_load(f)

    for model_cfg in configs["models"]:
        model = train_model(
            features=model_cfg["features"],
            model_type=model_cfg["model_type"],
            base_classifier=model_cfg.get("base_classifier", "logistic"),
            loss=model_cfg.get("loss", "bce"),
            weights_path=model_cfg["weights_path"],
            hidden_dims=tuple(model_cfg.get("hidden_dims", (256, 128))),
            dropout=model_cfg.get("dropout", 0.2),
            num_epochs=model_cfg.get("num_epochs", 500),
        )
        logging.info(
            "Trained %s and saved weights to %s",
            model_cfg["name"],
            model_cfg["weights_path"],
        )
