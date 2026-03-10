#!/usr/bin/env python

import argparse
import os
import logging
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import yaml
import json
from scripts.train import train_model
from scripts.evaluate import evaluate
from scripts.predict import predict_with_model

def main():
    parser = argparse.ArgumentParser(prog="illuin", description="Illuin CLI: train, evaluate, predict")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train command
    train_parser = subparsers.add_parser("train", help="Train models")
    train_parser.add_argument("--config", required=True, help="Path to model config YAML")
    train_parser.add_argument("--log-level", default="INFO", help="Logging level")
    train_parser.add_argument("--output-dir", default=None, help="Directory to save model weights")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate models")
    eval_parser.add_argument("--config", required=True, help="Path to model config YAML")
    eval_parser.add_argument("--log-level", default="INFO", help="Logging level")

    # Predict command
    pred_parser = subparsers.add_parser("predict", help="Predict with a model")
    pred_parser.add_argument("--config", required=True, help="Path to model config YAML")
    pred_parser.add_argument("--model", required=True, help="Model name to use for prediction")
    pred_parser.add_argument("--input", required=False, help="Path to JSON file with input sample or inline JSON string")
    pred_parser.add_argument("--log-level", default="INFO", help="Logging level")

    args = parser.parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s - %(levelname)s - %(message)s")


    if args.command == "train":
        with open(args.config, "r") as f:
            configs = yaml.safe_load(f)
        for model_cfg in configs["models"]:
            # Update weights_path if output-dir is specified
            weights_path = model_cfg["weights_path"]
            if args.output_dir:
                base_name = os.path.basename(weights_path)
                weights_path = os.path.join(args.output_dir, base_name)
            train_model(
                features=model_cfg["features"],
                model_type=model_cfg["model_type"],
                base_classifier=model_cfg.get("base_classifier", "logistic"),
                loss=model_cfg.get("loss", "bce"),
                weights_path=weights_path,
                hidden_dims=tuple(model_cfg.get("hidden_dims", (256, 128))),
                dropout=model_cfg.get("dropout", 0.2),
                num_epochs=model_cfg.get("num_epochs", 500),
            )
            logging.info("Trained %s and saved weights to %s", model_cfg["name"], weights_path)

    elif args.command == "evaluate":
        evaluate(config_path=args.config)

    elif args.command == "predict":
        input_sample = None
        if args.input:
            try:
                if args.input.endswith(".json"):
                    with open(args.input, "r") as f:
                        input_sample = json.load(f)
                else:
                    input_sample = json.loads(args.input)
            except Exception as e:
                logging.error("Failed to load input sample: %s", e)
                sys.exit(1)
        y_pred, tags_pred, y_proba, model_cfg = predict_with_model(args.model, input_sample)
        logging.info("Predictions for model %s:", model_cfg['name'])
        logging.info("%s", y_pred)
        logging.info("Decoded tags: %s", tags_pred)
        logging.info("Prediction probabilities: %s", y_proba)

if __name__ == "__main__":
    main()
