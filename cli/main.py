#!/usr/bin/env python
import argparse
import json
from scripts.train import train_model
from scripts.predict import predict_model


def train_command(args):
    model = train_model(
        features=args.features,
        model_type=args.model,
        base_classifier=args.base_classifier,
        loss=args.loss,
        weights_path=args.weights_path
    )
    print(f"Training complete. Model: {args.model}. Weights saved to: {args.weights_path}")


def predict_command(args):
    input_data = json.loads(args.input_data)
    y_pred = predict_model(
        features=args.features,
        model_type=args.model,
        base_classifier=args.base_classifier,
        loss=args.loss,
        weights_path=args.weights_path,
        input_data=input_data
    )
    print(f"Prediction: {y_pred}")


def main():
    parser = argparse.ArgumentParser(prog='illuin')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--features', nargs='+', required=True, help='Columns to use as features')
    train_parser.add_argument('--model', choices=['br', 'cc', 'dl'], required=True)
    train_parser.add_argument('--base_classifier', default='logistic', help='Base classifier for BR/CC')
    train_parser.add_argument('--loss', choices=['bce', 'focal'], default='bce', help='Loss for DL')
    train_parser.add_argument('--weights_path', default=None, help='Path to save model weights')
    train_parser.set_defaults(func=train_command)

    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict with a trained model')
    predict_parser.add_argument('--features', nargs='+', required=True, help='Columns to use as features')
    predict_parser.add_argument('--model', choices=['br', 'cc', 'dl'], required=True)
    predict_parser.add_argument('--base_classifier', default='logistic', help='Base classifier for BR/CC')
    predict_parser.add_argument('--loss', choices=['bce', 'focal'], default='bce', help='Loss for DL')
    predict_parser.add_argument('--weights_path', required=True, help='Path to load model weights')
    predict_parser.add_argument('--input_data', required=True, help='Input data as a JSON string')
    predict_parser.set_defaults(func=predict_command)

    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
