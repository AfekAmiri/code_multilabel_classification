# Illuin CLI

A command-line interface for training, evaluating, and predicting with models in the Illuin project.

## Installation

From the `cli` directory, install the CLI in editable mode using uv:

```bash
uv pip install -e .
```

Or with pip:

```bash
pip install -e .
```

This will create the `illuin` command in your virtual environment. Make sure your environment's Scripts/bin directory is in your PATH.

## Usage

Run the CLI from your project root or any directory (ensure config paths are correct):

### Train models

```bash
illuin train --config config/model_configs.yaml [--log-level INFO] [--output-dir path/to/save]
```
- `--config`: Path to the model config YAML file (required)
- `--log-level`: Logging level (default: INFO)
- `--output-dir`: Directory to save model weights (optional)

### Evaluate models

```bash
illuin evaluate --config config/model_configs.yaml [--log-level INFO]
```
- `--config`: Path to the model config YAML file (required)
- `--log-level`: Logging level (default: INFO)

### Predict with a model

```bash
illuin predict --config config/model_configs.yaml --model MODEL_NAME [--input sample.json|'{"feature": value}'] [--log-level INFO]
```
- `--config`: Path to the model config YAML file (required)
- `--model`: Model name to use for prediction (required)
- `--input`: Path to JSON file or inline JSON string (optional)
- `--log-level`: Logging level (default: INFO)

## Configuration

Place your model config YAML in the `config/` directory (e.g., `config/model_configs.yaml`).

## Help

Run `illuin --help` or `illuin <command> --help` to see available options and usage details.

## Example

```bash
illuin evaluate --config config/model_configs.yaml
```

## Troubleshooting

- If you get `ModuleNotFoundError`, ensure you installed from the `cli` directory and your environment is activated.
- Always specify the correct path to your config file.
