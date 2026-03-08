# tech_challenge_illuin_technology

## Project Structure

```
project-root/
│
├── data/                # Datasets and extracted files
│   └── code_classification_dataset/  # Sample JSONs for code classification
├── scripts/             # Download, training, evaluation, and prediction scripts
├── models/              # Model definitions and checkpoints
├── utils/               # Utility functions and helpers
├── config/              # Model configuration YAML files
├── cli/                 # Illuin CLI for model operations
├── notebooks/           # Jupyter notebooks for data analysis and preprocessing
├── docs/                # Documentation for GitHub Pages
├── pyproject.toml       # Project metadata and dependencies
├── README.md            # Project overview and instructions
└── main.py              # Main entry point
```

## Installation

Install dependencies and the CLI using uv from the project root or cli directory:

```bash
uv pip install -e .
```

Activate your virtual environment before running any scripts or CLI commands.

## Components

- **data/**: Contains datasets and sample files for model training and evaluation.
- **scripts/**: Python scripts for downloading data, training, evaluating, and predicting.
- **models/**: Model classes and saved checkpoints.
- **utils/**: Helper functions for metrics, plotting, and reproducibility.
- **config/**: YAML files specifying model configurations.
- **cli/**: Illuin CLI for training, evaluating, and predicting. See cli/README.md for usage.
- **notebooks/**: Jupyter notebooks for data analysis and preprocessing (`analysis.ipynb`, `preprocessing.ipynb`).
- **docs/**: Project documentation and results for GitHub Pages.

## Usage

- Use the Illuin CLI for model operations:
	- `illuin train --config config/model_configs.yaml`
	- `illuin evaluate --config config/model_configs.yaml`
	- `illuin predict --config config/model_configs.yaml --model MODEL_NAME`
- Explore and preprocess data in the notebooks.
- Refer to scripts for custom workflows and automation.

## Configuration

Model and training parameters are defined in YAML files under `config/`.

## Documentation

See the `docs/` directory or GitHub Pages for detailed guides and results.

## Professional Notes

- Keep your environment isolated using virtual environments.
- Always specify correct paths for config and data files.
- For troubleshooting and advanced usage, see cli/README.md and docs.