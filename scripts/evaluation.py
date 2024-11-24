import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import models
from models import GRUClassifier, LSTMClassifier, LogisticRegressionClassifier, TransformerClassifier

# Import utilities
from utils.evaluation_utils import evaluate_model, compute_metrics

# Load configuration
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def initialize_model(model_config, data_loader=None):
    """Initialize model dynamically based on configuration."""
    model_name = model_config["name"]

    if model_name == "GRU":
        return GRUClassifier(model_config["input_size"], model_config["hidden_size"], model_config["num_layers"])
    
    elif model_name == "LSTM":
        return LSTMClassifier(model_config["input_size"], model_config["hidden_size"], model_config["num_layers"])
    
    elif model_name == "LogisticRegression":
        if data_loader is None:
            raise ValueError("DataLoader is required for LogisticRegression initialization.")
        feature_dict = LogisticRegressionClassifier.build_feature_dict(data_loader, model_config["max_N"])
        return LogisticRegressionClassifier(feature_dict=feature_dict)
    
    elif model_name == "Transformer":
        return TransformerClassifier(
            input_dim=model_config["input_dim"],
            model_dim=model_config["model_dim"],
            num_heads=model_config["num_heads"],
            num_layers=model_config["num_layers"],
            dim_feedforward=model_config["dim_feedforward"]
        )
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    

def analyze_model_agreement(labels, **model_predictions):
    """
    Analyzes agreement between multiple models' predictions.   
    Args:
    - labels (Tensor): Ground-truth labels.
    - model_predictions (dict): Dictionary of model predictions.
    """
    # Compute agreement for all models being correct or incorrect
    all_correct_indices = torch.ones_like(labels, dtype=torch.bool)
    all_incorrect_indices = torch.ones_like(labels, dtype=torch.bool)

    for predictions in model_predictions.values():
        all_correct_indices &= (predictions == labels)
        all_incorrect_indices &= (predictions != labels)

    # Get indices
    all_correct_samples = torch.nonzero(all_correct_indices).squeeze().tolist()
    all_incorrect_samples = torch.nonzero(all_incorrect_indices).squeeze().tolist()

    # Print statistics
    print(f"Number of samples where all models are correct: {len(all_correct_samples)}")
    print(f"Number of samples where all models are incorrect: {len(all_incorrect_samples)}")
    print(f"Total number of samples: {len(labels)}")

def main():
    # Load configuration
    config = load_config("config.yaml")
    training_config = config["training"]
    testing_config = config["testing"]
    model_configs = config["models"]
    paths_config = config["paths"]

    # Set device
    device = torch.device(testing_config["device"] if torch.cuda.is_available() else "cpu")

    # Load test data
    data_dir = Path(paths_config["data"])
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    X_train_loaded = torch.load(data_dir / "X_train.pt")
    y_train_loaded = torch.load(data_dir / "y_train.pt")
    lengths_train_loaded = torch.load(data_dir / "lengths_train.pt") 
    train_loader = DataLoader(
        TensorDataset(X_train_loaded, y_train_loaded, lengths_train_loaded),
        batch_size=training_config["batch_size"],
        shuffle=True,
    )

    X_test_loaded = torch.load(data_dir / "X_test.pt")
    y_test_loaded = torch.load(data_dir / "y_test.pt")
    lengths_test_loaded = torch.load(data_dir / "lengths_test.pt") 
    test_loader = DataLoader(
        TensorDataset(X_test_loaded, y_test_loaded, lengths_test_loaded),
        batch_size=testing_config["batch_size"],
        shuffle=False,
    )
    
    # Directory for trained models
    output_dir = Path(paths_config["output"])
    if not output_dir.exists():
        raise FileNotFoundError(f"Checkpoints directory not found: {output_dir}")

    # Test each model in the configuration
    model_predictions = {}

    for model_config in model_configs:
        model_name = model_config["name"]
        print(f"Evaluating {model_name} model...")

        # Initialize model and load weights
        model = initialize_model(model_config, train_loader).to(device)

        model_path = output_dir / f"{model_name}.pth"
        if not model_path.exists():
            print(f"Model not found: {model_path}")
            continue
        
        model.load_state_dict(torch.load(model_path))

        # Evaluate
        y_true, y_pred, y_probability = evaluate_model(model, test_loader, device=device, threshold=0.5)

        model_predictions[model_name] = y_pred

        # Compute metrics
        metrics = compute_metrics(y_true, y_pred, y_probability)
        print(f"Metrics for {model_name}:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        print()

    # Analyze agreement among models
    analyze_model_agreement(y_test_loaded, **model_predictions)

if __name__ == "__main__":
    main()
