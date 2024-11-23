import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import yaml

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import models
from models import GRUClassifier, LSTMClassifier, LogisticRegressionClassifier, TransformerClassifier

# Import utilities
from utils.train_utils import train_one_epoch, save_model

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
    
def main():
    # Load configuration
    config = load_config("config.yaml")
    training_config = config["training"]
    model_configs = config["models"]
    paths_config = config["paths"]

    # Ensure the output directory exists
    output_dir = Path(paths_config["output_model"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set device
    device = torch.device(training_config["device"] if torch.cuda.is_available() else "cpu")

    # Load data
    data_dir = Path(paths_config["data"])
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    X_train_loaded = torch.load(data_dir / "X_train.pt")
    y_train_loaded = torch.load(data_dir / "y_train.pt")
    lengths_train_loaded = torch.load(data_dir / "lengths_train.pt") 

    # Create DataLoader
    train_loader = DataLoader(
        TensorDataset(X_train_loaded, y_train_loaded, lengths_train_loaded),
        batch_size=training_config["batch_size"],
        shuffle=True,
    )

    # Train each model in the configuration
    for model_config in model_configs:
        model_name = model_config["name"]
        print(f"Training {model_name} model...")

        # Initialize model and move to device
        model = initialize_model(model_config, train_loader).to(device)

        # Define loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=model_config["learning_rate"])

        # Training loop
        for epoch in range(model_config["num_epochs"]):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            print(f"{model_name} - Epoch {epoch + 1}, Loss: {train_loss:.4f}")

        # Save model
        model_save_path = output_dir / f"{model_name}.pth"
        save_model(model, model_save_path)
        print(f"Model {model_name} saved to {model_save_path}.\n")

    print(f"Training completed for all models.")

if __name__ == "__main__":
    main()
