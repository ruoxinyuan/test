import yaml
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
import sys
from pathlib import Path

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import models
from models import GRUClassifier, LSTMClassifier, LogisticRegressionClassifier, TransformerClassifier

# Import utilities
from utils.curve_utils import compute_probability_curve, save_probability_curves, fit_bspline_curves, save_spline_curves
from utils.plot_utils import plot_probability_curves, plot_spline_curves

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
    output_dir = Path(paths_config["output"])
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
    train_loader = DataLoader(
        TensorDataset(X_train_loaded, y_train_loaded, lengths_train_loaded),
        batch_size=training_config["batch_size"],
        shuffle=True,
    )

    X_test_loaded = torch.load(data_dir / "X_test.pt")
    y_test_loaded = torch.load(data_dir / "y_test.pt")
    lengths_test_loaded = torch.load(data_dir / "lengths_test.pt") 

    # Compute probability curves for each model in the configuration
    prob_curves = {}
    labels = {}
    for model_config in model_configs:
        model_name = model_config["name"]
        print(f"Processing {model_name} model...")

        # Initialize model and move to device
        model = initialize_model(model_config, train_loader).to(device)
        model.eval()

        # Compute probability curves for sequences longer than 10
        model_prob_curves = {}
        for i, (test_seq, test_label, test_len) in enumerate(zip(X_test_loaded, y_test_loaded, lengths_test_loaded)):
            if test_len > 10:
                model_prob_curves[i] = compute_probability_curve(model, test_seq, test_len, device)
                labels[i] = y_test_loaded[i].numpy()

        prob_curves[model_name] = model_prob_curves
        

    # Save probability curves
    save_path1 = output_dir / "probability_curves.npz"
    save_probability_curves(prob_curves, labels, save_path1)
    print(f"Probability curves saved to {save_path1}")

    # Plot raw probability curves
    plot_probability_curves(
        prob_curves=prob_curves, 
        labels=labels, 
        max_samples=25, 
        grid_size=(5, 5), 
        figsize=(12, 12), 
        save_path=output_dir / "probability_curves.png"
    )

    # Fit B-spline curves and save results
    print("B-spline fitting ...")
    fine_grained_spline, spline_curve = fit_bspline_curves(prob_curves, labels)
        
    # Save B-spline curves
    save_path2 = output_dir / "B-spline_curves.npz"
    save_spline_curves(fine_grained_spline, spline_curve, save_path2)
    print(f"B-spline curves saved to {save_path2}")

    # Plot B-spline curves and probability curves
    plot_spline_curves(
        fine_grained_spline=fine_grained_spline,
        prob_curves=prob_curves, 
        labels=labels, 
        max_samples=25, 
        grid_size=(5, 5), 
        figsize=(12, 12), 
        save_path=output_dir / "spline_curves.png"
    )


if __name__ == "__main__":
    main()
