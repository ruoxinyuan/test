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
from utils.train_utils import train_one_epoch, save_model
from utils.evaluation_utils import evaluate_model, compute_metrics
from utils.curve_utils import prob_curves_for_long_seqs, save_probability_curves, fit_bspline_curves, save_spline_curves
from utils.plot_utils import plot_probability_curves, plot_spline_curves
from utils.task1_utils import calculate_guessing_parameter
from utils.task2_utils import process_action_diffs, process_2gram_diffs
from utils.task3_utils import perform_clustering, plot_cluster_curves

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

    test_loader = DataLoader(
        TensorDataset(X_test_loaded, y_test_loaded, lengths_test_loaded),
        batch_size=testing_config["batch_size"],
        shuffle=False,
    )


    # Train & test each model in the configuration
    # Compute probability curves for each model in the configuration
    model_predictions = {}
    prob_curves = {}
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

        # Evaluate
        print(f"Evaluating {model_name} model...")
        y_true, y_pred, y_probability = evaluate_model(model, test_loader, device=device, threshold=0.5)

        model_predictions[model_name] = y_pred

        # Compute metrics
        metrics = compute_metrics(y_true, y_pred, y_probability)
        print(f"Metrics for {model_name}:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        print()

        # Compute probability curves for sequences longer than 10
        print(f"Computing probability curves for {model_name} model...")
        model_prob_curves, labels = prob_curves_for_long_seqs(model, X_test_loaded, y_test_loaded, lengths_test_loaded, device, min_length=10)

        prob_curves[model_name] = model_prob_curves

    # Analyze agreement among models
    analyze_model_agreement(y_test_loaded, **model_predictions)
    
    # Plot raw probability curves
    plot_probability_curves(
        prob_curves=prob_curves, 
        labels=labels, 
        max_samples=25, 
        grid_size=(5, 5), 
        figsize=(12, 12), 
        # save_path=output_dir / "probability_curves.png"
    )

    # Fit B-spline curves and save results
    print("B-spline fitting ...")
    fine_grained_spline, spline_curve = fit_bspline_curves(prob_curves, labels)

    # Plot B-spline curves and probability curves
    plot_spline_curves(
        fine_grained_spline=fine_grained_spline,
        prob_curves=prob_curves, 
        labels=labels, 
        max_samples=25, 
        grid_size=(5, 5), 
        figsize=(12, 12), 
        # save_path=output_dir / "spline_curves.png"
    )

    # Task 1: calculate guessing parameter
    calculate_guessing_parameter(spline_curve)

    # Task 2: identify key 1-grams and 2-grams
    process_action_diffs(X_test_loaded, spline_curve)
    process_2gram_diffs(X_test_loaded, spline_curve)

    # Task 3: cluster and plot
    cluster_groups = perform_clustering(fine_grained_spline, n_clusters=3)

    plot_cluster_curves(
        fine_grained_spline,
        prob_curves,
        cluster_groups,
        max_samples_per_cluster=25,
        grid_size=(5, 5),
        figsize=(15, 15),
        # save_path=output_dir / "cluster_plot"
    )


if __name__ == "__main__":
    main()
