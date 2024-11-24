import sys
import os
from pathlib import Path
import torch
import yaml

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import utilities
from utils.curve_utils import load_spline_curves, load_probability_curves
from utils.task1_utils import calculate_guessing_parameter
from utils.task2_utils import process_action_diffs, process_2gram_diffs
from utils.task3_utils import perform_clustering, plot_cluster_curves

# Load configuration
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def main():
    # Load configuration
    config = load_config("config.yaml")
    paths_config = config["paths"]

    # Load test data
    data_dir = Path(paths_config["data"])
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    X_test_loaded = torch.load(data_dir / "X_test.pt")

    # Load spline curves
    curve_dir = Path(paths_config["output"])
    if not curve_dir.exists():
        raise FileNotFoundError(f"curve directory not found: {curve_dir}")
    prob_curves, labels = load_probability_curves(curve_dir / "probability_curves.npz")
    fine_grained_spline, spline_curve = load_spline_curves(curve_dir / "B-spline_curves.npz")

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
        save_path=curve_dir / "cluster_plot",
    )


if __name__ == "__main__":
    main()