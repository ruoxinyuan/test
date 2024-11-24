import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

def plot_probability_curves(
    prob_curves, 
    labels, 
    max_samples=25, 
    grid_size=(5, 5), 
    figsize=(15, 15), 
    save_path=None
):
    """
    Plot probability curves for classification results.

    Args:
        prob_curves (dict): Dictionary where keys are model names, and values are dictionaries of probability curves by sample ID.
        labels (dict): Dictionary of true labels for each sample ID.
        max_samples (int, optional): Maximum number of samples to plot. Default is 25.
        grid_size (tuple, optional): Grid layout for subplots (rows, columns). Default is (5, 5).
        figsize (tuple, optional): Figure size. Default is (15, 15).
        save_path (str, optional): Path to save the plot as an image file. If None, the plot will not be saved.
    """
    # Extract model names from prob_curves keys
    model_names = list(prob_curves.keys())

    # Select sample IDs
    valid_ids = list(labels.keys())[:max_samples]
    rows, cols = grid_size

    # Initialize the plot
    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    axs = axs.flatten()  # Flatten to simplify indexing

    # Iterate through selected sample IDs
    for i, sample_id in enumerate(valid_ids):
        ax = axs[i]

        # Plot each model's probability curve
        for model_name in model_names:
            model_prob_curves = prob_curves.get(model_name, {})
            prob_curve = model_prob_curves.get(sample_id, [])
            if len(prob_curve) > 0:
                x = np.arange(1, len(prob_curve) + 1)
                ax.plot(x, prob_curve, label=model_name)

        # Retrieve true label for the sample
        label = labels[sample_id]

        # Configure plot
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlabel("Action Step")
        ax.set_ylabel("Predicted Probability")
        ax.set_title(f"Probability Curve for ID {sample_id}")

        # Ensure x-axis values are integers
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Annotate true label
        if len(prob_curve) > 0:
            time_point = len(prob_curve) - 1
            if label == 1:
                ax.annotate(
                    f"True Label: {label}",
                    xy=(time_point, 1.05),
                    xytext=(time_point * 0.6, 1.03),
                    fontsize=10,
                    color="red",
                )
            elif label == 0:
                ax.annotate(
                    f"True Label: {label}",
                    xy=(time_point, -0.05),
                    xytext=(time_point * 0.6, -0.06),
                    fontsize=10,
                    color="red",
                )

        # Add legend only in the first column
        if i % cols == 0:
            ax.legend()

    # Remove empty subplots if max_samples < grid_size[0] * grid_size[1]
    for j in range(i + 1, rows * cols):
        fig.delaxes(axs[j])

    # Adjust layout
    plt.tight_layout()

    # Save the figure if save_path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")

    # Show the figure
    plt.show()


def plot_spline_curves(
    fine_grained_spline,
    prob_curves,
    labels,
    max_samples=25,
    grid_size=(5, 5),
    figsize=(15, 15),
    save_path=None
):
    """
    Plot probability curves along with their fine-grained B-spline fitted curves.

    Args:
        fine_grained_spline (dict): Dictionary containing fine-grained B-spline curves for each sample ID.
        prob_curves (dict): Dictionary where keys are model names, and values are dictionaries of probability curves by sample ID.
        labels (dict): Dictionary of true labels for each sample ID.
        max_samples (int, optional): Maximum number of samples to plot. Default is 25.
        grid_size (tuple, optional): Grid layout for subplots (rows, columns). Default is (5, 5).
        figsize (tuple, optional): Figure size. Default is (15, 15).
        save_path (str, optional): Path to save the plot as an image file. If None, the plot will not be saved.
    """
    # Extract model names from prob_curves keys
    model_names = list(prob_curves.keys())

    # Select sample IDs to plot
    valid_ids = list(labels.keys())[:max_samples]
    rows, cols = grid_size

    # Initialize the plot
    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    axs = axs.flatten()  # Flatten to simplify indexing

    # Iterate through selected sample IDs
    for i, sample_id in enumerate(valid_ids):
        ax = axs[i]

        # Plot each model's probability curve
        for model_name in model_names:
            model_prob_curves = prob_curves.get(model_name, {})
            prob_curve = model_prob_curves.get(sample_id, [])
            if len(prob_curve) > 0:
                x = np.arange(1, len(prob_curve) + 1)
                ax.plot(x, prob_curve, label=model_name)

        # Plot the B-spline fitted curve
        if sample_id in fine_grained_spline:
            x_fine = np.linspace(1, len(prob_curve), 200)
            ax.plot(
                x_fine, fine_grained_spline[sample_id], label="B-spline", color="black"
            )

        # Retrieve true label for the sample
        label = labels[sample_id]

        # Configure plot
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlabel("Action Step")
        ax.set_ylabel("Probability")
        ax.set_title(f"Probability Curve for ID {sample_id}")

        # Annotate true label
        if len(prob_curve) > 0:
            time_point = len(prob_curve) - 1
            if label == 1:
                ax.annotate(
                    f"True Label: {label}",
                    xy=(time_point, 1.05),
                    xytext=(time_point * 0.6, 1.03),
                    fontsize=10,
                    color="red",
                )
            elif label == 0:
                ax.annotate(
                    f"True Label: {label}",
                    xy=(time_point, -0.05),
                    xytext=(time_point * 0.6, -0.06),
                    fontsize=10,
                    color="red",
                )

        # Ensure x-axis values are integers
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Add legend only in the first column
        if i % cols == 0:
            ax.legend()

    # Remove empty subplots if max_samples < grid_size[0] * grid_size[1]
    for j in range(i + 1, rows * cols):
        fig.delaxes(axs[j])

    # Adjust layout
    plt.tight_layout()

    # Save the figure if save_path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")

    # Show the figure
    plt.show()