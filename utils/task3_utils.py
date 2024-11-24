import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def perform_clustering(fine_grained_spline, n_clusters=3):
    """
    Perform clustering on fine grained B-spline curves using KMeans.
    Args:
        fine_grained_spline (dict): Dictionary containing fine-grained B-spline curves for each sample ID.
        n_clusters (int): Number of clusters for KMeans.
    Returns:
        dict: A dictionary mapping cluster IDs to lists of sample IDs.
    """
    # Convert curves to a NumPy array
    curves_array = np.array(list(fine_grained_spline.values()))
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(curves_array)
    labels = kmeans.labels_

    # Map clusters to sample IDs
    cluster_groups = {}
    sample_to_cluster = {}
    for idx, cluster_id in zip(fine_grained_spline.keys(), labels):
        if cluster_id not in cluster_groups:
            cluster_groups[cluster_id] = []
        cluster_groups[cluster_id].append(idx)
        sample_to_cluster[idx] = cluster_id

    # Print cluster counts
    unique_clusters, counts = np.unique(labels, return_counts=True)
    print("\nCluster counts:")
    for cluster, count in zip(unique_clusters, counts):
        print(f"Cluster {cluster}: {count} samples")

    return cluster_groups


def plot_cluster_curves(
    fine_grained_spline,
    prob_curves,
    cluster_groups,
    max_samples_per_cluster=25,
    grid_size=(5, 5),
    figsize=(15, 15),
    save_path=None,
):
    """
    Plot probability curves for each cluster along with their fine-grained B-spline fitted curves.
    Args:
        fine_grained_spline (dict): Dictionary containing fine-grained B-spline curves for each sample ID.
        prob_curves (dict): Dictionary where keys are model names, and values are dictionaries of probability curves by sample ID.
        cluster_groups (dict): Dictionary where keys are cluster IDs, and values are lists of sample IDs in that cluster.
        max_samples_per_cluster (int, optional): Maximum number of samples to plot per cluster. Default is 25.
        grid_size (tuple, optional): Grid layout for subplots (rows, columns). Default is (5, 5).
        figsize (tuple, optional): Figure size. Default is (15, 15).
        save_path (str, optional): Path to save the plot as an image file. If None, the plot will not be saved.
    """
    # Extract model names from prob_curves keys
    model_names = list(prob_curves.keys())
    rows, cols = grid_size

    # Iterate through each cluster
    for cluster_id, sample_ids in cluster_groups.items():
        # Select up to max_samples_per_cluster sample IDs to plot
        valid_ids = sample_ids[:max_samples_per_cluster]

        # Initialize the plot
        fig, axs = plt.subplots(rows, cols, figsize=figsize)
        axs = axs.flatten()  # Flatten to simplify indexing

        # Iterate through selected sample IDs in the cluster
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

            # Configure plot
            ax.set_ylim(0, 1)
            ax.set_xlabel("Action Step")
            ax.set_ylabel("Probability")
            ax.set_title(f"Cluster {cluster_id}, ID {sample_id}")

            # Ensure x-axis values are integers
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            # Add legend only in the first column
            if i % cols == 0:
                ax.legend()

        # Remove empty subplots if max_samples_per_cluster < grid_size[0] * grid_size[1]
        for j in range(i + 1, rows * cols):
            fig.delaxes(axs[j])

        # Adjust layout
        plt.tight_layout()

        # Save the figure if save_path is provided
        if save_path:
            cluster_save_path = f"{save_path}_cluster_{cluster_id}.png"
            plt.savefig(cluster_save_path)
            print(f"Cluster {cluster_id} figure saved to {cluster_save_path}")

        # Show the figure
        plt.show()

