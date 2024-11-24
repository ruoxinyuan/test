import torch
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import BSpline

def prob_curves_for_long_seqs(model, X_test, y_test, lengths, device, min_length=10):
    """
    Compute probability curves for sequences longer than a specified min_length.   
    Args:
        X_test (torch.Tensor): Test sequences, a tensor of shape (num_samples, seq_len).
        y_test (torch.Tensor): True labels for the test sequences.
        lengths (torch.Tensor): Lengths of the test sequences.
        min_length (int, optional): Minimum sequence length to process. Default is 10.  
    Returns:
        dict: A dictionary containing probability curves, keyed by sequence index.
        dict: A dictionary containing labels for the sequences with probability curves.
    """
    
    def compute_probability_curve(model, test_seq, seq_len, device):
        """
        Compute the probability curves for the model.
        Args:
            test_seq (torch.Tensor): Test sequence with shape (2, seq_len).
            seq_len (int): Length of the test sequence.
            device (str): Device for computation ('cpu' or 'cuda').
        Returns:
            np.array: Probability curve with shape (seq_len, ).
        """
        prob_curve = []
        for t in range(1, seq_len + 1):
            current_seq = test_seq[:t].unsqueeze(0).to(device)  # Add batch dimension
            with torch.no_grad():
                logits = model(current_seq, torch.tensor([t], device=device))
                probs = torch.sigmoid(logits)
            prob_curve.append(probs.squeeze().cpu().numpy())
        return np.array(prob_curve)
    
    model.eval()  # Ensure the model is in evaluation mode
    model_prob_curves = {}
    labels = {}

    # Loop through test sequences
    for i, (test_seq, test_label, test_len) in enumerate(zip(X_test, y_test, lengths)):
        if test_len > min_length:
            # Compute probability curve for the current sequence
            model_prob_curves[i] = compute_probability_curve(model, test_seq, test_len, device)
            # Store the corresponding label
            labels[i] = test_label.numpy()

    return model_prob_curves, labels


def save_probability_curves(prob_curves, labels, save_path):
    """
    Save probability curves to a file.
    Args:
        prob_curves (dict): Dictionary of probability curves for different models.
        labels (dict): Dictionary of true labels for samples.
        save_path (str): Path to save the file (supports .npz format).
    """
    np.savez_compressed(save_path, prob_curves=prob_curves, labels=labels, allow_pickle=True)


def load_probability_curves(load_path):
    """
    Load probability curves from a file.
    Args:
        load_path (str): Path to the .npz file.
    Returns:
        dict, dict: Probability curves and labels as dictionaries.
    """
    data = np.load(load_path, allow_pickle=True)
    prob_curves = data["prob_curves"].item()
    labels = data["labels"].item()
    return prob_curves, labels


def fit_bspline_curves(prob_curves, labels, degree=3, knot_interval=3):
    """
    Fit B-spline curves for a given set of probability curves.
    Args:
        prob_curves (dict): Dictionary where keys are model names, and values are dictionaries of probability curves by sample ID.
        labels (dict): Dictionary containing labels for corresponding IDs.
        degree (int, optional): Degree of the B-spline. Default is 3 (cubic).
        knot_interval (int, optional): Interval for placing knots along the x-axis. Default is 3.
    Returns:
        fine_grained_spline (dict): A dictionary containing fine-grained B-spline curves for each sample ID.
        spline_curve (dict): A dictionary containing fitted B-spline values at original points for each sample ID.
    """
    fine_grained_spline = {}
    spline_curve = {}

    # Define loss function for B-spline fitting
    def loss_function(control_points, knots, x, y_list):
        # Construct B-spline
        spl = BSpline(knots, control_points, degree)
        # Compute loss as the sum of squared errors across all probability curves
        loss = sum(np.sum((spl(x) - y)**2) for y in y_list)
        return loss

    # Group probability curves by sample ID
    sample_ids = labels.keys()
    prob_curves_by_id = {id_: [] for id_ in sample_ids}

    for model_name, model_curves in prob_curves.items():
        for id_, curve in model_curves.items():
            if id_ in prob_curves_by_id:
                prob_curves_by_id[id_].append(curve)

    # Iterate through sample IDs
    for id_, y_list in prob_curves_by_id.items():
        # Ensure y_list is a list of probability curves
        if not isinstance(y_list, list) or len(y_list) < 1:
            continue

        x = np.arange(1, len(y_list[0]) + 1)  # Original x-axis points
        x_fine = np.linspace(1, len(x), 200)  # Fine-grained x-axis points

        # Define knots (evenly spaced based on knot_interval)
        knots = np.concatenate(([1], x[::knot_interval], [len(x), len(x), len(x)]))

        # Initial control points
        initial_control_points = np.random.rand(len(knots) - (degree + 1))  # Control points depend on knot count

        # Optimize control points for B-spline fitting
        result = minimize(
            loss_function, 
            initial_control_points, 
            args=(knots, x, y_list), 
            method='BFGS'
        )

        # Construct optimized B-spline
        optimized_control_points = result.x
        spline = BSpline(knots, optimized_control_points, degree)

        # Save fitted curves
        spline_curve[id_] = spline(x)  # Fitted values at original x points
        fine_grained_spline[id_] = spline(x_fine)  # Fine-grained curve

    return fine_grained_spline, spline_curve

def save_spline_curves(fine_grained_spline, spline_curve, file_path):
    """
    Save fine-grained spline and fitted spline curve data to a Numpy (.npz) file.
    Args:
        fine_grained_spline (dict): Dictionary containing fine-grained B-spline curves for each sample ID.
        spline_curve (dict): Dictionary containing fitted B-spline values at original points for each sample ID.
        file_path (str): Path to save the .npz file.
    """
    # Convert dictionaries to arrays of tuples (ID, values) for easier storage
    fine_grained_array = np.array(list(fine_grained_spline.items()), dtype=object)
    spline_curve_array = np.array(list(spline_curve.items()), dtype=object)
    
    # Save to .npz file
    np.savez(file_path, fine_grained_spline=fine_grained_array, spline_curve=spline_curve_array, allow_pickle=True)

def load_spline_curves(file_path):
    """
    Load fine-grained spline and fitted spline curve data from a Numpy (.npz) file.
    Args:
        file_path (str): Path to the .npz file.
    Returns:
        fine_grained_spline (dict): Dictionary containing fine-grained B-spline curves for each sample ID.
        spline_curve (dict): Dictionary containing fitted B-spline values at original points for each sample ID.
    """
    # Load the .npz file
    data = np.load(file_path, allow_pickle=True)

    # Convert arrays of tuples back to dictionaries
    fine_grained_spline = dict(data["fine_grained_spline"])
    spline_curve = dict(data["spline_curve"])

    return fine_grained_spline, spline_curve
