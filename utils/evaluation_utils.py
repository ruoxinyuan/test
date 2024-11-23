from sklearn.metrics import accuracy_score, roc_auc_score

def compute_metrics(y_true, y_pred, y_probability):
    """
    Compute evaluation metrics for binary classification.

    Args:
        y_true: List of true labels.
        y_pred: List of predicted labels.

    Returns:
        Dictionary of evaluation metrics.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_probability)
    }
    return metrics
