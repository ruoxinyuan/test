import torch
from sklearn.metrics import accuracy_score, roc_auc_score

def evaluate_model(model, test_loader, device, threshold=0.5):
    """
    Evaluate the model on the test dataset and return true labels, predicted labels, and probabilities.   
    Args:
        model (torch.nn.Module): The model to evaluate.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        device (str): The device to use ('cpu' or 'cuda').
        threshold (float, optional): Threshold for converting probabilities to binary predictions. Default is 0.5.       
    Returns:
        torch.Tensor: True labels (y_true).
        torch.Tensor: Predicted labels (y_pred).
        torch.Tensor: Predicted probabilities (y_probability).
    """
    model.eval()  # Set the model to evaluation mode
    
    y_true, y_pred, y_probability = [], [], []
    
    with torch.no_grad():
        for x, y, lengths in test_loader:
            # Move inputs and labels to the specified device
            x, y, lengths = x.to(device), y.to(device), lengths.to(device)
            
            # Compute logits from the model
            logits = model(x, lengths)
            
            # Apply sigmoid to get probabilities
            probability = torch.sigmoid(logits.squeeze(dim=-1))
            
            # Store true labels, probabilities, and binary predictions
            y_true.append(y.cpu())
            y_probability.append(probability.cpu())
            y_pred.append((probability >= threshold).int().cpu())
    
    # Concatenate results across all batches
    y_true = torch.cat(y_true)
    y_probability = torch.cat(y_probability)
    y_pred = torch.cat(y_pred)
    
    return y_true, y_pred, y_probability

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
