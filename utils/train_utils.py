import torch

def train_one_epoch(model, data_loader, criterion, optimizer):
    """
    Train the model for one epoch.

    Args:
        model: The model to be trained.
        data_loader: DataLoader providing the training data.
        criterion: Loss function.
        optimizer: Optimizer.

    Returns:
        Average loss for the epoch.
    """
    model.train()
    total_loss = 0.0

    for batch_idx, (inputs, labels, lengths) in enumerate(data_loader):
        inputs, labels = inputs.to(model.device), labels.to(model.device)

        optimizer.zero_grad()

        logits = model(inputs, lengths)  # Pass inputs through the model

        loss = criterion(logits.squeeze(), labels.float())

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(data_loader)
    return average_loss

def save_model(model, path):
    """
    Save the model to the specified path.

    Args:
        model: The model to be saved.
        path: File path to save the model.
    """
    torch.save(model.state_dict(), path)
