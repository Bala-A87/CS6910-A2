import torch

class CategoricalAccuracy(torch.nn.Module):
    """
    Implements the categorical accuracy scoring metric.

    Args:
        y_pred (torch.Tensor): The predictions (one-hot) of size (num_samples, num_classes).
        y_true (torch.Tensor): The true class labels, of size (num_samples,).
    """
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, y_pred, y_true):
        return (y_pred.detach().cpu().argmax(dim=1) == y_true.cpu()).sum() / len(y_true.cpu())
    