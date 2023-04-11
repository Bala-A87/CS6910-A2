import torch
from typing import Dict, List
from math import ceil

def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metric: torch.nn.Module,
    epochs: int = 10,
    verbose: bool = True,
    device: torch.device = 'cpu'
) -> Dict[str, List]:
    """
    Trains a torch neural network according to the specified parameters.

    Args:
        model (torch.nn.Module): The model to train.
        train_dataloader (torch.utils.data.DataLoader): Dataloader for the training data.
        val_dataloader (torch.utils.data.DataLoader): Dataloader for the validation data.
        loss_fn (torch.nn.Module): The loss function to use for optimization.
        optimizer (torch.optim.Optimizer): The optimizer to use for learning the parameters of `model`.
        metric (torch.nn.Module): The scoring metric used to evaluate the model.
        epochs (int, optional): The number of epochs to train the model for.
            Defaults to 10.
        verbose (bool, optional): Whether information should be printed during training.
            Defaults to True.
        device (torch.device, optional): The device to perform the training on.
            Defaults to cpu.
    
    Returns:
        history (Dict[str, List]): A history dictionary specifying losses and scores during each epoch of training.
            Valid keys: epoch, train_loss, train_score, val_loss, val_score
    """
    history = {
        'epoch': [],
        'train_loss': [],
        'train_score': [],
        'val_loss': [],
        'val_score': []
    }
    for epoch in range(epochs):
        train_loss, val_loss = 0.0, 0.0
        train_score, val_score = 0.0, 0.0
        model.train()
        for (X, y) in train_dataloader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            pred_train = model(X)
            loss_curr = loss_fn(pred_train, y)
            score_curr = metric(pred_train, y)
            optimizer.zero_grad()
            loss_curr.backward()
            optimizer.step()
            train_loss += loss_curr.detach().cpu()
            train_score += score_curr.cpu()
        train_loss /= len(train_dataloader)
        train_score /= len(train_dataloader)

        model.eval()
        with torch.inference_mode():
            for (X, y) in val_dataloader:
                X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
                pred_val = model(X)
                val_loss += loss_fn(pred_val, y)
                val_score += metric(pred_val, y)
            val_loss /= len(val_dataloader)
            val_score /= len(val_dataloader)
        
        history['epoch'].append(epoch+1)
        history['train_loss'].append(train_loss)
        history['train_score'].append(train_score)
        history['val_loss'].append(val_loss)
        history['val_score'].append(val_score)

        if verbose:
            print(f'[{epoch+1}/{epochs}] ==> Train loss: {train_loss:.6f}, Train score: {train_score:.4f}, Val loss: {val_loss:.6f}, Val score: {val_score:.4f}')
        
    print(f'Training complete. Train loss: {history["train_loss"][-1]:.6f}, Train score: {history["train_score"][-1]:.4f}, Val loss: {history["val_loss"][-1]:.6f}, Val score: {history["val_score"][-1]:.4f}')

    return history

def predict(
    model: torch.nn.Module,
    X: torch.Tensor,
    batch_size: int = 128,
    device: torch.device = 'cpu'
) -> torch.Tensor:
    """
    Predicts labels for given data using the given torch neural network.

    Args:
        model (torch.nn.Module): The model to use for prediction.
        X (torch.Tensor): The data for which to make the predictions.
        batch_size (int, optional): The batch size to use while computing the predictions.
            Defaults to 128.
        device (torch.device, optional): The computing device on which to predict.
            Defaults to cpu.
    """
    preds = torch.tensor([]).to(device, non_blocking=True)
    batches = range(int(ceil(len(X / batch_size))))
    model.eval()
    with torch.inference_mode():
        for batch in batches:
            X_sub = X[batch*batch_size : min((batch+1)*batch_size, len(X))].to(device, non_blocking=True)
            preds_batch = model(X_sub)
            preds = torch.cat([preds, preds_batch]).to(device, non_blocking=True)
    return preds
