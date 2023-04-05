import torch
from torch import nn
from typing import List, Tuple, Dict

class ConvNet(nn.Module):
    """
    Implements a simple convolutional neural network, with 5 convolutional layers, each followed by maxpooling and optionally batchnorm, and two fully-connected layers with dropout optionally present between them.

    Args:
        filters (List[Tuple[int, int]]): Details of all the convolutional layers, each given by a tuple (num_filters, kernel_size), where num_filters is the number of convolutional filters and kernel_size is the size of the filter.
        width_dense (int): The number of units in the hidden fully-connected/dense layer.
        input_size (Tuple[int, int], optional): The size of the input images (images are assumed RGB, i.e., 3 channels).
            Defaults to (224, 224).
        activation_conv (torch.nn.Module, optional): The activation/non-linearity to use for the convolutional layers.
            Defaults to torch.nn.ReLU.
        activation_conv (torch.nn.Module, optional): The activation/non-linearity to use for the hidden dense layer.
            Defaults to torch.nn.ReLU.
        batch_norm (bool, optional): Whether or not to apply batch normalization.
            Defaults to True.
        dropout (float, optional): The dropout rate to use while training the network.
            Defaults to None.
        pool_size_final (int, optional): The kernel size and stride for the final maxpool layer in the convnet.
            Defaults to 2.
    """
    def __init__(
        self,
        filters: List[Tuple[int, int]],
        width_dense: int,
        input_size: Tuple[int, int] = (224, 224),
        activation_conv: nn.Module = nn.ReLU,
        activation_dense: nn.Module = nn.ReLU,
        batch_norm: bool = True,
        dropout: float = None,
        pool_size_final: int = 2
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=filters[0][0], kernel_size=filters[0][1], padding='same'),
            activation_conv(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=filters[0][0]) if batch_norm else nn.Identity(),
            nn.Conv2d(in_channels=filters[0][0], out_channels=filters[1][0], kernel_size=filters[1][1], padding='same'),
            activation_conv(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=filters[1][0]) if batch_norm else nn.Identity(),
            nn.Conv2d(in_channels=filters[1][0], out_channels=filters[2][0], kernel_size=filters[2][1], padding='same'),
            activation_conv(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=filters[2][0]) if batch_norm else nn.Identity(),
            nn.Conv2d(in_channels=filters[2][0], out_channels=filters[3][0], kernel_size=filters[3][1], padding='same'),
            activation_conv(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=filters[3][0]) if batch_norm else nn.Identity(),
            nn.Conv2d(in_channels=filters[3][0], out_channels=filters[4][0], kernel_size=filters[4][1], padding='same'),
            activation_conv(),
            nn.MaxPool2d(kernel_size=pool_size_final, stride=pool_size_final),
            nn.BatchNorm2d(num_features=filters[4][0]) if batch_norm else nn.Identity(),
        )
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Identity() if dropout is None else nn.Dropout(p=dropout),
            nn.Linear(in_features=int(filters[4][0]*(input_size[0] * input_size[1])/(16*pool_size_final)**2), out_features=width_dense),
            activation_dense(),
            nn.Identity() if dropout is None else nn.Dropout(p=dropout),
            nn.Linear(in_features=width_dense, out_features=10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.dense(self.conv(x))

def train(
    model: ConvNet,
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
    Trains a ConvNet according to the specified parameters.

    Args:
        model (ConvNet): The model to train.
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
