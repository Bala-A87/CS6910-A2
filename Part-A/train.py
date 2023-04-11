from model import ConvNet, train, predict
from metrics import CategoricalAccuracy
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Resize, RandAugment, Compose
from torch.utils.data import DataLoader, ConcatDataset, random_split
import wandb
from argparse import ArgumentParser
from typing import List
import json
import os
from pathlib import Path

def get_activation(activation_str: str) -> torch.nn.Module:
    if activation_str == 'gelu':
        return torch.nn.GELU
    elif activation_str == 'silu':
        return torch.nn.SiLU
    elif activation_str == 'mish':
        return torch.nn.Mish
    elif activation_str == 'sigmoid':
        return torch.nn.Sigmoid
    elif activation_str == 'tanh':
        return torch.nn.Tanh
    else:
        return torch.nn.ReLU

args_parser = ArgumentParser()
args_parser.add_argument('-da', '--data_aug', action='store_true', help='Whether to add data augmentation to the training data')
args_parser.add_argument('-bs', '--batch_size', type=int, default=128, help='Batch size to use for DataLoader')
args_parser.add_argument('-dev', '--device', type=str, choices=['cpu', 'cuda'], default='cuda', help='Device to perform training on')
args_parser.add_argument('-f', '--filters', type=json.loads, default=[64]*5, help='Number of filters in each convolutional layer')
args_parser.add_argument('-k', '--kernel_size', type=json.loads, default=[3]*5, help='Size of convolutional filter in each convolutional layer')
args_parser.add_argument('-w', '--width', type=int, default=64, help='Number of neurons in the fully-connected hidden layer')
args_parser.add_argument('-do', '--dropout', type=float, default=None, help='Dropout to use in the dense layer')
args_parser.add_argument('-ac', '--activation_conv', type=str, default='relu', help='Activation function to use for the convolutional layers') # add choices
args_parser.add_argument('-ad', '--activation_dense', type=str, default='relu', help='Activation function to use for the dense hidden layer') # add choices
args_parser.add_argument('-ps', '--pool_size', type=int, default=2, help='Size of maxpooing filter for the last convolutional layer')
args_parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='Learning rate to use for optimization')
args_parser.add_argument('-wd', '--weight_decay', type=float, default=0.0, help='Weight decay to use for regularization during training')
args_parser.add_argument('-bn', '--batch_norm', action='store_true', help='Whether to use batch normalization in the network')
# Maybe change to make it use by default ^
args_parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs to train for')
# Add wandb args too
args_parser.add_argument('-we', '--wandb_entity', type=str, default=None, help='WandB entity where the run (if logged) is tracked')
args_parser.add_argument('-wp', '--wandb_project', type=str, default=None, help='WandB project under which the run (if logged) is tracked. Not logged if None is passed')
args_parser.add_argument('-run', '--run_test', action='store_true', help='Whether to use the trained model to predict on test data')
args_parser.add_argument('-save', '--save_model', action='store_true', help="Whether to save the trained model's state_dict")

args = args_parser.parse_args()

transforms = Compose([Resize((224, 224)), ToTensor()])
transforms_aug = Compose([Resize((224, 224)), RandAugment(), ToTensor()])
dataset = ImageFolder('../data/train/', transform=transforms)
dataset_aug = ImageFolder('../data/train/', transform=transforms_aug)

device = 'cpu'
if args.device == 'cuda':
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        print('WARNING: CUDA device not found. Computing on cpu.')
    
dataset_total = ConcatDataset([dataset, dataset_aug]) if args.data_aug else dataset
train_dataset, val_dataset = random_split(dataset_total, lengths=[0.8, 0.2])
train_dataloader, val_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True), DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

activation_conv = get_activation(args.activation_conv)
activation_dense = get_activation(args.activation_dense)

model = ConvNet(
    filters=list(zip(args.filters, args.kernel_size)),
    width_dense=args.width,
    activation_conv=activation_conv,
    activation_dense=activation_dense,
    batch_norm=args.batch_norm,
    dropout=args.dropout,
    pool_size_final=args.pool_size
).to(device, non_blocking=True)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
metric = CategoricalAccuracy()

history = train(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    metric=metric,
    epochs=args.epochs,
    verbose=True,
    device=device
)

if args.wandb_project is not None:
    with wandb.init(entity=args.wandb_entity, project=args.wandb_project) as run:
        run.name = f'{args.filters}_filters_{args.width}_width_{str(args.dropout)+"_dropout" if args.dropout is not None else ""}_{args.activation_conv}_{args.pool_size}_poolsize_{args.learning_rate}_lr_{args.weight_decay}_wd{"_bnorm" if args.batch_norm else ""}{"_aug" if args.data_aug else ""}'
        for i in range(args.epochs):
            wandb.log({'epoch': history['epoch'][i], 'loss': history['train_loss'][i], 'accuracy': history['train_score'][i], 'val_loss': history['val_loss'][i], 'val_accuracy': history['val_score'][i]})

if args.save_model:
    save_dir = Path('./models/')
    if not save_dir.is_dir():
        os.mkdir(save_dir)
    save_file_name = f'./models/{args.filters}_filters_{args.width}_width_{str(args.dropout)+"_dropout" if args.dropout is not None else ""}_{args.activation_conv}_{args.pool_size}_poolsize_{args.learning_rate}_lr_{args.weight_decay}_wd{"_bnorm" if args.batch_norm else ""}{"_aug" if args.data_aug else ""}'
    torch.save(model.state_dict(), f=save_file_name)

if args.run_test:
    dataset_test = ImageFolder('../data/val/', transform=transforms)
    # Get the entire test data
    full_dataloader = DataLoader(dataset, batch_size=len(dataset_test))
    X_test, Y_test = next(iter(full_dataloader))
    # then use predict to get the predictions
    Y_pred = predict(model, X_test, batch_size=args.batch_size, device=device)
    # then use metric to score it 
    metric = CategoricalAccuracy()
    score = metric(Y_pred, Y_test)
    print(f'Score on test data: {score}')
