import torch
from torchvision.transforms import ToTensor, Compose, Resize, RandAugment
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset, random_split
import wandb
from model import ConvNet, train
from metrics import CategoricalAccuracy

transforms = Compose([Resize((224, 224)), ToTensor()])
transforms_aug = Compose([Resize((224, 224)), RandAugment(), ToTensor()])
dataset = ImageFolder('../data/train/', transform=transforms)
dataset_aug = ImageFolder('../data/train/', transform=transforms_aug)

device = 'cpu'

def train_run(config=None):
    with wandb.init(config=config) as run:
        config = wandb.config
        run.name = f'{config.filters}_filters_{config.width}_width_{str(config.dropout)+"_dropout" if config.dropout is not None else ""}_{config.actn_conv}_{config.pool_size_final}_poolsize_{config.lr}_lr_{config.weight_decay}_wd{"_bnorm" if config.batch_norm else ""}{"_aug" if config.data_aug else ""}'
        if config.data_aug:
            dataset_total = ConcatDataset([dataset, dataset_aug])
        else:
            dataset_total = dataset
        train_dataset, val_dataset = random_split(dataset_total, lengths=[0.8, 0.2])
        train_dataloader, val_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True), DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)
        if config.actn_conv == 'relu':
            actn_conv = torch.nn.ReLU
        elif config.actn_conv == 'gelu':
            actn_conv = torch.nn.GELU
        elif config.actn_conv == 'silu':
            actn_conv = torch.nn.SiLU
        model = ConvNet(
            filters=[(config.filters, 3)]*5,
            width_dense=config.width,
            activation_conv=actn_conv,
            batch_norm=config.batch_norm,
            dropout=config.dropout,
            pool_size_final=config.pool_size_final
        ).to(device, non_blocking=True)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        metric = CategoricalAccuracy()
        history = train(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            metric=metric,
            epochs=config.epochs,
            verbose=False,
            device=device
        )
        for i in range(config.epochs):
            wandb.log({'epoch': history['epoch'][i], 'loss': history['train_loss'][i], 'accuracy': history['train_score'][i], 'val_loss': history['val_loss'][i], 'val_accuracy': history['val_score'][i]})
    torch.cuda.empty_cache()



sweep_config = {
    'method': 'bayes',
    'name': 'trial_1'
}
sweep_metric = {
    'name': 'val_accuracy',
    'goal': 'maximize'
}
sweep_config['metric'] = sweep_metric
parameters = {
    'width': {
        'values': [64, 128, 256, 512]
    },
    'filters': {
        'values': [32, 64, 128, 256]
    },
    'dropout': {
        'values': [None, 0.2, 0.3]
    },
    'batch_norm': {
        'values': [False, True]
    },
    'data_aug': {
        'values': [False, True]
    },
    'actn_conv': {
        'values': ['relu', 'gelu', 'silu']
    },
    'pool_size_final': {
        'values': [2, 7]
    },
    'lr': {
        'values': [1e-4, 1e-3, 1e-2]
    },
    'weight_decay': {
        'values': [0.0, 1e-5, 1e-3, 1e-1, 1.0]
    },
    'epochs': {
        'value': 10
    },
    'batch_size': {
        'value': 128
    }
}
sweep_config['parameters'] = parameters

sweep_id = wandb.sweep(sweep_config, project='CS6910-A2') 
wandb.agent(sweep_id, train_run, project='CS6910-A2')
