import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Resize, RandAugment, Compose
from torch.utils.data import DataLoader, ConcatDataset, random_split
from metrics import CategoricalAccuracy
from learn import train, predict
import wandb
from argparse import ArgumentParser

args_parser = ArgumentParser()

args_parser.add_argument('-da', '--data_aug', action='store_true', help='Whether to add data augmentation to the training data')
args_parser.add_argument('-bs', '--batch_size', type=int, default=128, help='Batch size to use for DataLoader')
args_parser.add_argument('-dev', '--device', type=str, choices=['cpu', 'cuda'], default='cuda', help='Device to perform training on')
args_parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='Learning rate to use for optimization')
args_parser.add_argument('-wd', '--weight_decay', type=float, default=1e-3, help='Weight decay to use for regularization during training')
args_parser.add_argument('-e', '--epochs', type=int, default=5, help='Number of epochs to train for')
args_parser.add_argument('-we', '--wandb_entity', type=str, default=None, help='WandB entity where the run (if logged) is tracked')
args_parser.add_argument('-wp', '--wandb_project', type=str, default=None, help='WandB project under which the run (if logged) is tracked. Not logged if None is passed')
args_parser.add_argument('-run', '--run_test', action='store_true', help='Whether to use the trained model to predict on test data')

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

model = resnet50(weights=ResNet50_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad_(False)
model.fc = torch.nn.Linear(model.fc.in_features, 10)
model = model.to(device, non_blocking=True)

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
        run.name = f'resnet_{args.learning_rate}_lr_{args.weight_decay}_wd_{args.epochs}_epochs{"_aug" if args.data_aug else ""}'
        for i in range(args.epochs):
            wandb.log({'epoch': history['epoch'][i], 'loss': history['train_loss'][i], 'accuracy': history['train_score'][i], 'val_loss': history['val_loss'][i], 'val_accuracy': history['val_score'][i]})

if args.run_test:
    dataset_test = ImageFolder('../data/val/', transform=transforms)
    full_dataloader = DataLoader(dataset, batch_size=len(dataset_test))
    X_test, Y_test = next(iter(full_dataloader))
    Y_pred = predict(model, X_test, batch_size=args.batch_size, device=device)
    metric = CategoricalAccuracy()
    score = metric(Y_pred, Y_test)
    print(f'Score on test data: {score}')
